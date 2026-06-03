"""
Multi-project layer for the heating forecaster.

A *project* is one self-contained city/network living under ``projects/<id>/``::

    projects/
      flensburg/
        config.json
        data/processed/demand_with_weather.csv
        models/registry.json
        models/multi_horizon/   (meta.json + lgbm_h01..h48.pkl)
        models/versions/        (custom retrained versions)

The :class:`Project` class resolves all per-project paths and config so the
rest of the codebase never hardcodes a single city.  This is the structural
foundation for the upcoming "forecast builder" wizard that will add new
projects from uploaded data (a later phase).
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone as _dt_timezone
from pathlib import Path

import pandas as pd

# Repo root resolved robustly from this file's location (src/ → repo root).
REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECTS_DIR = REPO_ROOT / "projects"

DEFAULT_PROJECT_ID = "flensburg"

# Lifecycle stages a project can be in.
#   awaiting_weather  : skeleton created by the setup wizard — demand only, no
#                       weather enrichment or trained models yet (Phase B output).
#   awaiting_training : weather has been enriched (demand_with_weather.csv exists)
#                       but no models are trained yet (Phase C output).
#   ready             : has demand+weather data and at least one trained model.
STAGE_AWAITING_WEATHER = "awaiting_weather"
STAGE_AWAITING_TRAINING = "awaiting_training"
STAGE_READY = "ready"


class Project:
    """A self-contained forecasting project (one city / heating network)."""

    def __init__(self, config: dict, root: Path) -> None:
        self.root = Path(root)
        self._config = config

        self.id: str = config["id"]
        self.name: str = config.get("name", self.id)
        self.city: str = config.get("city", self.name)
        self.lat: float = float(config["lat"])
        self.lon: float = float(config["lon"])
        self.timezone: str = config.get("timezone", "UTC")
        self.demand_unit: str = config.get("demand_unit", "MW")
        self.target_column: str = config.get("target_column", "heat_demand_mw")
        self.created_at: str | None = config.get("created_at")
        self.stage: str | None = config.get("stage")
        self.test_start: str | None = config.get("test_start")
        self.country_code: str | None = (
            config.get("country_code") or config.get("country")
        )
        # Multi-substation fields (absent / empty in single-mode projects).
        # mode: 'single' (default) | 'multi'
        self.mode: str = config.get("mode", "single")
        # substations: [{"id": "FS-001", "name": "Substation 001", "column": "sub_001"}, …]
        self.substations: list[dict] = config.get("substations", [])

    # ── Mode helpers ────────────────────────────────────────────────────────────

    def is_multi(self) -> bool:
        """True for multi-substation projects (90 individual forecasters)."""
        return self.mode == "multi" and bool(self.substations)

    # ── Resolved paths ─────────────────────────────────────────────────────────

    @property
    def data_path(self) -> Path:
        """Processed demand+weather CSV for this project."""
        return self.root / "data" / "processed" / "demand_with_weather.csv"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def default_pkl_dir(self) -> Path:
        """Directory holding the default 48 PKLs + meta.json."""
        return self.models_dir / "multi_horizon"

    @property
    def registry_path(self) -> Path:
        return self.models_dir / "registry.json"

    @property
    def default_meta_path(self) -> Path:
        return self.default_pkl_dir / "meta.json"

    @property
    def versions_dir(self) -> Path:
        return self.models_dir / "versions"

    @property
    def demand_only_path(self) -> Path:
        """Demand-only CSV written by the setup wizard before weather enrichment."""
        return self.root / "data" / "processed" / "demand_only.csv"

    # ── Per-substation model paths (multi-mode only) ────────────────────────────

    def substation_models_dir(self) -> Path:
        """Root directory that holds one sub-dir per substation model set."""
        return self.models_dir / "substations"

    def substation_model_dir(self, sub_id: str) -> Path:
        """``models/substations/<sub_id>/`` — contains multi_horizon/, versions/, registry.json."""
        return self.substation_models_dir() / sub_id

    def substation_default_pkl_dir(self, sub_id: str) -> Path:
        """Default PKL directory for *sub_id* (batch-trained or first trained)."""
        return self.substation_model_dir(sub_id) / "multi_horizon"

    def substation_registry_path(self, sub_id: str) -> Path:
        return self.substation_model_dir(sub_id) / "registry.json"

    def substation_default_meta_path(self, sub_id: str) -> Path:
        return self.substation_default_pkl_dir(sub_id) / "meta.json"

    def substation_versions_dir(self, sub_id: str) -> Path:
        return self.substation_model_dir(sub_id) / "versions"

    def rel_substation_pkl_dir(self, sub_id: str) -> str:
        """Repo-relative POSIX path for the substation's default PKL dir."""
        return self.rel(self.substation_default_pkl_dir(sub_id)) + "/"

    # ── Readiness / lifecycle ───────────────────────────────────────────────────

    def _has_trained_models(self) -> bool:
        """True if any model directory (default, versions, live/WO, or substation) holds pkls + meta."""
        def _dir_ok(d: Path) -> bool:
            return d.is_dir() and (d / "meta.json").exists() and any(d.glob("*.pkl"))

        if _dir_ok(self.default_pkl_dir):
            return True
        if self.versions_dir.is_dir():
            for sub in self.versions_dir.iterdir():
                if _dir_ok(sub):
                    return True
        # Weather-only (Live Forecaster) models live in models/live/.
        if _dir_ok(self.models_dir / "live"):
            return True
        # Multi-substation projects have models under models/substations/<id>/
        # and never train an aggregate model, so check each substation dir.
        subs_root = self.substation_models_dir()
        if subs_root.is_dir():
            for sub_dir in subs_root.iterdir():
                if _dir_ok(sub_dir / "multi_horizon"):
                    return True
                if (sub_dir / "versions").is_dir():
                    for ver_dir in (sub_dir / "versions").iterdir():
                        if _dir_ok(ver_dir):
                            return True
        return False

    def is_ready(self) -> bool:
        """Whether the project can produce forecasts in the dashboard.

        Ready means the processed demand+weather CSV exists *and* at least one
        model directory with pkls + meta is present.  Readiness is inferred
        from files so legacy projects without a ``stage`` field (flensburg)
        keep working unchanged; an explicit ``stage == "ready"`` is also
        honoured as a fast path.
        """
        if self.stage == STAGE_READY:
            return True
        return self.data_path.exists() and self._has_trained_models()

    # ── Weather enrichment (Phase C) ────────────────────────────────────────────

    def attach_weather(self, enriched_df: "pd.DataFrame") -> Path:
        """Write the enriched demand+weather CSV and advance the lifecycle stage.

        Persists *enriched_df* to ``data/processed/demand_with_weather.csv`` and,
        unless models already exist, moves ``stage`` from ``awaiting_weather`` to
        ``awaiting_training`` (data ready, training still required).  Returns the
        path written.
        """
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        df = enriched_df.copy()
        df.index.name = "timestamp"
        df.to_csv(self.data_path)

        # Only advance the stage if there are no trained models yet; a project
        # that already has models is effectively ``ready`` and we leave it be.
        if not self._has_trained_models():
            self._set_stage(STAGE_AWAITING_TRAINING)
        return self.data_path

    def _set_stage(self, stage: str) -> None:
        """Update ``stage`` in memory and persist it to config.json."""
        self.stage = stage
        self._config["stage"] = stage
        config_path = self.root / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                on_disk = json.load(f)
        else:
            on_disk = dict(self._config)
        on_disk["stage"] = stage
        with open(config_path, "w") as f:
            json.dump(on_disk, f, indent=2)

    def update_meta(self, **kwargs) -> None:
        """Patch arbitrary fields in config.json and on self."""
        config_path = self.root / "config.json"
        on_disk = dict(self._config)
        if config_path.exists():
            with open(config_path) as f:
                on_disk = json.load(f)
        for k, v in kwargs.items():
            on_disk[k] = v
            self._config[k] = v
            setattr(self, k, v)
        with open(config_path, "w") as f:
            json.dump(on_disk, f, indent=2)

    # ── Creation (setup wizard) ─────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        *,
        project_id: str,
        name: str,
        city: str,
        lat: float,
        lon: float,
        timezone: str,
        demand: "pd.Series | pd.DataFrame",
        test_start: str,
        demand_unit: str = "MW",
        target_column: str = "heat_demand_mw",
        created_at: str | None = None,
        country_code: str | None = None,
        # Multi-substation parameters — omit for single-substation projects.
        mode: str = "single",
        substations: list[dict] | None = None,
    ) -> "Project":
        """Create a new project *skeleton* on disk and return the loaded Project.

        Writes ``projects/<id>/config.json`` (with ``stage="awaiting_weather"``)
        and ``data/processed/demand_only.csv``.  For multi-substation projects
        *demand* may be a DataFrame with one column per substation plus
        ``total_demand_mw``; for single-substation projects it is a Series.

        *substations* is a list of dicts:
            ``[{"id": "FS-001", "name": "Substation 001", "column": "sub_001"}, …]``
        """
        root = PROJECTS_DIR / project_id
        if (root / "config.json").exists():
            raise FileExistsError(f"Project {project_id!r} already exists at {root}.")

        (root / "data" / "processed").mkdir(parents=True, exist_ok=True)
        (root / "models" / "versions").mkdir(parents=True, exist_ok=True)

        # Demand-only CSV
        if isinstance(demand, pd.DataFrame):
            df_out = demand.copy()
            df_out.index.name = "timestamp"
            df_out.dropna(how="all").to_csv(root / "data" / "processed" / "demand_only.csv")
        else:
            series = demand.copy()
            series.name = target_column
            series.index.name = "timestamp"
            series.dropna().to_frame().to_csv(root / "data" / "processed" / "demand_only.csv")

        if created_at is None:
            created_at = datetime.now(_dt_timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        config: dict = {
            "id": project_id,
            "name": name,
            "city": city,
            "lat": float(lat),
            "lon": float(lon),
            "timezone": timezone,
            "demand_unit": demand_unit,
            "target_column": target_column,
            "created_at": created_at,
            "stage": STAGE_AWAITING_WEATHER,
            "test_start": test_start,
        }
        if country_code:
            config["country_code"] = str(country_code).strip().upper()
        if mode == "multi" and substations:
            config["mode"] = "multi"
            config["substations"] = substations
        with open(root / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        return cls.load(project_id)

    # ── Deletion ────────────────────────────────────────────────────────────────

    @classmethod
    def delete(cls, project_id: str) -> bool:
        """Permanently delete the entire ``projects/<id>/`` directory.

        Removes the project's config, data **and** trained models from disk.
        Mirrors :meth:`ModelRegistry.remove_version`'s path-containment safety:
        the resolved target must be *strictly inside* the repo's ``projects/``
        directory (and never equal to ``projects/`` itself) before anything is
        removed, so a crafted ``project_id`` (``..``, an absolute path, etc.)
        can never escape the projects tree.

        Returns ``True`` after a successful removal, or ``False`` if the project
        does not exist or fails the containment guard.
        """
        target = (PROJECTS_DIR / project_id).resolve()
        projects_resolved = PROJECTS_DIR.resolve()

        # Containment guard: refuse projects/ itself or anything outside it.
        if target == projects_resolved or projects_resolved not in target.parents:
            return False

        # Treat a missing config.json (or missing dir) as "does not exist".
        if not (target / "config.json").exists():
            return False

        shutil.rmtree(target)
        return True

    # ── Repo-relative helpers (registry stores paths relative to the repo root) ──

    def rel(self, path: Path) -> str:
        """Return *path* as a POSIX string relative to the repo root."""
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()

    @property
    def rel_default_pkl_dir(self) -> str:
        """e.g. ``projects/flensburg/models/multi_horizon/`` (trailing slash)."""
        return self.rel(self.default_pkl_dir) + "/"

    # ── Construction / discovery ────────────────────────────────────────────────

    @classmethod
    def load(cls, project_id: str) -> "Project":
        """Read ``projects/<id>/config.json`` and build a Project."""
        root = PROJECTS_DIR / project_id
        config_path = root / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"No project config found at {config_path}. "
                f"Available projects: {cls.list_all()}"
            )
        with open(config_path) as f:
            config = json.load(f)
        return cls(config, root)

    @classmethod
    def list_all(cls) -> list[str]:
        """Return available project ids (default project first, then alphabetical)."""
        if not PROJECTS_DIR.exists():
            return []
        ids = sorted(p.parent.name for p in PROJECTS_DIR.glob("*/config.json"))
        if DEFAULT_PROJECT_ID in ids:
            ids.remove(DEFAULT_PROJECT_ID)
            ids.insert(0, DEFAULT_PROJECT_ID)
        return ids

    @classmethod
    def default_id(cls) -> str:
        """A sensible default/active project id (prefers ``flensburg``)."""
        ids = cls.list_all()
        if DEFAULT_PROJECT_ID in ids:
            return DEFAULT_PROJECT_ID
        return ids[0] if ids else DEFAULT_PROJECT_ID

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return f"Project(id={self.id!r}, city={self.city!r})"
