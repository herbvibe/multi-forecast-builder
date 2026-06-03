"""
Model registry backed by models/registry.json.

Tracks trained model versions with their metadata, allowing the dashboard
to switch between different training configurations without restarting.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

# Defaults resolve to the Flensburg project (the migrated default project).
# Project-aware callers pass registry_path / default_meta_path / default_pkl_dir
# from a Project instance instead of relying on these module-level defaults.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_FLENSBURG = _REPO_ROOT / "projects" / "flensburg"
REGISTRY_PATH = _FLENSBURG / "models" / "registry.json"
DEFAULT_META_PATH = _FLENSBURG / "models" / "multi_horizon" / "meta.json"
DEFAULT_PKL_DIR = "projects/flensburg/models/multi_horizon/"


class ModelRegistry:
    """
    Lightweight JSON-backed registry of trained model versions.

    Version entry schema::

        {
          "id": "2020-2023_default",
          "label": "default",
          "date_range_start": "2020-01-01",
          "date_range_end": "2023-12-31",
          "trained_at": "2026-05-30T17:52:00",
          "mape_avg": 7.8,
          "mape_per_horizon": {"h1": 6.6, "h24": 7.9, ...},
          "pkl_dir": "models/multi_horizon/",
          "is_default": true
        }
    """

    def __init__(
        self,
        registry_path: Path = REGISTRY_PATH,
        default_meta_path: Path = DEFAULT_META_PATH,
        default_pkl_dir: str = DEFAULT_PKL_DIR,
    ) -> None:
        self._path = registry_path
        self._default_meta_path = default_meta_path
        self._default_pkl_dir = default_pkl_dir
        self._versions: list[dict] = []

    # ── I/O ───────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load registry from disk (no-op if file does not exist)."""
        if self._path.exists():
            with open(self._path) as f:
                data = json.load(f)
            self._versions = data.get("versions", [])
        else:
            self._versions = []

    def save(self) -> None:
        """Persist registry to disk, creating parent directories as needed."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump({"versions": self._versions}, f, indent=2)

    # ── Queries ───────────────────────────────────────────────────────────────

    def list_versions(self) -> list[dict]:
        """Return all versions sorted by trained_at descending (newest first)."""
        self.load()
        return sorted(
            self._versions,
            key=lambda v: v.get("trained_at", ""),
            reverse=True,
        )

    def get_version(self, version_id: str) -> dict | None:
        """Return a single version entry by id, or None if not found."""
        self.load()
        for v in self._versions:
            if v["id"] == version_id:
                return v
        return None

    # ── Mutations ─────────────────────────────────────────────────────────────

    def register_version(
        self,
        version_id: str,
        label: str,
        date_range_start: str,
        date_range_end: str,
        trained_at: str,
        mape_avg: float,
        mape_per_horizon: dict,
        pkl_dir: str,
        is_default: bool = False,
    ) -> None:
        """
        Add or replace a version entry, then persist.

        If ``is_default=True``, clears ``is_default`` on all other entries first.
        """
        self.load()
        self._versions = [v for v in self._versions if v["id"] != version_id]
        if is_default:
            for v in self._versions:
                v["is_default"] = False
        entry: dict = {
            "id": version_id,
            "label": label,
            "date_range_start": date_range_start,
            "date_range_end": date_range_end,
            "trained_at": trained_at,
            "mape_avg": round(float(mape_avg), 2),
            "mape_per_horizon": mape_per_horizon,
            "pkl_dir": pkl_dir,
            "is_default": is_default,
        }
        self._versions.append(entry)
        self.save()

    def remove_version(
        self,
        version_id: str,
        *,
        delete_files: bool = False,
        project_root: Path | None = None,
    ) -> bool:
        """Remove a version from the registry; optionally delete its PKL directory."""
        self.load()
        entry = next((v for v in self._versions if v["id"] == version_id), None)
        if entry is None:
            return False

        self._versions = [v for v in self._versions if v["id"] != version_id]
        self.save()

        if delete_files:
            root = project_root or Path(__file__).parent.parent
            pkl_path = (root / entry["pkl_dir"]).resolve()
            root_resolved = root.resolve()
            if (
                pkl_path.is_dir()
                and str(pkl_path).startswith(str(root_resolved))
                and pkl_path != root_resolved
            ):
                shutil.rmtree(pkl_path)

        return True

    # ── Bootstrap ─────────────────────────────────────────────────────────────

    def auto_register_existing_default(self) -> None:
        """
        If registry.json is missing, seed it from models/multi_horizon/meta.json.

        Called silently on first app load.  Silent no-op if meta.json is missing
        or registry.json already exists.
        """
        if self._path.exists():
            return
        try:
            if not self._default_meta_path.exists():
                return
            with open(self._default_meta_path) as f:
                meta = json.load(f)
            eval_records: list[dict] = meta.get("eval", [])
            if eval_records:
                mapes = [r["mape_pct"] for r in eval_records]
                mape_avg = round(sum(mapes) / len(mapes), 2)
                mape_per_horizon = {
                    f"h{r['horizon_h']}": r["mape_pct"] for r in eval_records
                }
            else:
                mape_avg = 7.8
                mape_per_horizon = {}
            self.register_version(
                version_id="2020-2023_default",
                label="default",
                date_range_start="2020-01-01",
                date_range_end="2023-12-31",
                trained_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
                mape_avg=mape_avg,
                mape_per_horizon=mape_per_horizon,
                pkl_dir=self._default_pkl_dir,
                is_default=True,
            )
        except Exception:
            pass
