"""
Train 48 direct-strategy LightGBM models — one per forecast horizon.

Run from the project root:
    python3 train_multi_horizon.py

    # Custom date range + versioned output:
    python3 train_multi_horizon.py \\
        --start-date 2021-01-01 \\
        --end-date   2022-12-31 \\
        --label      "2yr-short" \\
        --output-dir models/versions/2021-2022_2yr-short/

Models are saved to models/multi_horizon/lgbm_h01.pkl … lgbm_h48.pkl
alongside a meta.json with the feature list and per-horizon evaluation.
When --output-dir is specified, models are saved there instead and the
version is registered in models/registry.json.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.data import (
    DEFAULT_TEST_START,
    MH_CALENDAR_COLS,
    MH_FEATURES,
    MH_WEATHER_COLS,
    build_horizon_dataset,
    load_raw,
)
from src.model import (
    BEST_PARAMS,
    FAST_PARAMS,
    compute_mape,
    eval_multi_horizon,
    train_multi_horizon,
)
from src.model_registry import ModelRegistry
from src.project import REPO_ROOT, Project


# ── Training ──────────────────────────────────────────────────────────────────

def _train_on_range(
    df_raw: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
    features: list[str],
    callback=None,
    fast: bool = False,
    target: str = "heat_demand_mw",
) -> dict[int, lgb.LGBMRegressor]:
    """
    Train 48 LightGBM models limited to [start_date, end_date].

    Pre-filters df_raw so that model.py's internal ``year < 2024`` gate
    only sees data within the requested window.

    target: demand column to use for lag features and the training target.
    Defaults to ``heat_demand_mw`` for the aggregate model; pass a substation
    column name for per-substation training.
    """
    lgb_params = FAST_PARAMS if fast else BEST_PARAMS
    df_train = df_raw.copy()
    if start_date:
        df_train = df_train.loc[df_train.index >= pd.Timestamp(start_date)]
    if end_date:
        df_train = df_train.loc[df_train.index <= pd.Timestamp(end_date)]

    models: dict[int, lgb.LGBMRegressor] = {}
    total = 48
    for h in range(1, 49):
        df_h = build_horizon_dataset(df_train, h, noise_augment=True, target=target)
        train_split = df_h[df_h.index <= pd.Timestamp(end_date)] if end_date else df_h
        mdl = lgb.LGBMRegressor(**lgb_params, n_jobs=-1, random_state=42, verbose=-1)
        mdl.fit(train_split[features], train_split["_target"])
        models[h] = mdl
        if callback:
            callback(h, total)
    return models


# ── Persistence ───────────────────────────────────────────────────────────────

def _eval_with_features(
    df_raw: pd.DataFrame,
    models: dict[int, lgb.LGBMRegressor],
    features: list[str],
    test_start: str = DEFAULT_TEST_START,
    target: str = "heat_demand_mw",
) -> pd.DataFrame:
    """Evaluate each model on the held-out set (timestamp >= test_start).

    target: demand column used for lags and the evaluation target.
    """
    test_start_ts = pd.Timestamp(test_start)
    rows = []
    for h, mdl in sorted(models.items()):
        df_h = build_horizon_dataset(df_raw, h, target=target)
        test = df_h[df_h.index >= test_start_ts]
        preds = mdl.predict(test[features])
        y = test["_target"]
        signed_res = y.values - preds          # actual − predicted
        # Relative residuals keep bands proportional to demand magnitude.
        # Guard against near-zero predictions to avoid division blow-up.
        rel_res = signed_res / np.maximum(np.abs(preds), 1e-6)
        mape = compute_mape(y, pd.Series(preds, index=y.index))
        mae = float(np.mean(np.abs(signed_res)))
        rmse = float(np.sqrt(np.mean(signed_res ** 2)))
        rows.append({
            "horizon_h":  h,
            "mape_pct":   round(mape, 2),
            "mae_mw":     round(mae, 2),
            "rmse_mw":    round(rmse, 2),
            # Normalized conformal calibration: CI = forecast * (1 + cal_q{10,90})
            # Relative fractions so summer bands are narrow and winter bands are wide.
            "cal_q10":    round(float(np.percentile(rel_res, 10)), 4),
            "cal_q90":    round(float(np.percentile(rel_res, 90)), 4),
        })
    return pd.DataFrame(rows)


def _save_to_dir(
    output_dir: Path,
    models: dict[int, lgb.LGBMRegressor],
    eval_df: pd.DataFrame | None,
    features: list[str],
) -> None:
    """Pickle all models and write meta.json to a custom output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for h, mdl in models.items():
        with open(output_dir / f"lgbm_h{h:02d}.pkl", "wb") as f:
            pickle.dump(mdl, f)
    meta: dict = {
        "features": features,
        "weather_cols": MH_WEATHER_COLS,
        "calendar_cols": MH_CALENDAR_COLS,
        "horizons": sorted(models.keys()),
    }
    if eval_df is not None:
        meta["eval"] = eval_df.to_dict(orient="records")
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


# ── Registry ──────────────────────────────────────────────────────────────────

def _rel_pkl_dir(output_dir: Path) -> str:
    """Return output_dir as a repo-root-relative path string (trailing slash)."""
    try:
        rel = Path(output_dir).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        rel = str(output_dir).rstrip("/")
    return rel.rstrip("/") + "/"


def _register(
    output_dir: Path,
    start_date: str,
    end_date: str,
    label: str,
    eval_df: pd.DataFrame | None,
    registry: ModelRegistry,
    is_default: bool = False,
    label_prefix: str = "",
) -> str:
    """Build version id, register it, and return the id."""
    safe_label = label.strip().replace(" ", "-") if label.strip() else ""
    version_id = f"{label_prefix}{start_date}_{end_date}"
    if safe_label:
        version_id += f"_{safe_label}"

    if eval_df is not None:
        mape_avg = round(float(eval_df["mape_pct"].mean()), 2)
        mape_per_horizon = {
            f"h{int(r['horizon_h'])}": r["mape_pct"]
            for r in eval_df.to_dict(orient="records")
        }
    else:
        mape_avg = float("nan")
        mape_per_horizon = {}

    registry.register_version(
        version_id=version_id,
        label=label.strip(),
        date_range_start=start_date,
        date_range_end=end_date,
        trained_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        mape_avg=mape_avg,
        mape_per_horizon=mape_per_horizon,
        pkl_dir=_rel_pkl_dir(output_dir),
        is_default=is_default,
    )
    return version_id


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train 48 direct LightGBM models, one per forecast horizon."
    )
    p.add_argument(
        "--project",
        default="flensburg",
        help="Project id under projects/ to train (default: flensburg).",
    )
    p.add_argument(
        "--start-date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Training start date (default: earliest row in CSV).",
    )
    p.add_argument(
        "--end-date",
        default=None,
        metavar="YYYY-MM-DD",
        help=(
            "Training end date. Default: the day before the project's "
            "test_start (e.g. 2023-12-31 for a 2024-01-01 split)."
        ),
    )
    p.add_argument(
        "--label",
        default="",
        help="Human-readable version label stored in registry.json.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        metavar="PATH",
        help=(
            "Directory to save PKLs and meta.json "
            "(default: the project's models/multi_horizon/ — legacy behaviour)."
        ),
    )
    p.add_argument(
        "--features",
        default=None,
        metavar="CSV",
        help="Comma-separated feature names to train with (default: all 31).",
    )
    p.add_argument(
        "--weather-only",
        action="store_true",
        help=(
            "Train a weather-only model (no demand lags) for the Live Forecaster tab. "
            "Saves to models/live/ by default and registers with a 'live' label prefix."
        ),
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Use lightweight LightGBM params (n_estimators=100 instead of 500). "
            "~5× faster training, ~+0.5–1% MAPE. Recommended on weak cloud VMs."
        ),
    )
    p.add_argument(
        "--substation",
        default=None,
        metavar="ID",
        help=(
            "Substation id (as defined in the project's substations list) to train. "
            "The matching demand column is used as the training target and models are "
            "saved under models/substations/<id>/. "
            "Omit to train the aggregate model (default behaviour)."
        ),
    )
    p.add_argument(
        "--label-prefix",
        default="",
        metavar="PREFIX",
        help=(
            "Prepend PREFIX to the version id written to registry.json. "
            "Use 'batch_' for batch-trained models so they can be "
            "filtered separately in the app sidebar."
        ),
    )
    return p.parse_args()


def _parse_features(raw: str | None) -> list[str]:
    if not raw:
        return list(MH_FEATURES)
    chosen = [f.strip() for f in raw.split(",") if f.strip()]
    unknown = [f for f in chosen if f not in MH_FEATURES]
    if unknown:
        raise SystemExit(f"Unknown features: {unknown}")
    if not chosen:
        raise SystemExit("At least one feature is required.")
    return chosen


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    use_custom = args.output_dir is not None
    start_date = args.start_date # None → use all available history
    features   = _parse_features(args.features)
    label_prefix = getattr(args, "label_prefix", "") or ""

    # ── Resolve project (paths, data, registry all scoped to it) ──────────────
    project = Project.load(args.project)

    # ── Substation mode: resolve target column + per-substation paths ──────────
    sub_id: str | None = getattr(args, "substation", None) or None
    target_column = project.target_column  # default: aggregate or heat_demand_mw

    if sub_id is not None:
        # Look up the matching column name in the project's substations list.
        sub_entry = next(
            (s for s in project.substations if s.get("id") == sub_id),
            None,
        )
        if sub_entry is None:
            # If not found by id, try treating sub_id directly as a column name.
            sub_entry = next(
                (s for s in project.substations if s.get("column") == sub_id),
                {"id": sub_id, "name": sub_id, "column": sub_id},
            )
        target_column = sub_entry.get("column", sub_id)

        # Unless the caller already specified --output-dir, route to the per-
        # substation model directory and use the substation's own registry.
        if not use_custom:
            args.output_dir = str(project.substation_default_pkl_dir(sub_id))
            use_custom = True
        registry = ModelRegistry(
            registry_path=project.substation_registry_path(sub_id),
            default_meta_path=project.substation_default_meta_path(sub_id),
            default_pkl_dir=project.rel_substation_pkl_dir(sub_id),
        )
        print(f"Substation: {sub_id}  (column: {target_column!r})")
    else:
        if args.weather_only:
            from src.data import MH_LIVE_FEATURES
            features = list(MH_LIVE_FEATURES)
            if not args.label:
                args.label = "WO"
            if args.output_dir is None:
                args.output_dir = str(project.root / "models" / "live")
            use_custom = True
        registry = ModelRegistry(
            registry_path=project.registry_path,
            default_meta_path=project.default_meta_path,
            default_pkl_dir=project.rel_default_pkl_dir,
        )

    # ── weather-only for non-substation runs (moved inside else above) ─────────
    if sub_id is None and not args.weather_only:
        pass  # registry already created above

    # Per-project train/test boundary: rows < test_start train, rows >= eval.
    # Falls back to the global default (2024-01-01) so flensburg is unchanged.
    test_start = project.test_start or DEFAULT_TEST_START
    test_start_ts = pd.Timestamp(test_start)
    # Default training end is the day before the test split unless overridden.
    end_date = args.end_date or (test_start_ts - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"Project: {project.id} ({project.city})")
    print(f"Train/test split at test_start={test_start} (train < it, eval >= it)")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading data…")
    t0 = time.time()
    df_raw = load_raw(project.data_path, target=target_column)

    effective_start = start_date or str(df_raw.index.min().date())

    if use_custom:
        n_train = ((df_raw.index >= pd.Timestamp(effective_start)) &
                   (df_raw.index <= pd.Timestamp(end_date))).sum()
    else:
        n_train = (df_raw.index < test_start_ts).sum()
    n_test = (df_raw.index >= test_start_ts).sum()
    print(
        f"  {len(df_raw):,} rows  |  "
        f"train {effective_start}–{end_date}: {n_train:,}  |  "
        f"test (>= {test_start}): {n_test:,}"
    )

    # ── Train 48 models ───────────────────────────────────────────────────────
    _mode = "FAST_PARAMS" if args.fast else "BEST_PARAMS"
    print(f"\nTraining 48 LightGBM models ({_mode}, noise_augment=True)…")
    print(f"  Features: {len(features)} of {len(MH_FEATURES)}")
    print("  NWP noise augmentation: σ(h) added to fc_* weather features during training.")
    trained = 0

    def on_done(h: int, total: int) -> None:
        nonlocal trained
        trained += 1
        bar = "#" * trained + "-" * (total - trained)
        print(f"  [{bar}] h={h:2d}/{total}", flush=True)

    t1 = time.time()
    if use_custom:
        models = _train_on_range(
            df_raw, start_date, end_date, features,
            callback=on_done, fast=args.fast, target=target_column,
        )
    else:
        models = train_multi_horizon(
            df_raw, noise_augment=True, callback=on_done,
            test_start=test_start, target=target_column,
        )
    elapsed_train = time.time() - t1
    print(f"\n  Done in {elapsed_train:.0f}s")

    # ── Evaluate on the held-out set (timestamp >= test_start) ────────────────
    print(f"\nEvaluating on held-out set (>= {test_start})…")
    if use_custom:
        eval_df = _eval_with_features(
            df_raw, models, features, test_start=test_start, target=target_column,
        )
    else:
        eval_df = eval_multi_horizon(df_raw, models, test_start=test_start, target=target_column)

    print(f"\n{'h':>3}  {'MAPE':>6}  {'MAE':>7}  {'RMSE':>7}")
    print("─" * 30)
    for _, row in eval_df.iterrows():
        h    = int(row["horizon_h"])
        mape = row["mape_pct"]
        mae  = row["mae_mw"]
        rmse = row["rmse_mw"]
        flag = " ◄ best" if h == eval_df["mape_pct"].idxmin() + 1 else ""
        print(f"{h:>3}  {mape:>5.1f}%  {mae:>6.1f} MW  {rmse:>6.1f} MW{flag}")

    print("─" * 30)
    print(
        f"{'avg':>3}  {eval_df['mape_pct'].mean():>5.1f}%  "
        f"{eval_df['mae_mw'].mean():>6.1f} MW  "
        f"{eval_df['rmse_mw'].mean():>6.1f} MW"
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    print("\nSaving models…")
    if use_custom:
        output_dir = Path(args.output_dir)
        _save_to_dir(output_dir, models, eval_df, features)
        print(f"  Saved to {output_dir}/  ({len(models)} files)")
        version_id = _register(
            output_dir, effective_start, end_date, args.label, eval_df, registry,
            label_prefix=label_prefix,
        )
        print(f"Version registered: {version_id}")
    else:
        output_dir = project.default_pkl_dir
        _save_to_dir(output_dir, models, eval_df, features)
        print(f"  Saved to {output_dir}/  ({len(models)} files)")
        version_id = _register(
            output_dir, effective_start, end_date, args.label, eval_df,
            registry, is_default=True, label_prefix=label_prefix,
        )
        print(f"Default version registered: {version_id}")

    print(f"\nTotal time: {time.time() - t0:.0f}s")
    print("\nNote: models trained with NWP noise augmentation.")
    print("      Confidence bands in the app now come from Monte Carlo sampling.")
    print("      Re-run without --no-augment to compare against the clean baseline.")


if __name__ == "__main__":
    main()
