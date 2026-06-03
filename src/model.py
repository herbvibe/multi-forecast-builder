"""
LightGBM model training, evaluation, persistence, and inference.
All heavy operations are designed to be wrapped in @st.cache_resource.

Production inference path: get_mh_forecast_window() — 48 direct LightGBM models,
one per forecast horizon (direct strategy, no recursive feedback).
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.data import (
    TARGET,
    DEFAULT_TEST_START,
    MH_MODELS_DIR,
    MH_WEATHER_COLS,
    MH_CALENDAR_COLS,
    MH_FEATURES,
    load_raw,
    build_horizon_dataset,
    snapshot_lag_features,
)
from src.forecast_noise import noise_sigmas
from src.weather import fetch_open_meteo_forecast  # re-exported for consumers

# ── Constants ──────────────────────────────────────────────────────────────────

# Best params found in notebook 04 (grid-searched, 5-fold TimeSeriesSplit)
BEST_PARAMS: dict = {
    "num_leaves": 31,
    "learning_rate": 0.05,
    "min_data_in_leaf": 20,
    "n_estimators": 500,
}

# Lightweight params for fast training (~5× faster, ~+0.5–1% MAPE).
# Suitable for weak cloud VMs or quick iteration.
FAST_PARAMS: dict = {
    "num_leaves": 31,
    "learning_rate": 0.05,
    "min_data_in_leaf": 20,
    "n_estimators": 100,
}

# Test-set fallback metrics — used when meta.json eval is unavailable
TEST_MAPE = 7.8
TEST_MAE  = 8.62
TEST_RMSE = 12.67


# ── Metrics ───────────────────────────────────────────────────────────────────

def _is_finite_number(x) -> bool:
    """True only if *x* can become a finite float (rejects None, NaN, inf, str)."""
    if x is None:
        return False
    try:
        return bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def compute_mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """MAPE excluding near-zero actuals (threshold relative to data scale)."""
    # Align on common index so differently-indexed series compare correctly.
    y_true, y_pred = y_true.align(y_pred, join="inner")
    _ref = float(y_true.abs().median()) if len(y_true) > 0 else 0.0
    # Exclude values below 1% of the median — avoids division-by-near-zero while
    # adapting to any demand unit (W, kW, MW, …).
    _min_val = max(_ref * 0.01, 1e-9)
    mask = y_true > _min_val
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ── Multi-horizon (direct) training ───────────────────────────────────────────

def train_multi_horizon(
    df_raw: pd.DataFrame,
    horizons: range = range(1, 49),
    params: dict | None = None,
    noise_augment: bool = True,
    callback=None,
    test_start: str = DEFAULT_TEST_START,
    target: str = TARGET,
) -> dict[int, lgb.LGBMRegressor]:
    """
    Train one LightGBM per forecast horizon using the direct strategy.

    Args:
        df_raw:         raw DataFrame from load_raw() (no pre-added lag cols).
        horizons:       which horizons to train (default 1–48).
        params:         LightGBM params; defaults to BEST_PARAMS from notebook 04.
        noise_augment:  if True, add horizon-calibrated NWP noise to fc_* weather
                        features during training.  Closes the train/test mismatch
                        where production weather comes from a forecast, not ERA5.
        callback:       optional callable(h, total) called after each model trains.
        test_start:     ISO date marking the start of the held-out evaluation
                        set.  Training rows are those with timestamp < test_start.
                        Defaults to ``2024-01-01`` for backward compatibility
                        (the original hardcoded ``year < 2024`` boundary).
        target:         demand column to use for lag features and the training
                        target. Defaults to ``heat_demand_mw``.

    Returns:
        dict mapping horizon h → fitted LGBMRegressor.
    """
    if params is None:
        params = BEST_PARAMS

    test_start_ts = pd.Timestamp(test_start)
    models: dict[int, lgb.LGBMRegressor] = {}
    total = len(horizons)
    for i, h in enumerate(horizons, 1):
        df_h = build_horizon_dataset(df_raw, h, noise_augment=noise_augment, target=target)
        train = df_h[df_h.index < test_start_ts]
        mdl = lgb.LGBMRegressor(**params, n_jobs=-1, random_state=42, verbose=-1)
        mdl.fit(train[MH_FEATURES], train["_target"])
        models[h] = mdl
        if callback:
            callback(h, total)

    return models


def eval_multi_horizon(
    df_raw: pd.DataFrame,
    models: dict[int, lgb.LGBMRegressor],
    test_start: str = DEFAULT_TEST_START,
    target: str = TARGET,
) -> pd.DataFrame:
    """
    Evaluate each model on the held-out test set (timestamp >= test_start).
    Returns a DataFrame with columns: horizon_h, mape_pct, mae_mw, rmse_mw,
    cal_q10, cal_q90 (conformal calibration bounds).

    ``test_start`` defaults to ``2024-01-01`` so the legacy "test == 2024"
    behaviour is preserved when no project-specific boundary is supplied.
    target: demand column to use for lags and evaluation. Defaults to
        ``heat_demand_mw``.
    """
    test_start_ts = pd.Timestamp(test_start)
    rows = []
    for h, mdl in sorted(models.items()):
        df_h = build_horizon_dataset(df_raw, h, noise_augment=True, target=target)
        test = df_h[df_h.index >= test_start_ts]
        preds = mdl.predict(test[MH_FEATURES])
        y = test["_target"]
        signed_res = y.values - preds
        rel_res = signed_res / np.maximum(np.abs(preds), 1e-6)
        mape = compute_mape(y, pd.Series(preds, index=y.index))
        mae  = float(np.mean(np.abs(signed_res)))
        rmse = float(np.sqrt(np.mean(signed_res ** 2)))
        rows.append({
            "horizon_h": h,
            "mape_pct":  round(mape, 2),
            "mae_mw":    round(mae, 2),
            "rmse_mw":   round(rmse, 2),
            "cal_q10":   round(float(np.percentile(rel_res, 10)), 4),
            "cal_q90":   round(float(np.percentile(rel_res, 90)), 4),
        })
    return pd.DataFrame(rows)


# ── Persistence ───────────────────────────────────────────────────────────────

def save_multi_horizon_models(
    models: dict[int, lgb.LGBMRegressor],
    eval_df: pd.DataFrame | None = None,
) -> None:
    """Pickle all models + metadata into MH_MODELS_DIR."""
    MH_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for h, mdl in models.items():
        with open(MH_MODELS_DIR / f"lgbm_h{h:02d}.pkl", "wb") as f:
            pickle.dump(mdl, f)
    meta: dict = {
        "features": MH_FEATURES,
        "weather_cols": MH_WEATHER_COLS,
        "calendar_cols": MH_CALENDAR_COLS,
        "horizons": sorted(models.keys()),
    }
    if eval_df is not None:
        meta["eval"] = eval_df.to_dict(orient="records")
    with open(MH_MODELS_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_multi_horizon_models() -> dict[int, lgb.LGBMRegressor]:
    """Load all saved multi-horizon models. Returns {} if not yet trained."""
    models: dict[int, lgb.LGBMRegressor] = {}
    for h in range(1, 49):
        p = MH_MODELS_DIR / f"lgbm_h{h:02d}.pkl"
        if p.exists():
            try:
                with open(p, "rb") as f:
                    models[h] = pickle.load(f)
            except Exception:
                try:
                    p.unlink()
                except OSError:
                    pass
    return models


def load_mh_eval() -> pd.DataFrame | None:
    """Load the per-horizon evaluation results saved alongside the models."""
    meta_path = MH_MODELS_DIR / "meta.json"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    if "eval" not in meta:
        return None
    return pd.DataFrame(meta["eval"])


# ── Inference ─────────────────────────────────────────────────────────────────

def get_mh_forecast_window(
    df_raw: pd.DataFrame,
    models: dict[int, lgb.LGBMRegressor],
    snapshot_dt: pd.Timestamp,
    history_hours: int = 48,
    wx_forecast: pd.DataFrame | None = None,
    target: str = TARGET,
) -> dict | None:
    """
    Generate a forecast using the multi-horizon ensemble.

    For each horizon h the corresponding model produces one prediction using:
      - demand lags read directly from df_raw at snapshot_dt
      - weather/calendar from df_raw at snapshot_dt + h  (oracle weather)

    wx_forecast, if provided, overrides weather columns with real NWP values.
    target: column to read demand history and actuals from. Defaults to
        ``heat_demand_mw``; pass a substation column for per-substation inference.
    Returns a dict: history, forecast, upper, lower, actual_future, weather.
    """
    hist_start = snapshot_dt - pd.Timedelta(hours=history_hours)
    history = df_raw.loc[hist_start:snapshot_dt, target]
    if len(history) < 2:
        return None

    # Demand-lag features — same for every horizon
    lag_feats = snapshot_lag_features(df_raw, snapshot_dt, target=target)
    if any(np.isnan(v) for v in lag_feats.values()):
        return None

    forecasts: dict[pd.Timestamp, float] = {}
    actuals:   dict[pd.Timestamp, float] = {}

    for h in sorted(models):
        fc_ts = snapshot_dt + pd.Timedelta(hours=h)
        if fc_ts not in df_raw.index:
            continue
        fc_row = df_raw.loc[fc_ts]
        if wx_forecast is not None and fc_ts in wx_forecast.index:
            wx_row = wx_forecast.loc[fc_ts]
            fc_feats = {
                f"fc_{c}": float(wx_row[c]) if c in wx_row.index else float(fc_row[c])
                for c in MH_WEATHER_COLS + MH_CALENDAR_COLS
            }
        else:
            fc_feats = {f"fc_{c}": float(fc_row[c]) for c in MH_WEATHER_COLS + MH_CALENDAR_COLS}
        feat_vec = pd.DataFrame([{**lag_feats, **fc_feats}])[MH_FEATURES]
        forecasts[fc_ts] = float(models[h].predict(feat_vec)[0])
        actuals[fc_ts]   = float(df_raw.loc[fc_ts, target])

    if not forecasts:
        return None

    fc  = pd.Series(forecasts).sort_index()
    act = pd.Series(actuals).sort_index()
    n   = len(fc)

    # Load per-horizon RMSE if available, else fall back to global TEST_RMSE
    eval_df = load_mh_eval()
    if eval_df is not None:
        rmse_map = dict(zip(eval_df["horizon_h"], eval_df["rmse_mw"]))
        sigma = np.array([rmse_map.get(h, TEST_RMSE) for h in sorted(models) if
                          snapshot_dt + pd.Timedelta(hours=h) in df_raw.index])
    else:
        sigma = TEST_RMSE * np.sqrt(np.arange(1, n + 1) / 24)
        sigma = np.clip(sigma, None, 40.0)

    if wx_forecast is not None:
        # NWP temperature error ≈ 0.5°C per 12 h; demand sensitivity ≈ 4 MW/°C
        wx_sigma = 4.0 * 0.5 * np.arange(1, n + 1) / 12
        sigma = np.sqrt(sigma ** 2 + wx_sigma ** 2)

    return {
        "history":       history,
        "actual_future": act,
        "forecast":      fc,
        "lower":         pd.Series(fc.values - 1.28 * sigma, index=fc.index),
        "upper":         pd.Series(fc.values + 1.28 * sigma, index=fc.index),
        "weather":       df_raw.loc[fc.index,
                             ["temperature_c", "wind_speed_ms", "cloud_cover_pct",
                              "humidity_pct"]],
    }


# ── Conformal / Monte-Carlo forecast ─────────────────────────────────────────

def simulate_forecast(
    df_raw: pd.DataFrame,
    models: dict[int, lgb.LGBMRegressor],
    snapshot_dt: pd.Timestamp,
    n_samples: int = 200,
    wx_forecast: pd.DataFrame | None = None,
    history_hours: int = 48,
    seed: int = 0,
    features: list[str] | None = None,
    mh_eval: "pd.DataFrame | None" = None,
    target: str = TARGET,
) -> dict | None:
    """
    Forecast from *snapshot_dt* for all 48 horizons.

    target: demand column to read history and actuals from. Defaults to
        ``heat_demand_mw``; pass a substation column for per-substation inference.

    Confidence bands strategy (in priority order):
    1. **Conformal** — use per-horizon cal_q10/cal_q90 residual quantiles stored
       in *mh_eval* (computed on the held-out test set during training). These
       give empirically calibrated, asymmetric 80 % bands.
    2. **Monte Carlo fallback** — if *mh_eval* is None or lacks cal columns,
       perturb weather features with horizon-calibrated Gaussian noise (legacy).

    Returns the same dict schema as get_mh_forecast_window:
      history, actual_future, forecast (median), lower, upper, weather, samples.
    """
    hist_start = snapshot_dt - pd.Timedelta(hours=history_hours)
    history = df_raw.loc[hist_start:snapshot_dt, target]
    if len(history) < 2:
        return None

    feat_cols = features if features is not None else MH_FEATURES

    lag_feats = snapshot_lag_features(df_raw, snapshot_dt, target=target)
    _used_lag_cols = [c for c in lag_feats if c in feat_cols]
    if _used_lag_cols and any(np.isnan(lag_feats[c]) for c in _used_lag_cols):
        return None

    # Build conformal calibration lookup {horizon_h: (q10, q90)} if available.
    cal: dict[int, tuple[float, float]] = {}
    if mh_eval is not None and "cal_q10" in mh_eval.columns and "cal_q90" in mh_eval.columns:
        for _, row in mh_eval.iterrows():
            h_int = int(row["horizon_h"])
            cal[h_int] = (float(row["cal_q10"]), float(row["cal_q90"]))

    use_conformal = bool(cal)
    rng = np.random.default_rng(seed)
    active_wx_cols = [c for c in MH_WEATHER_COLS if f"fc_{c}" in feat_cols]
    weather_idx = [feat_cols.index(f"fc_{c}") for c in active_wx_cols]

    horizon_list = sorted(models.keys())
    valid_horizons = [
        h for h in horizon_list
        if snapshot_dt + pd.Timedelta(hours=h) in df_raw.index
    ]
    if not valid_horizons:
        return None

    all_medians: dict[pd.Timestamp, float] = {}
    all_lower:   dict[pd.Timestamp, float] = {}
    all_upper:   dict[pd.Timestamp, float] = {}
    all_actuals: dict[pd.Timestamp, float] = {}
    sample_matrix: list[np.ndarray] = []

    for h in valid_horizons:
        fc_ts = snapshot_dt + pd.Timedelta(hours=h)
        fc_row = df_raw.loc[fc_ts]

        base_wx = {}
        for c in MH_WEATHER_COLS:
            cell = None
            if (
                wx_forecast is not None
                and fc_ts in wx_forecast.index
                and c in wx_forecast.columns
            ):
                cell = wx_forecast.loc[fc_ts, c]
                if isinstance(cell, pd.Series):
                    cell = cell.iloc[0] if len(cell) else None
            base_wx[c] = float(cell) if _is_finite_number(cell) else float(fc_row[c])
        cal_feats = {f"fc_{c}": float(fc_row[c]) for c in MH_CALENDAR_COLS}

        base_row = {**lag_feats, **{f"fc_{c}": v for c, v in base_wx.items()}, **cal_feats}
        base_vec = pd.DataFrame([base_row])[feat_cols].values[0]

        median_pred = float(models[h].predict(
            pd.DataFrame([base_row])[feat_cols]
        )[0])

        if use_conformal and h in cal:
            q10, q90 = cal[h]
            # Relative conformal: bands scale with forecast magnitude
            lower_val = median_pred * (1.0 + q10)
            upper_val = median_pred * (1.0 + q90)
            preds = np.array([median_pred])
        elif n_samples > 1:
            sigmas_h = noise_sigmas(active_wx_cols, h)
            noise = rng.normal(0, 1, (n_samples, len(active_wx_cols))) * sigmas_h
            X_batch = np.tile(base_vec, (n_samples, 1))
            if weather_idx:
                X_batch[:, weather_idx] += noise
            X_df = pd.DataFrame(X_batch, columns=feat_cols)
            preds = models[h].predict(X_df)
            lower_val = float(np.percentile(preds, 10))
            upper_val = float(np.percentile(preds, 90))
        else:
            preds = np.array([median_pred])
            lower_val = median_pred
            upper_val = median_pred

        sample_matrix.append(preds)
        all_medians[fc_ts] = median_pred
        all_lower[fc_ts]   = lower_val
        all_upper[fc_ts]   = upper_val
        all_actuals[fc_ts] = float(fc_row[target])

    fc  = pd.Series(all_medians).sort_index()
    act = pd.Series(all_actuals).sort_index()

    return {
        "history":       history,
        "actual_future": act,
        "forecast":      fc,
        "lower":         pd.Series(all_lower).sort_index(),
        "upper":         pd.Series(all_upper).sort_index(),
        "weather":       df_raw.loc[fc.index,
                             ["temperature_c", "wind_speed_ms", "cloud_cover_pct",
                              "humidity_pct"]],
        "samples":       np.array(sample_matrix),   # (n_horizons, n_samples)
    }


def simulate_live_forecast(
    models: dict[int, lgb.LGBMRegressor],
    wx_forecast: pd.DataFrame,
    snapshot_dt: pd.Timestamp,
    n_samples: int = 200,
    seed: int = 0,
    features: list[str] | None = None,
    country_code: str | None = None,
    mh_eval: "pd.DataFrame | None" = None,
) -> dict | None:
    """
    Live forecast using only weather + calendar features (no demand lags).

    Confidence bands use conformal calibration (cal_q10/cal_q90 from *mh_eval*)
    when available, otherwise fall back to Monte Carlo weather-noise sampling.

    Returns a dict with keys: forecast, lower, upper, weather.
    """
    from src.data import MH_LIVE_FEATURES, MH_WEATHER_COLS, MH_CALENDAR_COLS
    from src.weather_history import _add_calendar_features

    feat_cols = features if features is not None else MH_LIVE_FEATURES
    active_wx_cols = [c for c in MH_WEATHER_COLS if f"fc_{c}" in feat_cols]
    active_cal_cols = [c for c in MH_CALENDAR_COLS if f"fc_{c}" in feat_cols]
    weather_idx = [feat_cols.index(f"fc_{c}") for c in active_wx_cols]

    wx_cal = wx_forecast.copy()
    wx_cal = _add_calendar_features(wx_cal, country_code)

    # Build conformal calibration lookup if available.
    cal: dict[int, tuple[float, float]] = {}
    if mh_eval is not None and "cal_q10" in mh_eval.columns and "cal_q90" in mh_eval.columns:
        for _, row in mh_eval.iterrows():
            h_int = int(row["horizon_h"])
            cal[h_int] = (float(row["cal_q10"]), float(row["cal_q90"]))
    use_conformal = bool(cal)

    rng = np.random.default_rng(seed)

    horizon_list = sorted(models.keys())
    valid_horizons = [
        h for h in horizon_list
        if snapshot_dt + pd.Timedelta(hours=h) in wx_cal.index
    ]
    if not valid_horizons:
        return None

    all_medians: dict[pd.Timestamp, float] = {}
    all_lower:   dict[pd.Timestamp, float] = {}
    all_upper:   dict[pd.Timestamp, float] = {}

    for h in valid_horizons:
        fc_ts = snapshot_dt + pd.Timedelta(hours=h)
        fc_row = wx_cal.loc[fc_ts]
        if isinstance(fc_row, pd.DataFrame):
            fc_row = fc_row.iloc[0]

        base_row: dict[str, float] = {}
        for c in active_wx_cols:
            val = fc_row.get(c, 0.0) if hasattr(fc_row, "get") else getattr(fc_row, c, 0.0)
            base_row[f"fc_{c}"] = float(val) if _is_finite_number(val) else 0.0
        for c in active_cal_cols:
            val = fc_row.get(c, 0.0) if hasattr(fc_row, "get") else getattr(fc_row, c, 0.0)
            base_row[f"fc_{c}"] = float(val) if _is_finite_number(val) else 0.0

        base_vec = pd.DataFrame([base_row])[feat_cols].values[0]
        median_pred = float(models[h].predict(pd.DataFrame([base_row])[feat_cols])[0])

        if use_conformal and h in cal:
            q10, q90 = cal[h]
            # Relative conformal: bands scale with forecast magnitude
            lower_val = median_pred * (1.0 + q10)
            upper_val = median_pred * (1.0 + q90)
        elif n_samples > 1:
            sigmas_h = noise_sigmas(active_wx_cols, h)
            noise = rng.normal(0, 1, (n_samples, len(active_wx_cols))) * sigmas_h
            X_batch = np.tile(base_vec, (n_samples, 1))
            if weather_idx:
                X_batch[:, weather_idx] += noise
            X_df = pd.DataFrame(X_batch, columns=feat_cols)
            preds = models[h].predict(X_df)
            lower_val = float(np.percentile(preds, 10))
            upper_val = float(np.percentile(preds, 90))
        else:
            lower_val = median_pred
            upper_val = median_pred

        all_medians[fc_ts] = median_pred
        all_lower[fc_ts]   = lower_val
        all_upper[fc_ts]   = upper_val

    fc = pd.Series(all_medians).sort_index()

    _WX_DISPLAY = ["temperature_c", "wind_speed_ms", "cloud_cover_pct", "humidity_pct"]
    wx_avail = [c for c in _WX_DISPLAY if c in wx_cal.columns]

    return {
        "forecast": fc,
        "lower":    pd.Series(all_lower).sort_index(),
        "upper":    pd.Series(all_upper).sort_index(),
        "weather":  wx_cal.loc[fc.index, wx_avail] if wx_avail else pd.DataFrame(index=fc.index),
    }


