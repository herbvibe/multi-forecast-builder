"""
Data loading and feature-construction utilities for the heating forecaster.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "heat_demand_mw"

# ── Shared physical / split constants ─────────────────────────────────────────
# Heating-degree base temperature (°C).  ``heating_degrees`` is computed as
# ``max(HEATING_BASE_C - temperature_c, 0)`` and is the single source of truth
# for BOTH historical enrichment (src/weather_history.py) and live forecast
# fetch (src/weather.py).  Keeping one constant avoids a train/serve mismatch:
# Flensburg's training data was built at base 15, so serving must match.
HEATING_BASE_C = 15.0

# Default train/test split boundary used when a project does not specify its own
# ``test_start``.  Rows with timestamp < test_start are training data; rows with
# timestamp >= test_start are the held-out evaluation set.  This default
# preserves the original hardcoded "train < 2024 / test == 2024" behaviour.
DEFAULT_TEST_START = "2024-01-01"

# Default paths resolve to the Flensburg project (the migrated default project).
# Callers that are project-aware should pass paths from Project.* instead of
# relying on these module-level defaults.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA_PATH = _REPO_ROOT / "projects" / "flensburg" / "data" / "processed" / "demand_with_weather.csv"
MH_MODELS_DIR = _REPO_ROOT / "projects" / "flensburg" / "models" / "multi_horizon"

# ── Multi-horizon feature columns ─────────────────────────────────────────────
MH_WEATHER_COLS: list[str] = [
    "temperature_c", "heating_degrees", "temp_change_3h",
    "wind_speed_ms", "wind_sin", "wind_cos",
    "solar_radiation_wm2", "humidity_pct",
    "snowfall_cm", "snow_depth_m", "cloud_cover_pct",
]
MH_CALENDAR_COLS: list[str] = [
    "hour", "day_of_week", "month", "day_of_year",
    "is_weekend", "is_holiday", "is_school_holiday", "heating_season_day",
]
# Demand lags observed at snapshot time t.
# demand_lag_0h = y(t): last closed hour, known when the forecast is issued.
# Short-term lags (1h–12h) give h=1…12 models visibility of the recent trend;
# LightGBM simply down-weights them for longer horizons.
MH_LAG_COLS: list[str] = [
    "demand_lag_0h",                                     # current demand at t
    "demand_lag_1h", "demand_lag_2h", "demand_lag_3h",   # short-term trend
    "demand_lag_6h", "demand_lag_12h",                   # intra-day shape
    "demand_lag_24h", "demand_lag_48h",                  # daily pattern
    "demand_lag_168h", "demand_lag_336h",                # weekly pattern
    "demand_roll_24h", "demand_roll_168h",
]
# Full feature list for every multi-horizon model (always 31 features)
MH_FEATURES: list[str] = (
    MH_LAG_COLS + [f"fc_{c}" for c in MH_WEATHER_COLS + MH_CALENDAR_COLS]
)

# Weather-only feature list for the Live Forecaster (no demand lags).
# Used when training and serving the weather-only model variant.
MH_LIVE_FEATURES: list[str] = [f"fc_{c}" for c in MH_WEATHER_COLS + MH_CALENDAR_COLS]

# Live-forecaster model directory path (weather-only models)
MH_LIVE_MODELS_DIR = _REPO_ROOT / "projects" / "flensburg" / "models" / "live"


def load_raw(path: Path = _DATA_PATH, target: str = TARGET) -> pd.DataFrame:
    """Load the raw CSV without adding lag features (for multi-horizon use).

    ``path`` is normally supplied by the caller from ``Project.data_path``.
    It defaults to the Flensburg project for backward compatibility.
    """
    df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
    return df.dropna(subset=[target])


def build_horizon_dataset(
    df_raw: pd.DataFrame,
    h: int,
    noise_augment: bool = False,
    noise_seed: int | None = None,
    target: str = TARGET,
) -> pd.DataFrame:
    """
    Build a training dataset for direct forecast horizon h (1 ≤ h ≤ 48).

    Each row at time t contains:
      - demand lags observed at t (no future leakage)
      - weather + calendar features at t+h  (the 'forecast' inputs)
    Target: demand at t+h.

    noise_augment: if True, add horizon-calibrated Gaussian noise to every
        fc_* weather feature, simulating the uncertainty present in real NWP
        forecasts.  This closes the train/test mismatch where the model would
        otherwise see perfect future weather during training.

    target: demand column to use for lags and the training target.  Defaults to
        ``heat_demand_mw`` for backward compatibility.  Pass a substation column
        name (e.g. ``"FS-001"``) to train a per-substation model.
    """
    from src.forecast_noise import noise_sigmas  # local import avoids circularity

    df = df_raw.copy()

    # Demand lags — all anchored at t (no shift into the future)
    df["demand_lag_0h"]   = df[target]
    df["demand_lag_1h"]   = df[target].shift(1)
    df["demand_lag_2h"]   = df[target].shift(2)
    df["demand_lag_3h"]   = df[target].shift(3)
    df["demand_lag_6h"]   = df[target].shift(6)
    df["demand_lag_12h"]  = df[target].shift(12)
    df["demand_lag_24h"]  = df[target].shift(24)
    df["demand_lag_48h"]  = df[target].shift(48)
    df["demand_lag_168h"] = df[target].shift(168)
    df["demand_lag_336h"] = df[target].shift(336)
    df["demand_roll_24h"]  = df[target].rolling(24).mean()
    df["demand_roll_168h"] = df[target].rolling(168).mean()

    # Weather & calendar at the forecast time t+h (shift columns backward by h)
    for col in MH_WEATHER_COLS + MH_CALENDAR_COLS:
        df[f"fc_{col}"] = df[col].shift(-h)

    # Target: demand at t+h
    df["_target"] = df[target].shift(-h)

    # Defensive belt: a weather/calendar feature that is entirely absent (all
    # NaN — e.g. snow_depth_m at a location Open-Meteo never reports snow for)
    # would make the dropna() below wipe EVERY row, leaving an empty training
    # set ("Input data must be 2 dimensional and non empty").  The legitimate
    # rows to drop are only the lag/target edge rows, not an absent feature, so
    # fill any entirely-NaN weather/calendar column (both the raw column and its
    # fc_ shifted twin) with 0.0 first.  Columns that are fully present
    # (Flensburg/Aalborg) contain no all-NaN column, so this is a no-op for them
    # and their row counts are unchanged.
    for col in MH_WEATHER_COLS + MH_CALENDAR_COLS:
        if col in df.columns and df[col].isna().all():
            df[col] = 0.0
        fc_col = f"fc_{col}"
        if fc_col in df.columns and df[fc_col].isna().all():
            df[fc_col] = 0.0

    df = df.dropna()

    if noise_augment:
        rng = np.random.default_rng(seed=noise_seed if noise_seed is not None else h)
        sigmas = noise_sigmas(MH_WEATHER_COLS, h)          # (n_weather_cols,)
        n = len(df)
        noise = rng.normal(0, 1, (n, len(MH_WEATHER_COLS))) * sigmas[None, :]
        for i, col in enumerate(MH_WEATHER_COLS):
            df[f"fc_{col}"] = df[f"fc_{col}"].values + noise[:, i]

    return df


def snapshot_lag_features(
    df_raw: pd.DataFrame,
    snapshot_dt: pd.Timestamp,
    target: str = TARGET,
) -> dict[str, float]:
    """Demand lag features at forecast issue time t (= snapshot_dt).

    target: column to pull demand lags from.  Defaults to ``heat_demand_mw``.
    Pass a substation column name for per-substation inference.
    """
    def _demand_at(ts: pd.Timestamp) -> float:
        return float(df_raw.loc[ts, target]) if ts in df_raw.index else np.nan

    recent = df_raw.loc[:snapshot_dt, target]
    return {
        "demand_lag_0h":   _demand_at(snapshot_dt),
        "demand_lag_1h":   _demand_at(snapshot_dt - pd.Timedelta(hours=1)),
        "demand_lag_2h":   _demand_at(snapshot_dt - pd.Timedelta(hours=2)),
        "demand_lag_3h":   _demand_at(snapshot_dt - pd.Timedelta(hours=3)),
        "demand_lag_6h":   _demand_at(snapshot_dt - pd.Timedelta(hours=6)),
        "demand_lag_12h":  _demand_at(snapshot_dt - pd.Timedelta(hours=12)),
        "demand_lag_24h":  _demand_at(snapshot_dt - pd.Timedelta(hours=24)),
        "demand_lag_48h":  _demand_at(snapshot_dt - pd.Timedelta(hours=48)),
        "demand_lag_168h": _demand_at(snapshot_dt - pd.Timedelta(hours=168)),
        "demand_lag_336h": _demand_at(snapshot_dt - pd.Timedelta(hours=336)),
        "demand_roll_24h":  float(recent.tail(24).mean()),
        "demand_roll_168h": float(recent.tail(168).mean()),
    }
