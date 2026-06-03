"""
Open-Meteo weather forecast integration for the heating forecaster.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import requests

from src.data import HEATING_BASE_C, MH_WEATHER_COLS

# Flensburg district heating network coordinates
FLENSBURG_LAT = 54.79
FLENSBURG_LON = 9.44


def fetch_open_meteo_forecast(
    snapshot_dt: pd.Timestamp,
    horizon: int = 48,
    lat: float = FLENSBURG_LAT,
    lon: float = FLENSBURG_LON,
    history_hours: int = 0,
) -> pd.DataFrame | None:
    """
    Fetch hourly weather for snapshot_dt-history_hours … snapshot_dt+horizon.

    For historical dates the Historical Forecast API is used (archived NWP
    output actually issued on that date).  For near-current dates the regular
    Forecast API is used.

    Returns a DataFrame indexed by timestamp with columns matching
    MH_WEATHER_COLS plus `precipitation_mm`, or None on any error.
    """
    fc_start = snapshot_dt - pd.Timedelta(hours=history_hours)
    fc_end   = snapshot_dt + pd.Timedelta(hours=horizon)

    variables = [
        "temperature_2m", "relative_humidity_2m",
        "precipitation",                           # rain + snow combined (mm)
        "snowfall", "snow_depth", "cloud_cover",
        "wind_speed_10m", "wind_direction_10m",
        "shortwave_radiation",
    ]

    # Historical Forecast API covers dates from 2022 onwards.
    # Use it for anything older than ~3 days; regular API for near-future.
    cutoff = pd.Timestamp.utcnow().tz_localize(None).normalize() - pd.Timedelta(days=3)
    if snapshot_dt < cutoff:
        url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
        params: dict = dict(
            latitude=lat, longitude=lon,
            start_date=fc_start.strftime("%Y-%m-%d"),
            end_date=fc_end.strftime("%Y-%m-%d"),
            hourly=variables,
            timezone="UTC",
        )
    else:
        url = "https://api.open-meteo.com/v1/forecast"
        params = dict(
            latitude=lat, longitude=lon,
            hourly=variables,
            forecast_days=4,
            timezone="UTC",
        )

    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        raw = r.json()["hourly"]
    except Exception:
        return None

    df = pd.DataFrame(raw)
    df.index = pd.to_datetime(df.pop("time"))
    df.index.name = "timestamp"

    df = df.rename(columns={
        "temperature_2m":       "temperature_c",
        "relative_humidity_2m": "humidity_pct",
        "precipitation":        "precipitation_mm",
        "snowfall":             "snowfall_cm",
        "snow_depth":           "snow_depth_m",
        "cloud_cover":          "cloud_cover_pct",
        "wind_speed_10m":       "wind_speed_ms",
        "shortwave_radiation":  "solar_radiation_wm2",
    })

    # Derived features the model needs
    df["heating_degrees"] = (HEATING_BASE_C - df["temperature_c"]).clip(lower=0)
    df["temp_change_3h"]  = df["temperature_c"].diff(3).fillna(0)
    wd_rad = np.deg2rad(df["wind_direction_10m"].fillna(0))
    df["wind_sin"] = np.sin(wd_rad)
    df["wind_cos"] = np.cos(wd_rad)
    df = df.drop(columns=["wind_direction_10m"])

    mask = (df.index >= fc_start) & (df.index <= fc_end)
    return df.loc[mask].copy() if mask.any() else None
