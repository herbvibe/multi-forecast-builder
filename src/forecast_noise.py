"""
NWP forecast error model for the heating demand forecaster.

Provides σ(h) — the expected one-sigma forecast error for each weather
variable at lead time h hours — calibrated against published GFS
verification statistics for northern mid-latitudes (approx. Flensburg).

Reference: NCEP/ECMWF operational forecast verification reports.
"""
from __future__ import annotations

import numpy as np

# Linear error-growth model: σ(h) = baseline + slope * h (hours)
# Tuned to match GFS skill at 6 h / 24 h / 48 h lead times.
_SIGMA_PARAMS: dict[str, tuple[float, float]] = {
    #                              baseline   slope/h
    "temperature_c":       (0.30,  0.042),  # °C   h=6:0.55  h=24:1.31  h=48:2.32
    "heating_degrees":     (0.30,  0.042),  # °C   same as temperature_c
    "temp_change_3h":      (0.20,  0.018),  # °C   h=6:0.31  h=24:0.63  h=48:1.06
    "wind_speed_ms":       (0.40,  0.052),  # m/s  h=6:0.71  h=24:1.65  h=48:2.90
    "wind_sin":            (0.03,  0.0040), # [-1,1]
    "wind_cos":            (0.03,  0.0040), # [-1,1]
    "cloud_cover_pct":     (5.0,   0.40),   # %    h=6:7.4   h=24:14.6  h=48:24.2
    "solar_radiation_wm2": (5.0,   1.50),   # W/m² h=6:14    h=24:41    h=48:77
    "humidity_pct":        (2.0,   0.18),   # %    h=6:3.1   h=24:6.3   h=48:10.6
    "snowfall_cm":         (0.02,  0.003),  # cm
    "snow_depth_m":        (0.01,  0.001),  # m
}


def sigma(col: str, h: int) -> float:
    """
    Return the one-sigma NWP forecast error for weather variable `col`
    at lead time h hours.  Returns 0.0 for unknown variables.
    """
    baseline, slope = _SIGMA_PARAMS.get(col, (0.0, 0.0))
    return baseline + slope * h


def noise_sigmas(weather_cols: list[str], h: int) -> np.ndarray:
    """
    Return a (len(weather_cols),) array of σ values for a given horizon h.
    Used for vectorised batch generation.
    """
    return np.array([sigma(c, h) for c in weather_cols])
