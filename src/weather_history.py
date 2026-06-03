"""
Historical weather enrichment for the "new project" setup wizard (Phase C).

Turns a demand-only series into the model-ready ``demand_with_weather.csv`` by
fetching **historical** weather from the Open-Meteo ERA5 archive and computing
exactly the same derived + calendar features the original Flensburg notebook
(``notebooks/02_weather_features.ipynb``) produced.

Why the ERA5 *archive* and not the historical-forecast API?
    The archive endpoint (``archive-api.open-meteo.com/v1/archive``) provides
    ERA5 reanalysis covering 1940 → present.  The historical-forecast API only
    goes back to ~2022, so it cannot enrich older series (e.g. Aalborg 2018-2020).

Timezone assumption
    Open-Meteo returns hourly timestamps as *naive* wall-clock times in the
    ``timezone`` we request.  The demand series timestamps are assumed to be in
    the project's configured timezone, so by fetching weather with that same
    timezone the two align hour-for-hour.  ``timezone`` is exposed as a
    parameter and defaults to the project tz — but if the demand series is
    actually stored in UTC (as the Aalborg series is) the caller must pass
    ``timezone="UTC"`` so the alignment is correct.  The alignment sanity check
    (Pearson corr of demand vs temperature, which must be strongly negative for
    heating demand) exists precisely to catch a wrong timezone choice.

The single public entry point is :func:`enrich_with_weather`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import requests

from src.data import HEATING_BASE_C

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Hourly ERA5 variables requested (raw Open-Meteo names).
_ARCHIVE_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "snowfall",
    "snow_depth",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation",
]

# Raw Open-Meteo name → Flensburg schema name.  ``precipitation`` is fetched for
# completeness but dropped (it is not part of the target schema).
_RENAME = {
    "temperature_2m": "temperature_c",
    "relative_humidity_2m": "humidity_pct",
    "snowfall": "snowfall_cm",
    "snow_depth": "snow_depth_m",
    "cloud_cover": "cloud_cover_pct",
    "wind_speed_10m": "wind_speed_ms",
    "wind_direction_10m": "wind_direction_deg",
    "shortwave_radiation": "solar_radiation_wm2",
}

# Exact Flensburg column order (the demand target column comes first).
TARGET_SCHEMA = [
    "temperature_c", "wind_speed_ms", "wind_direction_deg", "solar_radiation_wm2",
    "humidity_pct", "snowfall_cm", "snow_depth_m", "cloud_cover_pct",
    "hour", "day_of_week", "month", "day_of_year", "is_weekend",
    "is_holiday", "is_school_holiday", "wind_sin", "wind_cos",
    "temp_change_3h", "heating_degrees", "heating_season_day",
]

# Span (in days) above which we chunk the archive request year-by-year.  A
# single multi-year call usually works, but this is the documented fallback.
_CHUNK_THRESHOLD_DAYS = 366 * 6


# ── Weather fetch ────────────────────────────────────────────────────────────

def _fetch_archive(
    lat: float, lon: float, start: str, end: str, timezone: str
) -> pd.DataFrame:
    """Fetch one ERA5 archive window and return a timestamp-indexed DataFrame.

    Raises on network/parse failure so the caller can decide how to handle it.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(_ARCHIVE_VARIABLES),
        "timezone": timezone,
    }
    r = requests.get(ARCHIVE_URL, params=params, timeout=60)
    r.raise_for_status()
    hourly = r.json()["hourly"]
    df = pd.DataFrame(hourly)
    df.index = pd.to_datetime(df.pop("time"))
    df.index.name = "timestamp"
    return df


def _fetch_weather_range(
    lat: float, lon: float,
    start_date: pd.Timestamp, end_date: pd.Timestamp,
    timezone: str,
    progress=None,
) -> pd.DataFrame:
    """Fetch the full weather range, chunking by calendar year if it is long.

    One request covers most series; very long spans are split year-by-year and
    concatenated (Open-Meteo occasionally limits very large windows).
    """
    span_days = (end_date - start_date).days
    if span_days <= _CHUNK_THRESHOLD_DAYS:
        if progress:
            progress(0.15, "Fetching ERA5 archive (single request)…")
        return _fetch_archive(
            lat, lon,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
            timezone,
        )

    years = list(range(start_date.year, end_date.year + 1))
    frames: list[pd.DataFrame] = []
    for i, yr in enumerate(years):
        chunk_start = max(start_date, pd.Timestamp(f"{yr}-01-01"))
        chunk_end = min(end_date, pd.Timestamp(f"{yr}-12-31"))
        if progress:
            progress(0.10 + 0.5 * (i / len(years)), f"Fetching weather for {yr}…")
        frames.append(
            _fetch_archive(
                lat, lon,
                chunk_start.strftime("%Y-%m-%d"),
                chunk_end.strftime("%Y-%m-%d"),
                timezone,
            )
        )
    combined = pd.concat(frames)
    return combined[~combined.index.duplicated(keep="first")].sort_index()


# ── Calendar / holiday features ──────────────────────────────────────────────

def _resolve_country_code(country_code: str | None) -> str | None:
    """Normalise a country code to the ISO-3166 alpha-2 the holidays pkg wants."""
    if not country_code:
        return None
    code = str(country_code).strip().upper()
    return code or None


def _holiday_series(index: pd.DatetimeIndex, country_code: str | None) -> pd.Series:
    """1 on public-holiday dates for *country_code*, else 0.

    Uses the ``holidays`` package keyed by ISO country code.  If the package is
    missing, the code is unknown/absent, or lookup fails, returns all-zeros and
    the caller records that holidays were unavailable.
    """
    code = _resolve_country_code(country_code)
    if code is None:
        return pd.Series(0, index=index, dtype=int)
    try:
        import holidays as _holidays

        years = range(int(index.year.min()), int(index.year.max()) + 1)
        cal = _holidays.country_holidays(code, years=years)
        dates = index.normalize()
        return pd.Series(
            [1 if d.date() in cal else 0 for d in dates], index=index, dtype=int
        )
    except Exception:
        return pd.Series(0, index=index, dtype=int)


def _school_holiday_series(index: pd.DatetimeIndex) -> pd.Series:
    """Approximate school-holiday flag (1/0) via a documented heuristic.

    The original Flensburg notebook hard-coded the official Schleswig-Holstein
    school calendar — that table is region-specific and cannot generalise to an
    arbitrary new project's country.  We therefore approximate with the common
    Northern-European school-break windows:

      * Summer break    : 1 Jul – 15 Aug
      * Autumn break     : 12 – 20 Oct
      * Christmas/winter : 20 Dec – 5 Jan
      * Winter/sports    : 8 – 16 Feb
      * Spring/Easter    : 1 – 10 Apr (fixed approximation; Easter actually moves)

    This is intentionally a coarse, documented approximation; consumers needing
    exact dates should supply a region calendar.  When in doubt the model simply
    treats it as a weak categorical signal.
    """
    month = index.month
    day = index.day
    summer = (month == 7) | ((month == 8) & (day <= 15))
    autumn = (month == 10) & (day >= 12) & (day <= 20)
    christmas = ((month == 12) & (day >= 20)) | ((month == 1) & (day <= 5))
    winter = (month == 2) & (day >= 8) & (day <= 16)
    spring = (month == 4) & (day <= 10)
    flag = summer | autumn | christmas | winter | spring
    return pd.Series(flag.astype(int), index=index)


def _heating_season_day(index: pd.DatetimeIndex) -> pd.Series:
    """Days since 1 Oct of the current heating season (clipped at 0).

    Replicates notebook 02: for a timestamp in month >= Oct the season starts on
    1 Oct of the same year, otherwise 1 Oct of the previous year.
    """
    season_year = np.where(index.month >= 10, index.year, index.year - 1)
    season_start = pd.to_datetime(
        [pd.Timestamp(f"{int(y)}-10-01") for y in season_year]
    )
    days = (index.normalize() - season_start).days
    return pd.Series(days, index=index).clip(lower=0)


def _add_calendar_features(
    df: pd.DataFrame, country_code: str | None
) -> pd.DataFrame:
    """Attach all calendar/holiday/season features computed from the index."""
    idx = df.index
    df["hour"] = idx.hour
    df["day_of_week"] = idx.dayofweek
    df["month"] = idx.month
    df["day_of_year"] = idx.dayofyear
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    df["is_holiday"] = _holiday_series(idx, country_code)
    df["is_school_holiday"] = _school_holiday_series(idx)
    df["heating_season_day"] = _heating_season_day(idx)
    return df


# Raw weather columns (post-rename) that make up the model schema, and the
# subset whose natural default is 0.0 when the archive omits the variable.
_RAW_WEATHER_COLS = (
    "temperature_c", "humidity_pct", "snowfall_cm", "snow_depth_m",
    "cloud_cover_pct", "wind_speed_ms", "wind_direction_deg", "solar_radiation_wm2",
)
# Snow variables are frequently absent for warm/maritime locations; "absent"
# legitimately means "no snow", so they default to 0.0 rather than being
# interpolated.
_ZERO_DEFAULT_COLS = ("snowfall_cm", "snow_depth_m")


def _fill_weather_gaps(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Guarantee every weather column exists and is never entirely NaN.

    Open-Meteo sometimes omits a variable for a given location (e.g. ``snow_depth``
    at warm/maritime sites), returning the column as all-nulls or not at all.  An
    entirely-NaN weather column later wipes the whole training set in
    :func:`src.data.build_horizon_dataset` (its ``df.dropna()`` drops every row),
    which is exactly the crash that broke the equator ``flensburg-2`` project.

    Fill policy (documented):
      * A column that is missing or **entirely** NaN is set to its default
        (0.0 for every weather variable) and reported as *defaulted*.
      * ``snowfall_cm`` / ``snow_depth_m``: remaining NaNs → 0.0 (no snow).
      * All other columns: isolated NaNs are time-interpolated, then edge gaps
        forward/back-filled, then any residual NaN → 0.0.

    Returns the filled frame and the list of columns that were entirely absent
    from the API and had to be defaulted (surfaced in the enrichment report).
    """
    defaulted: list[str] = []
    for col in _RAW_WEATHER_COLS:
        if col not in df.columns or df[col].isna().all():
            df[col] = 0.0
            defaulted.append(col)
            continue
        if col in _ZERO_DEFAULT_COLS:
            df[col] = df[col].fillna(0.0)
        else:
            filled = df[col].interpolate(method="time", limit_direction="both")
            df[col] = filled.ffill().bfill().fillna(0.0)
    return df, defaulted


def _add_derived_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Derived weather features — identical formulas to ``src/weather.py``."""
    df["heating_degrees"] = (HEATING_BASE_C - df["temperature_c"]).clip(lower=0)
    df["temp_change_3h"] = df["temperature_c"].diff(3).fillna(0)
    wd_rad = np.deg2rad(df["wind_direction_deg"].fillna(0))
    df["wind_sin"] = np.sin(wd_rad)
    df["wind_cos"] = np.cos(wd_rad)
    return df


# ── Public entry point ───────────────────────────────────────────────────────

def enrich_with_weather(
    demand_df: pd.DataFrame,
    lat: float,
    lon: float,
    timezone: str,
    country_code: str | None = None,
    progress=None,
    target_column: str = "heat_demand_mw",
) -> tuple[pd.DataFrame | None, dict]:
    """Enrich a demand-only series with historical ERA5 weather + features.

    Parameters
    ----------
    demand_df
        DataFrame indexed by timestamp containing the demand column.
    lat, lon
        Project coordinates.
    timezone
        Timezone passed to Open-Meteo so the returned wall-clock times align
        with the demand timestamps.  Pass ``"UTC"`` if the demand series is in
        UTC even when the project's configured tz differs (see module docstring).
    country_code
        ISO-3166 alpha-2 code used for public-holiday lookup; ``None`` disables
        holidays (recorded in the report).
    progress
        Optional ``callback(fraction: float, text: str)`` for UI progress.

    Returns
    -------
    (df, report)
        ``df`` is the enriched frame in the exact Flensburg column order, or
        ``None`` on a fatal fetch error.  ``report`` always carries diagnostics.
    """
    report: dict = {"ok": False}

    def _p(frac: float, text: str) -> None:
        if progress:
            progress(frac, text)

    demand = demand_df.copy()
    if target_column not in demand.columns:
        # Single-column frames are accepted by taking the first column.
        if demand.shape[1] == 1:
            demand = demand.rename(columns={demand.columns[0]: target_column})
        else:
            report["error"] = (
                f"Demand frame has no '{target_column}' column (cols={list(demand.columns)})."
            )
            return None, report

    demand.index = pd.to_datetime(demand.index)
    if getattr(demand.index, "tz", None) is not None:
        demand.index = demand.index.tz_localize(None)

    # Preserve any extra columns (e.g. individual substation demand columns in
    # multi-mode projects) so they survive the weather join below.
    extra_cols = [c for c in demand.columns if c != target_column]
    demand = demand[[target_column] + extra_cols].copy()
    # Drop rows where the *target* column is NaN, but keep extra cols.
    demand = demand[demand[target_column].notna()].sort_index()
    if demand.empty:
        report["error"] = "Demand series is empty after dropping NaNs."
        return None, report

    start_date = demand.index.min().normalize()
    end_date = demand.index.max().normalize()

    _p(0.05, "Preparing weather request…")
    try:
        weather = _fetch_weather_range(
            lat, lon, start_date, end_date, timezone, progress=_p
        )
    except Exception as exc:  # noqa: BLE001 - surface a friendly error to the UI
        report["error"] = f"Could not fetch ERA5 archive weather: {exc}"
        return None, report

    if weather.empty:
        report["error"] = "Open-Meteo returned no weather rows for this range."
        return None, report

    _p(0.65, "Renaming + deriving weather features…")
    weather = weather.rename(columns=_RENAME)
    weather = weather.drop(columns=["precipitation"], errors="ignore")
    # Fill gaps / default absent variables BEFORE deriving features so that
    # heating_degrees, temp_change_3h and the wind sin/cos are computed on clean
    # values and no weather column can be entirely NaN downstream.
    weather, defaulted_cols = _fill_weather_gaps(weather)
    weather = _add_derived_weather(weather)
    weather = _add_calendar_features(weather, country_code)

    _p(0.85, "Joining weather to demand…")
    enriched = demand.join(weather, how="inner")
    # Column order: target first, then extra (substation) columns, then weather schema.
    enriched = enriched[[target_column] + extra_cols + TARGET_SCHEMA]

    # ── Validation / report ───────────────────────────────────────────────────
    nan_counts = {c: int(enriched[c].isna().sum()) for c in enriched.columns}
    corr = float(
        enriched[target_column].corr(enriched["temperature_c"])
    ) if len(enriched) > 1 else float("nan")

    warnings: list[str] = []
    if _resolve_country_code(country_code) is None:
        warnings.append(
            "No country code supplied — is_holiday is all zeros."
        )
    if defaulted_cols:
        warnings.append(
            "Open-Meteo did not return these weather variables for this location; "
            f"they were defaulted to 0.0: {', '.join(defaulted_cols)}."
        )
    if not (corr < -0.4):
        warnings.append(
            f"ALIGNMENT WARNING: corr(demand, temperature) = {corr:.3f} is not "
            "strongly negative (< -0.4). The weather/demand timezone alignment "
            "is likely wrong — re-check the `timezone` argument (e.g. pass "
            "timezone='UTC' if the demand series is stored in UTC)."
        )

    report.update(
        ok=True,
        n_rows=int(len(enriched)),
        n_demand_rows=int(len(demand)),
        nan_counts=nan_counts,
        total_nans=int(sum(nan_counts.values())),
        date_start=str(enriched.index.min()),
        date_end=str(enriched.index.max()),
        alignment_corr=corr,
        country_code=_resolve_country_code(country_code),
        timezone=timezone,
        defaulted_weather_columns=defaulted_cols,
        warnings=warnings,
    )

    _p(1.0, "Weather enrichment complete.")
    return enriched, report
