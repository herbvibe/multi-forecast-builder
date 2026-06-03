"""
Pure-logic ingestion utilities for the "new project" setup wizard (Phase B).

This module turns an uploaded demand timeseries CSV into a clean, hourly
``heat_demand_mw`` series plus a structured quality report.  It deliberately
contains **no Streamlit imports** so every function is unit-testable in
isolation; the wizard UI in :mod:`src.builder_ui` is a thin layer on top.

Pipeline::

    parse_csv → detect_columns → normalize_demand → (Project.create)

The weather enrichment and model training are *later* phases — Phase B stops
once we have a validated demand-only series.
"""
from __future__ import annotations

import io
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

DEMAND_COLUMN = "heat_demand_mw"
# Aggregate column written into multi-substation processed CSVs.
AGGREGATE_COLUMN = "total_demand_mw"

# Unit → multiplicative factor to convert a per-hour energy/power value to MW.
#   MW            : already megawatts
#   kW            : 1000 kW = 1 MW
#   kWh_per_hour  : energy over one hour == average kW over that hour → /1000
#   W             : 1e6 W = 1 MW
_UNIT_FACTORS: dict[str, float] = {
    "MW": 1.0,
    "kW": 1.0 / 1_000.0,
    "kWh_per_hour": 1.0 / 1_000.0,
    "W": 1.0 / 1_000_000.0,
}

# Human-friendly aliases accepted from the UI / external callers.
_UNIT_ALIASES: dict[str, str] = {
    "mw": "MW",
    "megawatt": "MW",
    "megawatts": "MW",
    "kw": "kW",
    "kilowatt": "kW",
    "kilowatts": "kW",
    "kwh_per_hour": "kWh_per_hour",
    "kwh/h": "kWh_per_hour",
    "kwh per hour": "kWh_per_hour",
    "kwh": "kWh_per_hour",
    "w": "W",
    "watt": "W",
    "watts": "W",
}

# Name hints used to recognise demand / timestamp columns.
_DEMAND_HINTS = ("demand", "heat", "load", "consumption", "mw", "kwh", "kw", "power", "energy")
_TIME_HINTS = ("time", "date", "timestamp", "datetime", "ts", "hour", "period")


# ── 1. Parsing ────────────────────────────────────────────────────────────────

def _read_text(file_or_path) -> str:
    """Read raw text from a path, Path, or file-like object (e.g. UploadedFile)."""
    if hasattr(file_or_path, "read"):
        pos = None
        try:
            pos = file_or_path.tell()
        except Exception:
            pos = None
        raw = file_or_path.read()
        if pos is not None:
            try:
                file_or_path.seek(pos)
            except Exception:
                pass
        if isinstance(raw, bytes):
            return raw.decode("utf-8-sig", errors="replace")
        return str(raw)
    return Path(file_or_path).read_text(encoding="utf-8-sig", errors="replace")


def _sniff_dialect(text: str) -> tuple[str, str]:
    """Return ``(sep, decimal)`` by sniffing the first non-empty lines.

    Comma-separated / dot-decimal is the default.  European files that use a
    semicolon separator with comma decimals are detected when ``;`` is more
    frequent than ``,`` across the sampled lines.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()][:20]
    n_semicolon = sum(ln.count(";") for ln in lines)
    n_comma = sum(ln.count(",") for ln in lines)
    n_tab = sum(ln.count("\t") for ln in lines)
    if n_tab > n_semicolon and n_tab > n_comma:
        return "\t", "."
    if n_semicolon > n_comma:
        return ";", ","
    return ",", "."


def parse_csv(file_or_path) -> pd.DataFrame:
    """Read a demand CSV robustly.

    Defaults to comma separator / dot decimal but tolerates a ``;`` separator
    with ``,`` decimals (common in European exports) and tab-separated files by
    sniffing the raw text first.  Returns the raw DataFrame (no normalisation).
    """
    text = _read_text(file_or_path)
    if not text.strip():
        raise ValueError("The uploaded file is empty.")
    sep, decimal = _sniff_dialect(text)
    df = pd.read_csv(io.StringIO(text), sep=sep, decimal=decimal)
    # A single fat column usually means the separator guess was wrong — retry.
    if df.shape[1] == 1:
        for alt_sep, alt_dec in ((";", ","), (",", "."), ("\t", ".")):
            if alt_sep == sep:
                continue
            alt = pd.read_csv(io.StringIO(text), sep=alt_sep, decimal=alt_dec)
            if alt.shape[1] > 1:
                return alt
    return df


# ── 2. Column detection ───────────────────────────────────────────────────────

def _name_score(col: str, hints: tuple[str, ...]) -> float:
    low = str(col).lower()
    return 1.0 if any(h in low for h in hints) else 0.0


def detect_columns(df: pd.DataFrame) -> dict[str, str | None]:
    """Best-effort guess of the timestamp and demand columns.

    Returns ``{"timestamp": col|None, "demand": col|None}``.

    Timestamp: column whose values parse to datetimes spanning more than a day
    (rejects numeric columns that collapse to the 1970 epoch).  Demand: numeric,
    mostly non-negative column with the highest coefficient of variation,
    boosted by name hints like demand/heat/load/mw/kwh.
    """
    ts_col: str | None = None
    best_ts = -1.0
    for col in df.columns:
        parsed = pd.to_datetime(df[col], errors="coerce")
        frac = float(parsed.notna().mean())
        if frac < 0.8:
            continue
        valid = parsed.dropna()
        if valid.empty:
            continue
        span = valid.max() - valid.min()
        if span < pd.Timedelta(days=1):
            continue
        score = frac + _name_score(col, _TIME_HINTS)
        if score > best_ts:
            best_ts = score
            ts_col = col

    demand_col: str | None = None
    best_d = -np.inf
    for col in df.columns:
        if col == ts_col:
            continue
        num = pd.to_numeric(df[col], errors="coerce")
        frac = float(num.notna().mean())
        if frac < 0.5:
            continue
        vals = num.dropna()
        if vals.empty:
            continue
        if float((vals >= 0).mean()) < 0.5:
            continue
        mean = abs(float(vals.mean())) or 1.0
        cv = float(vals.std()) / mean
        score = cv + 5.0 * _name_score(col, _DEMAND_HINTS)
        if score > best_d:
            best_d = score
            demand_col = col

    return {"timestamp": ts_col, "demand": demand_col}


# ── 3. Normalisation + quality report ──────────────────────────────────────────

def normalize_unit(unit: str) -> str:
    """Map a free-form unit string to a canonical key in :data:`_UNIT_FACTORS`."""
    if unit in _UNIT_FACTORS:
        return unit
    canon = _UNIT_ALIASES.get(str(unit).strip().lower())
    if canon is None:
        raise ValueError(
            f"Unsupported unit {unit!r}. Expected one of "
            f"{sorted(_UNIT_FACTORS)} (or a common alias)."
        )
    return canon


def _longest_true_run(mask: np.ndarray) -> int:
    """Length of the longest run of ``True`` values in a boolean array."""
    best = cur = 0
    for v in mask:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def normalize_demand(
    df: pd.DataFrame,
    ts_col: str,
    demand_col: str,
    unit: str,
) -> tuple[pd.Series | None, dict]:
    """Convert raw rows into a continuous hourly demand series.

    Steps: parse timestamps → sort → drop duplicate timestamps → resample to a
    continuous hourly grid (mean per hour if finer than hourly; reject if
    coarser) → compute gap and quality statistics.

    Values are kept in the user's original unit. Display labels are driven by
    ``project.demand_unit``, not hardcoded.

    Returns ``(series_or_None, report)``.  On a hard error (e.g. coarser than
    hourly, no parseable timestamps) the series is ``None`` and
    ``report["ok"]`` is ``False`` with a human-readable ``report["error"]``.
    """
    report: dict = {"ok": False}

    canon_unit = normalize_unit(unit)
    report["unit_input"] = unit
    report["unit_canonical"] = canon_unit

    raw_ts = pd.to_datetime(df[ts_col], errors="coerce")
    raw_demand = pd.to_numeric(df[demand_col], errors="coerce")
    n_rows = int(len(df))
    n_unparsed_ts = int(raw_ts.isna().sum())

    s = pd.Series(raw_demand.to_numpy(), index=raw_ts)
    s = s[~s.index.isna()]
    if s.empty:
        report["error"] = "No rows had a parseable timestamp."
        return None, report
    s = s.sort_index()

    dup_mask = s.index.duplicated(keep="first")
    n_dupes = int(dup_mask.sum())
    s = s[~dup_mask]

    deltas = s.index.to_series().diff().dropna()
    if deltas.empty:
        report["error"] = "Need at least two distinct timestamps."
        return None, report
    median_delta = deltas.median()
    one_hour = pd.Timedelta(hours=1)
    tol = pd.Timedelta(minutes=1)
    report["native_median_delta_minutes"] = round(median_delta.total_seconds() / 60.0, 3)

    if median_delta > one_hour + tol:
        report["resample_action"] = "rejected_coarser_than_hourly"
        report["error"] = (
            f"Data resolution is coarser than hourly "
            f"(median spacing ≈ {median_delta}). Hourly or finer data is required."
        )
        return None, report

    if median_delta < one_hour - tol:
        report["resample_action"] = "downsampled_to_hourly_mean"
    else:
        report["resample_action"] = "already_hourly"

    # Continuous hourly grid (gaps become NaN) — works for finer and exact data.
    hourly = s.resample("1h").mean()
    # Convert to MW so all downstream code (models, charts, KPIs) uses a single
    # consistent unit.  After conversion the canonical unit is always "MW".
    _factor = _UNIT_FACTORS.get(canon_unit, 1.0)
    if _factor != 1.0:
        hourly = hourly * _factor
    report["unit_canonical"] = "MW"   # always MW after conversion
    series_out = hourly.rename(DEMAND_COLUMN)

    n_expected = int(len(series_out))
    present_mask = series_out.notna().to_numpy()
    n_present = int(present_mask.sum())
    n_missing = n_expected - n_present
    report["n_expected"] = n_expected
    report["n_present"] = n_present
    report["n_missing"] = n_missing
    report["missing_pct"] = round(100.0 * n_missing / n_expected, 2) if n_expected else 0.0
    report["largest_consecutive_gap_hours"] = _longest_true_run(~present_mask)

    present = series_out.dropna()
    report["n_duplicate_timestamps"] = n_dupes
    report["n_unparsed_timestamps"] = n_unparsed_ts
    report["n_input_rows"] = n_rows
    report["n_negative"] = int((present < 0).sum())
    report["longest_zero_run_hours"] = _longest_true_run((series_out == 0).to_numpy())

    mean = float(present.mean())
    std = float(present.std())
    spike_threshold = mean + 6.0 * std
    report["spike_threshold"] = round(spike_threshold, 3)
    report["n_spikes"] = int((present > spike_threshold).sum())

    span = series_out.index.max() - series_out.index.min()
    span_days = span.total_seconds() / 86_400.0
    span_years = span_days / 365.25
    report["start"] = series_out.index.min().isoformat()
    report["end"] = series_out.index.max().isoformat()
    report["total_span_days"] = round(span_days, 2)
    report["span_years"] = round(span_years, 3)
    report["span_warning"] = bool(span_years < 1.5)

    report["demand_min"] = round(float(present.min()), 3)
    report["demand_max"] = round(float(present.max()), 3)
    report["demand_mean"] = round(mean, 3)

    report["ok"] = True
    return series_out, report


# ── 4. Multi-substation column detection ─────────────────────────────────────

def detect_substation_columns(
    df: pd.DataFrame,
    ts_col: str,
) -> list[str]:
    """Return all numeric columns (excluding *ts_col*) sorted by name.

    These are the candidates the wizard will pre-select as substation columns.
    Columns whose name matches the demand-hint vocabulary are returned first
    (higher priority) so a CSV that has both substation columns and a
    pre-aggregated total gets an intuitive default ordering.

    Every column is still a candidate — the user can deselect any of them in
    the mapping step.
    """
    hint_cols: list[str] = []
    other_cols: list[str] = []
    for col in df.columns:
        if col == ts_col:
            continue
        num = pd.to_numeric(df[col], errors="coerce")
        if float(num.notna().mean()) < 0.3:
            continue  # too many non-numeric values — skip
        if _name_score(col, _DEMAND_HINTS) > 0:
            hint_cols.append(col)
        else:
            other_cols.append(col)
    return sorted(hint_cols) + sorted(other_cols)


def normalize_multi_demand(
    df: pd.DataFrame,
    ts_col: str,
    sub_cols: list[str],
    unit: str,
) -> tuple[pd.DataFrame | None, dict[str, dict]]:
    """Convert raw rows into a continuous hourly DataFrame with one column per
    substation and an additional ``total_demand_mw`` aggregate column.

    All substation columns share the same timestamp column and hourly resampling
    logic as :func:`normalize_demand`.  Values are converted to MW using the
    same ``_UNIT_FACTORS`` table as the single-substation path.

    Returns ``(df_or_None, reports)`` where *reports* is a dict keyed by
    column name.  Each per-column report mirrors the :func:`normalize_demand`
    report schema.  The overall result is ``None`` (with an ``"_overall"`` error
    entry in *reports*) when the timestamp column is unparseable or the hourly
    grid cannot be established.
    """
    if not sub_cols:
        return None, {"_overall": {"ok": False, "error": "No substation columns provided."}}

    normalize_unit(unit)  # validate unit early; raises ValueError if invalid

    # Parse and validate timestamp once for all columns
    raw_ts = pd.to_datetime(df[ts_col], errors="coerce")
    n_unparsed = int(raw_ts.isna().sum())
    valid_ts = raw_ts.dropna()
    if valid_ts.empty:
        return None, {"_overall": {"ok": False, "error": "No rows had a parseable timestamp."}}

    span = valid_ts.max() - valid_ts.min()
    if span < pd.Timedelta(days=1):
        return None, {"_overall": {"ok": False, "error": "Timestamp span is less than one day."}}

    # Determine the native resolution once (use first substation column)
    first_num = pd.to_numeric(df[sub_cols[0]], errors="coerce")
    s_probe = pd.Series(first_num.to_numpy(), index=raw_ts)
    s_probe = s_probe[~s_probe.index.isna()].sort_index()
    deltas = s_probe.index.to_series().diff().dropna()
    if deltas.empty:
        return None, {"_overall": {"ok": False, "error": "Need at least two distinct timestamps."}}
    median_delta = deltas.median()
    one_hour = pd.Timedelta(hours=1)
    tol = pd.Timedelta(minutes=1)
    if median_delta > one_hour + tol:
        return None, {
            "_overall": {
                "ok": False,
                "error": (
                    f"Data resolution is coarser than hourly "
                    f"(median spacing ≈ {median_delta}). Hourly or finer data is required."
                ),
            }
        }

    reports: dict[str, dict] = {}
    hourly_series: dict[str, pd.Series] = {}

    for col in sub_cols:
        raw_vals = pd.to_numeric(df[col], errors="coerce")
        s = pd.Series(raw_vals.to_numpy(), index=raw_ts)
        s = s[~s.index.isna()].sort_index()

        dup_mask = s.index.duplicated(keep="first")
        n_dupes = int(dup_mask.sum())
        s = s[~dup_mask]

        hourly = s.resample("1h").mean()
        n_expected = int(len(hourly))
        n_present = int(hourly.notna().sum())
        n_missing = n_expected - n_present
        present = hourly.dropna()

        mean = abs(float(present.mean())) if not present.empty else 1.0
        spike_threshold = float(present.mean()) + 6.0 * float(present.std()) if not present.empty else np.inf

        rep: dict = {
            "ok": True,
            "n_expected": n_expected,
            "n_present": n_present,
            "n_missing": n_missing,
            "missing_pct": round(100.0 * n_missing / n_expected, 2) if n_expected else 0.0,
            "n_duplicate_timestamps": n_dupes,
            "n_unparsed_timestamps": n_unparsed,
            "n_negative": int((present < 0).sum()) if not present.empty else 0,
            "demand_min": round(float(present.min()), 3) if not present.empty else 0.0,
            "demand_max": round(float(present.max()), 3) if not present.empty else 0.0,
            "demand_mean": round(float(present.mean()), 3) if not present.empty else 0.0,
            "n_spikes": int((present > spike_threshold).sum()) if not present.empty else 0,
            "largest_consecutive_gap_hours": _longest_true_run(hourly.isna().to_numpy()),
            "span_warning": bool((hourly.index.max() - hourly.index.min()) < pd.Timedelta(days=365 * 1.5)),
        }
        reports[col] = rep
        hourly_series[col] = hourly

    # Build the combined DataFrame on the union of all hourly indices
    out = pd.DataFrame(hourly_series)
    out.index.name = "timestamp"

    # Convert to MW so all downstream code uses a single consistent unit.
    _factor = _UNIT_FACTORS.get(normalize_unit(unit), 1.0)
    if _factor != 1.0:
        out = out * _factor

    # Aggregate: sum across all substation columns, NaN if ALL substations are NaN
    out[AGGREGATE_COLUMN] = out[sub_cols].sum(axis=1, min_count=1)

    return out, reports


# ── 5. Project id slug ──────────────────────────────────────────────────────────

def slugify_project_id(name: str, existing: set[str] | None = None) -> str:
    """Turn a project name into a filesystem-safe, unique project id.

    ascii, lowercase, hyphen-separated.  Uniqueness is enforced against
    *existing* (defaults to the ids already under ``projects/``) by appending
    ``-2``, ``-3``, ….
    """
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    if not text:
        text = "project"

    if existing is None:
        try:
            from src.project import Project  # local import keeps this module pure
            existing = set(Project.list_all())
        except Exception:
            existing = set()

    candidate = text
    n = 2
    while candidate in existing:
        candidate = f"{text}-{n}"
        n += 1
    return candidate
