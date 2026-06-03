"""
"New project" setup wizard UI (Phase B).

A multi-step Streamlit wizard that turns an uploaded demand CSV into a project
*skeleton* (demand-only).  All heavy lifting lives in the pure-logic modules
:mod:`src.ingest` and :mod:`src.geocode`; this file is the thin Streamlit layer.

Public entry points used by ``app.py``:
    render_wizard(...)              — the 5-step builder, shown in the main area
    render_incomplete_project(...)  — the graceful "setup incomplete" panel

Steps: 1 Upload · 2 Map & units · 3 Quality · 4 City & location · 5 Split & create.

The wizard intentionally stops at the demand-only skeleton — weather enrichment
and model training are later phases.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.geocode import geocode_city
from src.ingest import (
    AGGREGATE_COLUMN,
    detect_columns,
    detect_substation_columns,
    normalize_demand,
    normalize_multi_demand,
    parse_csv,
    slugify_project_id,
)
from src.project import Project
from src.weather_history import enrich_with_weather

# Gradyent palette — identical hex to app.py (no new brand colours introduced).
NAVY = "#11224D"
PINK = "#E31B54"
TEAL = "#2EC4B6"
GRAY_MID = "#9CA3AF"
TEXT_MUT = "#4B5563"
GRID = "rgba(17, 34, 77, 0.10)"
GREEN = "#16a34a"
AMBER = "#ca8a04"
RED = "#dc2626"

# Hard floor / warning thresholds for usable history length (years).
# < 6 months: blocked (not enough to train).  6–12 months: allowed with a warning.
SPAN_HARD_FLOOR_YEARS = 0.5   # 6 months
SPAN_WARN_YEARS = 1.0         # 1 year

_UNIT_OPTIONS = {
    "MW": "MW",
    "kW": "kW",
    "kWh per hour": "kWh_per_hour",
    "W": "W",
}

_TOTAL_STEPS = 5
_STEP_TITLES = {
    1: "Upload demand timeseries",
    2: "Map columns & units",
    3: "Data quality report",
    4: "City & location",
    5: "Train/test split & create",
}

# All wizard-scoped session keys carry this prefix so they are easy to clear.
_PREFIX = "b_"


# ── State helpers ──────────────────────────────────────────────────────────────

def _reset_builder() -> None:
    """Drop builder_mode + every wizard-scoped session key (keeps app state)."""
    st.session_state.pop("builder_mode", None)
    for key in [k for k in st.session_state.keys() if k.startswith(_PREFIX)]:
        st.session_state.pop(key, None)


def _goto(step: int) -> None:
    st.session_state["b_step"] = step
    st.rerun()


def _demand_line(series: pd.Series, height: int = 280, unit: str = "MW") -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=series.index, y=series.values,
            mode="lines", line=dict(color=NAVY, width=1.2),
            hovertemplate=f"%{{y:.1f}} {unit}<extra></extra>",
        )
    )
    fig.update_layout(
        height=height,
        margin=dict(l=48, r=16, t=8, b=28),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_yaxes(title_text=f"Heat demand · {unit}", gridcolor=GRID,
                     title_font=dict(color=TEXT_MUT), tickfont=dict(color=TEXT_MUT))
    fig.update_xaxes(gridcolor=GRID, tickfont=dict(color=TEXT_MUT))
    return fig


def _progress_header(step: int) -> None:
    st.caption(f"Step {step} of {_TOTAL_STEPS} · {_STEP_TITLES[step]}")
    st.progress(step / _TOTAL_STEPS)


# ── Step 1: Upload ──────────────────────────────────────────────────────────────

def _step_upload() -> None:
    st.markdown("#### 1 · Upload demand timeseries")
    st.caption(
        "Upload a CSV with a timestamp column and one or more heat-demand columns "
        "(hourly or finer, up to ~100 columns). You will map the columns on the next step."
    )
    up = st.file_uploader("Demand CSV", type=["csv"], key="b_uploader")
    if up is None:
        st.session_state.pop("b_df", None)
        return

    try:
        df = parse_csv(up)
    except Exception as exc:  # noqa: BLE001 - surface any parse failure to the user
        st.error(f"Could not read this CSV: {exc}")
        return

    st.session_state["b_df"] = df
    detected = detect_columns(df)
    st.session_state["b_detected"] = detected

    n_numeric = sum(
        1 for c in df.columns
        if c != detected.get("timestamp") and str(df[c].dtype).startswith(("int", "float"))
    )

    st.markdown("**Preview**")
    st.dataframe(df.head(10), use_container_width=True)
    c_rows, c_cols, c_num = st.columns(3)
    c_rows.metric("Rows", f"{len(df):,}")
    c_cols.metric("Columns", f"{df.shape[1]}")
    c_num.metric("Numeric columns", f"{n_numeric}")
    st.caption(
        f"Detected timestamp column: **{detected['timestamp'] or '—'}**  ·  "
        f"demand column: **{detected['demand'] or '—'}**"
    )

    if st.button("Next →", type="primary", key="b_s1_next"):
        _goto(2)


# ── Step 2: Map & units ─────────────────────────────────────────────────────────

def _step_map() -> None:
    df = st.session_state.get("b_df")
    if df is None:
        st.info("Upload a CSV first.")
        if st.button("← Back", key="b_s2_back0"):
            _goto(1)
        return

    st.markdown("#### 2 · Map columns & units")
    cols = list(df.columns)
    detected = st.session_state.get("b_detected", {})

    def _idx(col: str | None) -> int:
        return cols.index(col) if col in cols else 0

    ts_col = st.selectbox(
        "Timestamp column", cols, index=_idx(detected.get("timestamp")), key="b_ts_sel"
    )
    unit_label = st.selectbox(
        "Demand unit", list(_UNIT_OPTIONS.keys()), key="b_unit_sel",
        help="The unit your raw data is stored in. Values are automatically converted to **MW** for storage and display.",
    )

    # ── Mode selector ─────────────────────────────────────────────────────────
    n_numeric = sum(
        1 for c in df.columns
        if c != ts_col and str(df[c].dtype).startswith(("int", "float"))
    )
    default_mode = "Multiple substations" if n_numeric > 3 else "Single substation"
    mode_options = ["Single substation", "Multiple substations"]
    mode_label = st.radio(
        "Project type",
        mode_options,
        index=mode_options.index(st.session_state.get("b_mode_label", default_mode)),
        key="b_mode_radio",
        help="Single: one demand column. Multiple: one column per substation (up to ~100).",
        horizontal=True,
    )
    is_multi = mode_label == "Multiple substations"

    if not is_multi:
        # ── Single mode ───────────────────────────────────────────────────────
        demand_col = st.selectbox(
            "Demand column", cols, index=_idx(detected.get("demand")), key="b_demand_sel"
        )
        st.dataframe(df[[ts_col, demand_col]].head(8), use_container_width=True)

        c_back, c_next = st.columns([1, 1])
        if c_back.button("← Back", key="b_s2_back"):
            _goto(1)
        if c_next.button("Validate →", type="primary", key="b_s2_next"):
            st.session_state["b_ts_col"] = ts_col
            st.session_state["b_demand_col"] = demand_col
            st.session_state["b_unit"] = _UNIT_OPTIONS[unit_label]
            st.session_state["b_mode_label"] = mode_label
            st.session_state["b_is_multi"] = False
            st.session_state.pop("b_series", None)
            st.session_state.pop("b_multi_df", None)
            st.session_state.pop("b_report", None)
            _goto(3)
    else:
        # ── Multi mode ────────────────────────────────────────────────────────
        st.caption(
            "All numeric columns are pre-selected as substations. "
            "Uncheck columns to exclude them, and edit the **Display name** column "
            "to give each substation a friendly label."
        )
        sub_candidates = detect_substation_columns(df, ts_col)

        # Build (or recall) the mapping table in session state so edits persist
        # across Streamlit reruns within this step.
        if "b_sub_mapping" not in st.session_state or st.session_state.get("b_ts_sel_prev") != ts_col:
            st.session_state["b_sub_mapping"] = pd.DataFrame({
                "column": sub_candidates,
                "display_name": sub_candidates,
                "include": [True] * len(sub_candidates),
            })
            st.session_state["b_ts_sel_prev"] = ts_col

        mapping = st.data_editor(
            st.session_state["b_sub_mapping"],
            key="b_sub_mapping_editor",
            column_config={
                "column": st.column_config.TextColumn("CSV column", disabled=True),
                "display_name": st.column_config.TextColumn("Display name"),
                "include": st.column_config.CheckboxColumn("Include", default=True),
            },
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
        )
        st.session_state["b_sub_mapping"] = mapping

        included = mapping[mapping["include"]]
        st.caption(
            f"**{len(included)}** substation(s) selected out of {len(mapping)}."
        )

        c_back, c_next = st.columns([1, 1])
        if c_back.button("← Back", key="b_s2_back_multi"):
            _goto(1)
        if c_next.button("Validate →", type="primary", key="b_s2_next_multi",
                         disabled=len(included) == 0):
            st.session_state["b_ts_col"] = ts_col
            st.session_state["b_unit"] = _UNIT_OPTIONS[unit_label]
            st.session_state["b_mode_label"] = mode_label
            st.session_state["b_is_multi"] = True
            # Store the selected substation mapping for step 3 + step 5.
            st.session_state["b_sub_cols"] = list(included["column"])
            st.session_state["b_sub_names"] = dict(zip(included["column"], included["display_name"]))
            st.session_state.pop("b_series", None)
            st.session_state.pop("b_multi_df", None)
            st.session_state.pop("b_report", None)
            _goto(3)


# ── Step 3: Quality report ──────────────────────────────────────────────────────

def _metric_with_cue(col, label: str, value: str, color: str) -> None:
    col.metric(label, value)
    col.markdown(
        f"<div style='height:4px;border-radius:2px;background:{color};"
        f"margin-top:-8px;'></div>",
        unsafe_allow_html=True,
    )


def _multi_quality_table(reports: dict[str, dict], sub_names: dict[str, str]) -> pd.DataFrame:
    """Build a tidy summary DataFrame from per-substation quality reports."""
    rows = []
    for col, rep in reports.items():
        if col.startswith("_"):
            continue
        name = sub_names.get(col, col)
        rows.append({
            "Substation": name,
            "Rows": rep.get("n_present", 0),
            "Missing %": rep.get("missing_pct", 0.0),
            "Largest gap (h)": rep.get("largest_consecutive_gap_hours", 0),
            "Negatives": rep.get("n_negative", 0),
            "Spikes": rep.get("n_spikes", 0),
            "Min": rep.get("demand_min", 0.0),
            "Max": rep.get("demand_max", 0.0),
            "Mean": rep.get("demand_mean", 0.0),
        })
    return pd.DataFrame(rows)


def _step_quality() -> None:
    df = st.session_state.get("b_df")
    ts_col = st.session_state.get("b_ts_col")
    unit = st.session_state.get("b_unit")
    is_multi = st.session_state.get("b_is_multi", False)

    if df is None or ts_col is None:
        st.info("Map the columns first.")
        if st.button("← Back", key="b_s3_back0"):
            _goto(2)
        return

    st.markdown("#### 3 · Data quality report")

    if not is_multi:
        # ── Single mode (unchanged) ───────────────────────────────────────────
        demand_col = st.session_state.get("b_demand_col")
        series, report = normalize_demand(df, ts_col, demand_col, unit)
        st.session_state["b_series"] = series
        st.session_state["b_report"] = report

        if not report.get("ok"):
            st.error(report.get("error", "Could not normalise the demand series."))
            st.caption(f"Unit conversion: {report.get('unit_conversion', '—')}")
            if st.button("← Back", key="b_s3_back_err"):
                _goto(2)
            return

        span = float(report["span_years"])
        missing_pct = float(report["missing_pct"])
        largest_gap = int(report["largest_consecutive_gap_hours"])

        span_color = GREEN if span >= SPAN_WARN_YEARS else AMBER if span >= SPAN_HARD_FLOOR_YEARS else RED
        miss_color = GREEN if missing_pct < 1 else AMBER if missing_pct < 5 else RED
        gap_color = GREEN if largest_gap <= 6 else AMBER if largest_gap <= 48 else RED
        neg_color = GREEN if report["n_negative"] == 0 else RED
        spike_color = GREEN if report["n_spikes"] == 0 else AMBER

        r1 = st.columns(3)
        _metric_with_cue(r1[0], "Hourly rows", f"{report['n_present']:,}", GREEN)
        _metric_with_cue(r1[1], "Span (years)", f"{span:.2f}", span_color)
        _metric_with_cue(r1[2], "Missing hours", f"{report['n_missing']:,} ({missing_pct:.1f}%)", miss_color)
        r2 = st.columns(3)
        _metric_with_cue(r2[0], "Largest gap (h)", f"{largest_gap}", gap_color)
        _metric_with_cue(r2[1], "Negative values", f"{report['n_negative']:,}", neg_color)
        _metric_with_cue(r2[2], "Spikes (>μ+6σ)", f"{report['n_spikes']:,}", spike_color)

        st.caption(
            f"Resampling: {report['resample_action'].replace('_', ' ')}  ·  "
            f"unit: {report['unit_canonical']}  ·  "
            f"range {report['start'][:10]} → {report['end'][:10]}"
        )

        st.plotly_chart(_demand_line(series.dropna(), unit=report.get("unit_canonical", "MW")), use_container_width=True)
        blocked = span < SPAN_HARD_FLOOR_YEARS
        if blocked:
            st.error(
                f"Only {span * 12:.0f} months of data — at least 6 months is required "
                "to train a usable model. Upload a longer series."
            )
        elif span < SPAN_WARN_YEARS:
            st.warning(
                f"{span * 12:.0f} months of data — less than the recommended 1 year. "
                "Seasonal patterns may be under-represented, but you can proceed."
            )

        c_back, c_next = st.columns([1, 1])
        if c_back.button("← Back", key="b_s3_back"):
            _goto(2)
        if c_next.button("Next →", type="primary", key="b_s3_next", disabled=blocked):
            _goto(4)

    else:
        # ── Multi mode ────────────────────────────────────────────────────────
        sub_cols = st.session_state.get("b_sub_cols", [])
        sub_names = st.session_state.get("b_sub_names", {})

        multi_df, reports = normalize_multi_demand(df, ts_col, sub_cols, unit)
        st.session_state["b_multi_df"] = multi_df
        st.session_state["b_multi_reports"] = reports

        overall_err = reports.get("_overall", {})
        if multi_df is None:
            st.error(overall_err.get("error", "Could not normalise the substation data."))
            if st.button("← Back", key="b_s3_multi_back_err"):
                _goto(2)
            return

        n_subs = len(sub_cols)
        agg = multi_df[AGGREGATE_COLUMN].dropna()
        total_rows = int(agg.notna().sum()) if not agg.empty else 0
        any_gap = max(r.get("largest_consecutive_gap_hours", 0) for r in reports.values() if not isinstance(r, dict) or not r.get("_overall"))
        n_warn = sum(
            1 for c, r in reports.items()
            if not c.startswith("_") and (r.get("missing_pct", 0) > 5 or r.get("n_negative", 0) > 0)
        )

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        _metric_with_cue(kpi1, "Substations", str(n_subs), GREEN)
        _metric_with_cue(kpi2, "Aggregate rows", f"{total_rows:,}", GREEN)
        _metric_with_cue(kpi3, "Substations with issues", str(n_warn), AMBER if n_warn > 0 else GREEN)
        _metric_with_cue(kpi4, "Largest gap (any sub, h)", str(any_gap), AMBER if any_gap > 6 else GREEN)

        st.markdown("**Aggregate demand (sum of all substations)**")
        st.plotly_chart(
            _demand_line(agg, unit="MW", height=220),
            use_container_width=True,
        )

        st.markdown("**Per-substation quality summary**")
        tbl = _multi_quality_table(reports, sub_names)
        st.dataframe(
            tbl.style.background_gradient(subset=["Missing %"], cmap="RdYlGn_r", vmin=0, vmax=10),
            use_container_width=True,
            hide_index=True,
        )

        span_days = (agg.index.max() - agg.index.min()).total_seconds() / 86_400 if not agg.empty else 0.0
        span_years = span_days / 365.25
        blocked = span_years < SPAN_HARD_FLOOR_YEARS
        if blocked:
            st.error(
                f"Only {span_years * 12:.0f} months of aggregate data — "
                "at least 6 months is required."
            )
        elif span_years < SPAN_WARN_YEARS:
            st.warning(
                f"{span_years * 12:.0f} months of data — less than the recommended 1 year."
            )
        if n_warn > 0:
            st.info(
                f"{n_warn} substation(s) have >5% missing data or negative values. "
                "These will train with reduced coverage — you can proceed."
            )

        c_back, c_next = st.columns([1, 1])
        if c_back.button("← Back", key="b_s3_back_multi"):
            _goto(2)
        if c_next.button("Next →", type="primary", key="b_s3_next_multi", disabled=blocked):
            _goto(4)


# ── Step 4: City & location ─────────────────────────────────────────────────────

def _validate_location(lat, lon, city) -> tuple[bool, str | None]:
    """Validate a resolved project location.

    A location is usable only when a city/network name is given and the
    coordinates are real numbers within range and not the (0, 0) "null island"
    (Open-Meteo returns 0,0-equivalent garbage there, which is what produced the
    broken ``flensburg-2`` project — equator weather, all-NaN snow).  This single
    gate is reused by BOTH the step-4 *Next* button and the step-5 *Create*
    button so no project can ever be written without a resolved location.

    Returns ``(ok, error_message)``; ``error_message`` is ``None`` when ok.
    """
    if not str(city or "").strip():
        return False, "Enter a city / network name."
    if lat is None or lon is None:
        return False, "Search & select a city, or enter coordinates manually."
    try:
        latf, lonf = float(lat), float(lon)
    except (TypeError, ValueError):
        return False, "Coordinates must be numbers."
    if abs(latf) > 90.0 or abs(lonf) > 180.0:
        return False, "Coordinates out of range (need |lat| ≤ 90, |lon| ≤ 180)."
    if latf == 0.0 and lonf == 0.0:
        return False, (
            "Coordinates (0, 0) are not a real location — search & select a "
            "city or enter non-zero coordinates."
        )
    return True, None


def _fmt_candidate(c: dict) -> str:
    parts = [c.get("name"), c.get("admin1"), c.get("country")]
    loc = ", ".join(p for p in parts if p)
    lat, lon = c.get("latitude"), c.get("longitude")
    coords = f"{lat:.3f}, {lon:.3f}" if lat is not None and lon is not None else "?"
    return f"{loc} · {coords} · {c.get('timezone', '?')}"


def _step_location() -> None:
    st.markdown("#### 4 · City & location")
    st.caption(
        "Search for the city to fetch coordinates and timezone, or enter them "
        "manually. These drive the Open-Meteo weather fetch in the next phase."
    )

    def _do_geocode_search() -> None:
        q = st.session_state.get("b_city_query", "").strip()
        if not q:
            return
        found = geocode_city(q)
        st.session_state["b_geo_results"] = found
        st.session_state["b_geo_searched"] = True
        st.session_state.pop("b_geo_choice", None)
        if found:
            st.session_state["b_manual_loc"] = False

    c_query, c_btn = st.columns([3, 1])
    query = c_query.text_input(
        "City name",
        key="b_city_query",
        placeholder="e.g. Aalborg",
        on_change=_do_geocode_search,
    )
    if c_btn.button("Search", key="b_geo_search"):
        _do_geocode_search()
        st.rerun()

    results = st.session_state.get("b_geo_results", [])
    selected: dict | None = None

    if results:
        idx = st.radio(
            "Matches",
            options=list(range(len(results))),
            format_func=lambda i: _fmt_candidate(results[i]),
            key="b_geo_choice",
        )
        selected = results[int(idx)]
    elif st.session_state.get("b_geo_searched"):
        st.info(
            "No matches (or the geocoding service was unreachable). "
            "Enter the coordinates manually below."
        )

    with st.expander("Enter coordinates manually", expanded=False):
        manual = st.checkbox(
            "Use manual coordinates instead of a search result",
            value=False,
            key="b_manual_loc",
        )
        m_city = st.text_input(
            "City / network name",
            value=(selected.get("name") if selected else query),
            key="b_manual_city",
        )
        mc1, mc2, mc3 = st.columns(3)
        m_lat = mc1.number_input(
            "Latitude", value=float(selected["latitude"]) if selected else 0.0,
            min_value=-90.0, max_value=90.0, format="%.4f", key="b_manual_lat",
        )
        m_lon = mc2.number_input(
            "Longitude", value=float(selected["longitude"]) if selected else 0.0,
            min_value=-180.0, max_value=180.0, format="%.4f", key="b_manual_lon",
        )
        m_tz = mc3.text_input(
            "Timezone",
            value=(selected.get("timezone") if selected else "UTC") or "UTC",
            key="b_manual_tz",
        )

    if st.session_state.get("b_manual_loc") or not selected:
        eff_city = st.session_state.get("b_manual_city") or query
        eff_lat = st.session_state.get("b_manual_lat")
        eff_lon = st.session_state.get("b_manual_lon")
        eff_tz = st.session_state.get("b_manual_tz") or "UTC"
        # Manual entry has no reliable ISO country code; holidays are skipped.
        eff_cc = None
    else:
        eff_city = selected.get("name") or query
        eff_lat = selected.get("latitude")
        eff_lon = selected.get("longitude")
        eff_tz = selected.get("timezone") or "UTC"
        eff_cc = selected.get("country_code")

    valid, loc_msg = _validate_location(eff_lat, eff_lon, eff_city)
    if eff_lat is not None and eff_lon is not None:
        try:
            st.caption(
                f"Selected location: **{eff_city or '—'}** · {float(eff_lat):.4f}, "
                f"{float(eff_lon):.4f} · {eff_tz}"
            )
        except (TypeError, ValueError):
            pass

    c_back, c_next = st.columns([1, 1])
    if c_back.button("← Back", key="b_s4_back"):
        _goto(3)
    if c_next.button("Next →", type="primary", key="b_s4_next", disabled=not valid):
        # Re-validate inside the handler so the location is captured only when it
        # is genuinely resolved (belt-and-suspenders behind the disabled gate).
        ok, msg = _validate_location(eff_lat, eff_lon, eff_city)
        if not ok:
            st.error(msg)
            return
        st.session_state["b_city"] = eff_city
        st.session_state["b_lat"] = float(eff_lat)
        st.session_state["b_lon"] = float(eff_lon)
        st.session_state["b_tz"] = eff_tz
        st.session_state["b_country_code"] = (
            str(eff_cc).strip().upper() if eff_cc else None
        )
        _goto(5)
    if not valid:
        st.caption(loc_msg or "Search & select a city to continue.")


# ── Step 5: Split & create ───────────────────────────────────────────────────────

def _step_create() -> None:
    is_multi = st.session_state.get("b_is_multi", False)
    series = st.session_state.get("b_series")
    multi_df = st.session_state.get("b_multi_df")
    demand_available = (multi_df is not None) if is_multi else (series is not None)

    if not demand_available or st.session_state.get("b_lat") is None:
        st.info("Complete the earlier steps first.")
        if st.button("← Back", key="b_s5_back0"):
            _goto(4)
        return

    st.markdown("#### 5 · Train/test split & create")

    # Use the aggregate (or single) series to drive the date-range picker
    if is_multi:
        present_all = multi_df[AGGREGATE_COLUMN].dropna()
    else:
        present_all = series.dropna()

    min_d = present_all.index.min().date()
    max_d = present_all.index.max().date()
    total_days = (max_d - min_d).days

    if min_d >= max_d or total_days < 8:
        st.error("Not enough data range to choose a train/test split.")
        return

    # Default: leave last 12 months as test. If data is shorter, default to
    # using 75% for training.  Slider goes all the way to max_d so the user
    # can opt for NO test holdout (use every row for training).
    ideal_test_start = (present_all.index.max() - pd.DateOffset(months=12)).date()
    default_test = ideal_test_start if ideal_test_start > min_d else (
        min_d + pd.Timedelta(days=max(1, int(total_days * 0.75)))
    )
    default_test = min(default_test, max_d)

    test_start = st.slider(
        "Test set start — drag to the far right to use ALL data for training (no holdout)",
        min_value=min_d + pd.Timedelta(days=1),
        max_value=max_d,                          # allow full-data training
        value=default_test,
        format="YYYY-MM-DD",
    )
    # Match the index timezone (may be UTC-aware or naive) so the comparison never fails.
    _tz = present_all.index.tz
    _ts_split = pd.Timestamp(test_start).tz_localize(_tz) if _tz is not None else pd.Timestamp(test_start)
    n_train = int((present_all.index < _ts_split).sum())
    n_test  = int((present_all.index >= _ts_split).sum())

    train_days = (test_start - min_d).days
    test_days  = (max_d - test_start).days

    if test_start >= max_d:
        st.caption(f"Train: {min_d} → {max_d} ({n_train:,} h · {train_days} days) · no test holdout")
        st.info("All data will be used for training — no held-out evaluation period. MAPE will not be calculated.", icon="ℹ️")
    else:
        st.caption(
            f"Train: {min_d} → {test_start} ({n_train:,} h · {train_days} days)  ·  "
            f"Test: {test_start} → {max_d} ({n_test:,} h · {test_days} days)"
        )
        if train_days < 90:
            st.warning(
                f"Only {train_days} days of training data — the model may perform poorly. "
                "Consider uploading a longer history or moving the split earlier."
            )
        elif train_days < 180:
            st.warning(
                f"{train_days} days ({train_days // 30} months) of training data — "
                "less than the recommended 6 months. Seasonal patterns may be under-represented."
            )
        if test_days < 14:
            st.warning(f"Only {test_days} days of test data — MAPE evaluation will be very noisy.")

    st.divider()
    default_name = st.session_state.get("b_city") or "New project"
    name = st.text_input("Project name", value=default_name, key="b_name")

    existing = set(Project.list_all())
    if "b_id_user" not in st.session_state:
        st.session_state["b_id_user"] = slugify_project_id(name, existing)
    project_id = st.text_input(
        "Project id (folder name)", key="b_id_user",
        help="Lowercase, hyphenated, unique. Used as projects/<id>/.",
    )
    clean_id = slugify_project_id(project_id or name, existing=set())
    id_taken = clean_id in existing
    if project_id != clean_id:
        st.caption(f"Will be stored as: **{clean_id}**")
    if id_taken:
        st.error(f"A project with id **{clean_id}** already exists — choose another.")

    loc_ok, loc_msg = _validate_location(
        st.session_state.get("b_lat"),
        st.session_state.get("b_lon"),
        st.session_state.get("b_city"),
    )
    if not loc_ok:
        st.error(
            f"{loc_msg} Go back to **step 4** and search & select a city before "
            "creating the project."
        )

    can_create = bool(name.strip()) and bool(clean_id) and not id_taken and loc_ok

    c_back, c_create = st.columns([1, 1])
    if c_back.button("← Back", key="b_s5_back"):
        _goto(4)
    if c_create.button("Create forecaster", type="primary", key="b_create", disabled=not can_create):
        ok, msg = _validate_location(
            st.session_state.get("b_lat"),
            st.session_state.get("b_lon"),
            st.session_state.get("b_city"),
        )
        if not ok:
            st.error(f"Cannot create project — {msg}")
            return
        try:
            if is_multi:
                sub_cols = st.session_state.get("b_sub_cols", [])
                sub_names = st.session_state.get("b_sub_names", {})
                substations = [
                    {
                        "id": slugify_project_id(sub_names.get(c, c), existing=set()),
                        "name": sub_names.get(c, c),
                        "column": c,
                    }
                    for c in sub_cols
                ]
                Project.create(
                    project_id=clean_id,
                    name=name.strip(),
                    city=st.session_state.get("b_city") or name.strip(),
                    lat=st.session_state["b_lat"],
                    lon=st.session_state["b_lon"],
                    timezone=st.session_state.get("b_tz") or "UTC",
                    demand=multi_df,
                    test_start=str(test_start),
                    demand_unit="MW",
                    target_column=AGGREGATE_COLUMN,
                    country_code=st.session_state.get("b_country_code"),
                    mode="multi",
                    substations=substations,
                )
            else:
                Project.create(
                    project_id=clean_id,
                    name=name.strip(),
                    city=st.session_state.get("b_city") or name.strip(),
                    lat=st.session_state["b_lat"],
                    lon=st.session_state["b_lon"],
                    timezone=st.session_state.get("b_tz") or "UTC",
                    demand=series,
                    test_start=str(test_start),
                    demand_unit="MW",
                    country_code=st.session_state.get("b_country_code"),
                )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not create project: {exc}")
            return
        st.session_state["pending_project_switch"] = clean_id
        st.session_state["builder_created_id"] = clean_id
        _reset_builder()
        st.rerun()


# ── Public entry points ─────────────────────────────────────────────────────────

_STEP_FUNCS = {
    1: _step_upload,
    2: _step_map,
    3: _step_quality,
    4: _step_location,
    5: _step_create,
}


def render_wizard(header_fn=None) -> None:
    """Render the multi-step "new project" wizard in the main area."""
    if header_fn is not None:
        header_fn("Create a new project")
    else:
        st.markdown("## Create a new project")

    if st.button("← Back to dashboard", key="b_exit"):
        _reset_builder()
        st.rerun()

    step = int(st.session_state.get("b_step", 1))
    step = min(max(step, 1), _TOTAL_STEPS)
    _progress_header(step)
    st.divider()
    _STEP_FUNCS[step]()


def _weather_done(project: Project) -> bool:
    """True once the enriched demand+weather CSV exists for this project."""
    return project.data_path.exists()


def _fetch_weather_for_project(project: Project) -> None:
    """Fetch ERA5 weather, write the enriched CSV, advance the stage, rerun.

    Wrapped in ``st.status`` with a progress callback.  Surfaces the alignment
    correlation (and any timezone warning) so the operator can trust the join.

    For multi-substation projects the ``target_column`` is ``total_demand_mw``
    (the row-sum of all substations) so the alignment check uses the aggregate
    rather than an individual substation.
    """
    try:
        demand = pd.read_csv(
            project.demand_only_path, index_col="timestamp", parse_dates=True
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read the demand series: {exc}")
        return

    # For multi-mode the target column for weather alignment is the aggregate.
    target_col = project.target_column  # already set to AGGREGATE_COLUMN for multi

    with st.status("Fetching historical weather from Open-Meteo…", expanded=True) as box:
        bar = st.progress(0.0, text="Starting…")

        def _progress(frac: float, text: str) -> None:
            bar.progress(min(max(frac, 0.0), 1.0), text=text)

        enriched, report = enrich_with_weather(
            demand,
            lat=project.lat,
            lon=project.lon,
            timezone=project.timezone,
            country_code=project.country_code,
            progress=_progress,
            target_column=target_col,
        )

        if not report.get("ok") or enriched is None:
            box.update(label="Weather fetch failed", state="error")
            st.error(report.get("error", "Unknown error fetching weather."))
            return

        project.attach_weather(enriched)
        corr = report.get("alignment_corr", float("nan"))
        box.update(label="Weather enrichment complete", state="complete")

    st.success(
        f"Enriched **{report['n_rows']:,}** hours of weather "
        f"({report['date_start'][:10]} → {report['date_end'][:10]})."
    )
    corr_ok = corr < -0.4
    st.metric(
        "Alignment check · corr(demand, temperature)",
        f"{corr:.3f}",
        delta="strongly negative ✓" if corr_ok else "weak — check timezone",
        delta_color="normal" if corr_ok else "inverse",
        help="Heating demand should be strongly anti-correlated with temperature. "
        "A non-negative value usually means the weather/demand timezones disagree.",
    )
    for warn in report.get("warnings", []):
        st.warning(warn)
    st.rerun()


def render_incomplete_project(project: Project, header_fn=None) -> None:
    """Friendly "setup incomplete" panel for a project without weather/models."""
    if header_fn is not None:
        header_fn(f"{project.name} · setup incomplete")
    else:
        st.markdown(f"## {project.name} · setup incomplete")

    weather_done = _weather_done(project)

    created_id = st.session_state.pop("builder_created_id", None)
    if created_id == project.id:
        st.success(
            f"Project **{project.name}** created. The demand series is uploaded — "
            "fetch the historical weather below to make the project trainable."
        )

    if weather_done:
        if project.mode == "multi":
            st.info(
                "Weather is enriched and the dataset is ready. "
                "Click **Train all substations** below to train all models at once "
                "(batch mode — fast, 100 trees, auto feature selection).",
                icon="🚂",
            )
        else:
            st.info(
                "Weather is enriched and the dataset is ready. Train the 48 horizon "
                "models from the **🔁 Train model** expander in the sidebar to unlock "
                "the forecasting dashboard.",
                icon="🚂",
            )
    else:
        st.info(
            "This project is not ready for forecasting yet. Fetch the historical "
            "weather to build the model-ready dataset, then train the models.",
            icon="🛠️",
        )

    st.markdown("#### Setup checklist")
    weather_line = (
        "- ✅ **Weather data** — Open-Meteo ERA5 enrichment complete\n"
        if weather_done
        else "- ⬜ **Weather data** — fetch Open-Meteo ERA5 history (below)\n"
    )
    if project.mode == "multi":
        train_hint = "(click **Train all substations** below)" if weather_done else "(after weather)"
    else:
        train_hint = "(use **🔁 Train model** in the sidebar)" if weather_done else "(after weather)"
    st.markdown(
        "- ✅ **Demand uploaded** — hourly demand series stored\n"
        + weather_line
        + f"- ⬜ **Models trained** — 48 horizon models {train_hint}"
    )

    # ── Batch train button for multi-substation projects ─────────────────────
    if weather_done and project.mode == "multi":
        st.divider()
        st.markdown("#### Train all substations")
        st.caption(
            "Trains 48 horizon models for every substation simultaneously using fast settings "
            "(100 trees, auto feature selection). Models are versioned with the `batch_` prefix. "
            "You can retrain individual substations at any time from the Substation Detail sidebar."
        )
        _batch_wo = st.checkbox(
            "Weather-only models (enables live forecasting)",
            key="incomplete_batch_wo",
            help="Train weather-only models (no demand lags) in addition to the standard models.",
        )
        if st.button("🚀 Train all substations", type="primary", key="incomplete_batch_train"):
            import subprocess, sys
            _substations = project.substations if hasattr(project, "substations") else []
            _n = len(_substations)
            _all_ok = True
            with st.status(f"Training {_n} substation(s) — fast · 100 trees…", expanded=True) as _box:
                _log = st.empty()
                _all_lines: list[str] = []
                for _i, _sub in enumerate(_substations, 1):
                    _sub_id = _sub["id"]
                    _cmd = [
                        sys.executable, "train_multi_horizon.py",
                        "--project", project.id,
                        "--substation", _sub_id,
                        "--fast", "--label-prefix", "batch_",
                    ]
                    if _batch_wo:
                        _cmd.append("--weather-only")
                    _all_lines.append(f"\n── [{_i}/{_n}] {_sub_id} ──")
                    _log.code("\n".join(_all_lines[-30:]), language=None)
                    _proc = subprocess.Popen(
                        _cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                    )
                    for _line in _proc.stdout:
                        _line = _line.rstrip()
                        if _line:
                            _all_lines.append(_line)
                            _log.code("\n".join(_all_lines[-30:]), language=None)
                    _proc.wait()
                    if _proc.returncode != 0:
                        _all_ok = False
                        _all_lines.append(f"  ✗ {_sub_id} failed (exit {_proc.returncode})")
                        _log.code("\n".join(_all_lines[-30:]), language=None)
                    else:
                        _all_lines.append(f"  ✓ {_sub_id} done")
                        _log.code("\n".join(_all_lines[-30:]), language=None)

                if _all_ok:
                    _box.update(label=f"All {_n} substations trained ✅", state="complete")
                    st.cache_resource.clear()
                    st.rerun()
                else:
                    _box.update(label="Some substations failed — check output above", state="error")

    if not weather_done:
        st.divider()
        st.markdown("#### Fetch weather data")
        st.caption(
            "Pulls hourly ERA5 reanalysis for the full demand range from "
            f"Open-Meteo at {project.lat:.3f}, {project.lon:.3f} "
            f"(timezone {project.timezone}) and computes the temperature, wind, "
            "calendar and heating-degree features the models need."
        )
        if st.button("Fetch weather data", type="primary", key="fetch_weather_btn"):
            _fetch_weather_for_project(project)

    st.divider()
    st.markdown("#### Demand preview")
    try:
        df = pd.read_csv(project.demand_only_path, index_col="timestamp", parse_dates=True)
        # For multi-mode projects use the aggregate column; fall back to target_column.
        preview_col = project.target_column if project.target_column in df.columns else df.columns[0]
        series = df[preview_col]
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not load the demand preview: {exc}")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Hourly rows", f"{len(series):,}")
    span_years = (series.index.max() - series.index.min()).total_seconds() / 86_400 / 365.25
    c2.metric("Span (years)", f"{span_years:.2f}")
    c3.metric("Peak demand", f"{series.max():.2f} {project.demand_unit or 'MW'}")
    if project.test_start:
        st.caption(
            f"Location: {project.city} · {project.lat:.3f}, {project.lon:.3f} · "
            f"{project.timezone}  ·  test split from {project.test_start}"
        )

    st.plotly_chart(_demand_line(series.dropna(), unit=project.demand_unit or "MW"), use_container_width=True)