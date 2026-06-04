"""
District Heating Demand Forecast Dashboard — Flensburg
Run: streamlit run app.py
"""
from __future__ import annotations

import base64
import json
import pickle
import re
import subprocess
import sys
import time as _time
from datetime import time, timedelta
from pathlib import Path

import lightgbm as lgb
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.data import (
    TARGET,
    DEFAULT_TEST_START,
    MH_FEATURES,
    MH_WEATHER_COLS,
    build_horizon_dataset,
    load_raw,
)
from src.ingest import AGGREGATE_COLUMN
from src.model import (
    BEST_PARAMS,
    simulate_forecast,
    simulate_live_forecast,
    compute_mape,
    TEST_MAPE,
    TEST_MAE,
)
from src.model_registry import ModelRegistry
from src.project import REPO_ROOT, Project
from src.weather import fetch_open_meteo_forecast
from src.forecast_log import log_run, get_proxy_actuals

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heating Demand Forecast",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Password gate ─────────────────────────────────────────────────────────────
# Priority: environment variable APP_PASSWORD → .streamlit/secrets.toml → disabled.
# On Railway: set APP_PASSWORD in the Variables tab.
# On Streamlit Cloud: set it in the Secrets dashboard.
# Locally: add to .streamlit/secrets.toml or set the env var.
import os as _os
_APP_PASSWORD = _os.environ.get("APP_PASSWORD", "")
print(f"[auth] APP_PASSWORD env var present: {bool(_APP_PASSWORD)}", flush=True)
print(f"[auth] All env keys with PASS/AUTH: {[k for k in _os.environ if any(x in k.upper() for x in ['PASS','AUTH','SECRET','PWD'])]}", flush=True)
if not _APP_PASSWORD:
    try:
        _APP_PASSWORD = st.secrets.get("APP_PASSWORD", "")
        print(f"[auth] APP_PASSWORD from secrets: {bool(_APP_PASSWORD)}", flush=True)
    except Exception:
        pass
if _APP_PASSWORD:
    if st.session_state.get("_authenticated") != _APP_PASSWORD:
        _ASSETS_PW = Path(__file__).parent / "assets"
        _logo_pw = _ASSETS_PW / "gradyent-logo.svg"
        if not _logo_pw.exists():
            _logo_pw = _ASSETS_PW / "gradyent-logo-official.svg"
        if _logo_pw.exists():
            _b64_pw = base64.b64encode(_logo_pw.read_bytes()).decode()
            st.markdown(
                f'<div style="text-align:center;margin:2rem 0 1rem;">'
                f'<img src="data:image/svg+xml;base64,{_b64_pw}" width="180"/></div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            "<h3 style='text-align:center;'>Heating Demand Forecast</h3>",
            unsafe_allow_html=True,
        )
        _pw_col = st.columns([1, 2, 1])[1]
        _entered = _pw_col.text_input("Password", type="password", key="_pw_input")
        if _entered and _entered == _APP_PASSWORD:
            st.session_state["_authenticated"] = _APP_PASSWORD
            st.rerun()
        elif _entered:
            _pw_col.error("Incorrect password — please try again.")
        st.stop()

_ASSETS = Path(__file__).parent / "assets"

# Gradyent brand palette (graphic elements style guide)
NAVY      = "#11224D"
PINK      = "#E31B54"
PURPLE    = "#7B2CBF"
TEAL      = "#2EC4B6"
GRAY      = "#6B7280"
GRAY_LITE = "#D1D5DB"
GRAY_MID  = "#9CA3AF"
BG        = "#F8F9FC"
TEXT      = NAVY
TEXT_MUT  = "#4B5563"

PRIMARY   = NAVY
ACCENT    = PINK
ACCENT2   = TEAL
FORECAST  = "#263884"
BAND_COL  = "rgba(38, 56, 132, 0.18)"
NOW_COL   = GRAY_MID
GRID_COL  = "rgba(17, 34, 77, 0.10)"
FORECAST_SHADE = "rgba(17, 34, 77, 0.10)"
OVERRIDE_FILL = "rgba(123, 44, 191, 0.12)"
OVERRIDE_MARK = "rgba(123, 44, 191, 0.45)"


def _inject_brand_styles() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}

        [data-testid="stAppViewContainer"] {{
            background-color: {BG};
        }}

        [data-testid="stMainBlockContainer"] h1,
        [data-testid="stMainBlockContainer"] h2,
        [data-testid="stMainBlockContainer"] h3,
        [data-testid="stMainBlockContainer"] h4,
        [data-testid="stMainBlockContainer"] h5,
        [data-testid="stMainBlockContainer"] h6,
        [data-testid="stMainBlockContainer"] label {{
            color: {TEXT};
        }}

        [data-testid="stSidebar"] {{
            background-color: {BG};
            border-right: 1px solid rgba(17, 34, 77, 0.08);
        }}

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4 {{
            color: {TEXT};
        }}

        .gradyent-header-row {{
            display: flex;
            align-items: center;
            gap: 1.25rem;
            margin-bottom: 0.75rem;
        }}

        .gradyent-header-logo {{
            flex: 0 0 auto;
            display: flex;
            align-items: center;
        }}

        .gradyent-header-logo img {{
            width: 196px;
            max-width: 100%;
            height: auto;
            display: block;
        }}

        .gradyent-header-title {{
            flex: 1 1 auto;
            min-width: 0;
        }}

        .gradyent-page-title {{
            margin: 0;
            font-size: 1.1rem;
            font-weight: 600;
            line-height: 1.25;
            color: {TEXT};
        }}

        [data-testid="stMetricValue"] {{
            color: {TEXT};
            font-weight: 600;
        }}

        [data-testid="stMetricLabel"] {{
            color: {TEXT_MUT};
        }}

        div[data-testid="stExpander"] details summary {{
            color: {TEXT};
            font-weight: 500;
        }}

        .stCaption, small {{
            color: {TEXT_MUT} !important;
        }}

        hr {{
            border-color: rgba(17, 34, 77, 0.10);
        }}

        div[data-testid="stFormSubmitButton"] button,
        div[data-testid="stFormSubmitButton"] button *,
        button[kind="primary"],
        button[kind="primary"] *,
        [data-testid="stBaseButton-primary"],
        [data-testid="stBaseButton-primary"] p,
        [data-testid="stBaseButton-primary"] span,
        [data-testid="stBaseButton-primary"] div {{
            background-color: {PINK} !important;
            border-color: {PINK} !important;
            color: white !important;
        }}

        div[data-testid="stFormSubmitButton"] button:hover,
        button[kind="primary"]:hover,
        [data-testid="stBaseButton-primary"]:hover {{
            background-color: {PURPLE} !important;
            border-color: {PURPLE} !important;
            color: white !important;
        }}

        [data-testid="stSidebar"] div[data-testid="stExpander"] [data-testid="stPlotlyChart"] {{
            max-height: 220px;
            overflow: hidden;
        }}

        /* Disable browser scroll anchoring. When the Plotly demand chart remounts
           on a fragment rerun (substation switch / override add-remove) it briefly
           changes height; the browser's scroll-anchoring then "corrects" the scroll
           and jumps the page to the top. Disabling overflow-anchor keeps the view
           put. (Known Streamlit issue #6953.) */
        * {{
            overflow-anchor: none !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_page_header(title: str) -> None:
    logo_path = _ASSETS / "gradyent-logo.svg"
    if not logo_path.exists():
        logo_path = _ASSETS / "gradyent-logo-official.svg"
    logo_html = ""
    if logo_path.exists():
        b64 = base64.b64encode(logo_path.read_bytes()).decode()
        logo_html = (
            f'<div class="gradyent-header-logo">'
            f'<img src="data:image/svg+xml;base64,{b64}" alt="Gradyent" />'
            f"</div>"
        )
    st.markdown(
        f'<div class="gradyent-header-row">'
        f"{logo_html}"
        f'<div class="gradyent-header-title">'
        f'<div class="gradyent-page-title">{title}</div>'
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _format_year_range(start_year: int, end_year: int) -> str:
    """``2020–2023`` or just ``2020`` when the range collapses to one year."""
    return f"{start_year}" if start_year == end_year else f"{start_year}–{end_year}"


def _render_sidebar_footer(
    *,
    snapshot_dt: pd.Timestamp,
    horizon: int,
    model_label: str,
    wx_label: str,
    city: str,
    data_start: pd.Timestamp,
    data_end: pd.Timestamp,
    test_start_ts: pd.Timestamp,
) -> None:
    # Footer year ranges follow the project's train/test split:
    #   train  = data_start.year … (test_start.year − 1)
    #   test   = test_start.year onward
    #   data   = data_start.year … data_end.year
    train_range = _format_year_range(data_start.year, test_start_ts.year - 1)
    data_range = _format_year_range(data_start.year, data_end.year)
    test_year = test_start_ts.year
    # Rendered inline in the main area (NOT the sidebar): this runs inside the
    # workspace fragment, and Streamlit forbids writing to st.sidebar from a
    # fragment. Keeping it inline also means it stays in sync on fragment reruns.
    with st.expander("Run details", expanded=False):
        st.caption(f"Snapshot · {snapshot_dt.strftime('%A, %d %b %Y · %H:%M')}")
        st.caption(f"Horizon · {horizon}h")
        st.caption(f"Model · {model_label}")
        st.caption(f"Training · {train_range}")
        st.caption(f"Weather · {wx_label}")
        st.caption(f"{city} · LightGBM · 25 features · {data_range}")
        st.caption("One model per horizon h=1…48 · direct strategy")
        st.caption(f"Train: {train_range} · Test: {test_year} · NWP noise augmented")


_inject_brand_styles()

# ── Active project ────────────────────────────────────────────────────────────
# The project is the top-level selector: it scopes the dataset, models, registry
# and coordinates.  Resolved before any data/model load so init() is keyed by it.
_PROJECT_IDS = Project.list_all() or [Project.default_id()]
_PROJECT_NAMES = {pid: Project.load(pid).name for pid in _PROJECT_IDS}

# Apply a deferred project switch (e.g. after deleting the active project) before
# the project selectbox widget is instantiated — setting a widget-backed
# session_state key after the widget exists would raise a StreamlitAPIException.
if "pending_project_switch" in st.session_state:
    _pp = st.session_state.pop("pending_project_switch")
    if _pp in _PROJECT_IDS:
        st.session_state["active_project_id"] = _pp

if st.session_state.get("active_project_id") not in _PROJECT_IDS:
    st.session_state["active_project_id"] = (
        Project.default_id() if Project.default_id() in _PROJECT_IDS else _PROJECT_IDS[0]
    )
active_project_id = st.session_state["active_project_id"]
project = Project.load(active_project_id)

# When the project changes, drop the previous project's version/pkl selection so
# the version selector re-initialises against the new project's registry.
if st.session_state.get("_loaded_project_id") != active_project_id:
    for _k in ("active_pkl_dir", "active_version_id", "pending_version_switch"):
        st.session_state.pop(_k, None)
    st.session_state["_loaded_project_id"] = active_project_id


def _make_registry() -> ModelRegistry:
    """ModelRegistry scoped to the active project."""
    return ModelRegistry(
        registry_path=project.registry_path,
        default_meta_path=project.default_meta_path,
        default_pkl_dir=project.rel_default_pkl_dir,
    )


# ── Bootstrap registry (silent no-op if already exists) ───────────────────────
_make_registry().auto_register_existing_default()

# ── Version-aware, project-aware init ─────────────────────────────────────────
@st.cache_resource(show_spinner="Loading data and models…")
def init(project_id: str, pkl_dir: str):
    """
    Load a project's raw CSV + all 48 PKLs from *pkl_dir* (repo-relative).

    Streamlit caches separately for each distinct (project_id, pkl_dir), so
    switching projects or model versions gets its own cache slot without
    evicting the others.
    """
    proj = Project.load(project_id)
    df_raw = load_raw(proj.data_path, target=proj.target_column)
    pkl_path = REPO_ROOT / pkl_dir

    mh_models: dict = {}
    for h in range(1, 49):
        p = pkl_path / f"lgbm_h{h:02d}.pkl"
        if p.exists():
            try:
                with open(p, "rb") as fh:
                    mh_models[h] = pickle.load(fh)
            except Exception:
                try:
                    p.unlink()
                except OSError:
                    pass

    mh_eval = None
    mh_features = list(MH_FEATURES)
    meta_path = pkl_path / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        if "eval" in meta:
            mh_eval = pd.DataFrame(meta["eval"])
        if "features" in meta:
            mh_features = meta["features"]

    return df_raw, mh_models, mh_eval, mh_features


def _model_is_weather_only(features: list[str]) -> bool:
    """True when *features* contains no demand-lag columns (weather-only model)."""
    from src.data import MH_LAG_COLS
    return not any(c in features for c in MH_LAG_COLS)


def _is_nan_like(v) -> bool:
    """True if v is float NaN or None."""
    import math
    try:
        return v is None or math.isnan(float(v))
    except (TypeError, ValueError):
        return True


@st.cache_data(show_spinner="Analysing feature importance…")
def probe_feature_importance(
    project_id: str, start_date: str, end_date: str, sub_column: str | None = None
) -> dict[str, float]:
    """Quick probe model (h=24) to rank features for a training date range.

    Pass *sub_column* when probing for a specific substation (multi-mode).
    """
    proj = Project.load(project_id)
    target = sub_column if sub_column else proj.target_column
    df_raw = load_raw(proj.data_path, target=target)
    df_train = df_raw.loc[pd.Timestamp(start_date):pd.Timestamp(end_date)]
    df_h = build_horizon_dataset(df_train, 24, noise_augment=False, target=target)
    train = df_h[df_h.index <= pd.Timestamp(end_date)]
    probe_params = {**BEST_PARAMS, "n_estimators": 300}
    mdl = lgb.LGBMRegressor(**probe_params, n_jobs=-1, random_state=42, verbose=-1)
    mdl.fit(train[MH_FEATURES], train["_target"])
    imp = pd.Series(mdl.feature_importances_, index=MH_FEATURES)
    return imp.sort_values(ascending=False).to_dict()


def auto_select_features(importance: dict[str, float]) -> set[str]:
    """Auto-select features covering ~85% of total importance (min. 8)."""
    series = pd.Series(importance).sort_values(ascending=False)
    total = float(series.sum())
    if total <= 0:
        return set(MH_FEATURES)
    selected: set[str] = set()
    cum = 0.0
    for feat, val in series.items():
        selected.add(feat)
        cum += val
        if cum / total >= 0.85:
            break
    for feat in series.index[:8]:
        selected.add(feat)
    return selected


def _feat_label(name: str) -> str:
    return name.removeprefix("fc_").replace("_", " ")


def _importance_rows(importance: dict[str, float]) -> list[tuple[str, float]]:
    """Ascending by score — highest ends up at the top of a horizontal bar chart."""
    return sorted(importance.items(), key=lambda kv: kv[1])


def _build_feature_chart(
    rows: list[tuple[str, float]],
    selected: set[str],
    xaxis_title: str = "Importance",
    disabled: bool = False,
    max_height: int | None = None,
) -> go.Figure:
    plain_labels = [_feat_label(f) for f, _ in rows]
    if disabled:
        ticktext = plain_labels
        colors = [GRAY_LITE] * len(rows)
    else:
        ticktext = [
            f"<b>{lbl}</b>" if rows[i][0] in selected else lbl
            for i, lbl in enumerate(plain_labels)
        ]
        colors = [ACCENT if f in selected else GRAY_LITE for f, _ in rows]
    if max_height is not None:
        chart_h = max_height
        tick_size = max(10, min(12, max_height // len(rows))) if rows else 11
    else:
        chart_h = max(280, len(rows) * 22)
        tick_size = 12
    fig = go.Figure(go.Bar(
        x=[v for _, v in rows],
        y=plain_labels,
        orientation="h",
        customdata=[f for f, _ in rows],
        marker_color=colors,
        hovertemplate="%{customdata}<br>%{x:.0f}<extra></extra>",
    ))
    fig.update_layout(
        height=chart_h,
        margin=dict(l=4, r=4, t=4, b=4),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title=xaxis_title,
        showlegend=False,
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=plain_labels,
        ticktext=ticktext,
        tickfont=dict(size=tick_size, color=GRAY_MID if disabled else TEXT_MUT),
    )
    return fig


def _format_date_range(start: str, end: str) -> str:
    return f"{start} → {end}"


def _format_version_label(
    start: str, end: str, label: str, mape_avg: float | None = None
) -> str:
    label_part = f"  {label}" if label else ""
    name = f"{_format_date_range(start, end)}{label_part}"
    if mape_avg is not None:
        name += f"  (MAPE {mape_avg:.1f}%)"
    return name


def _read_mape_avg(output_dir: str) -> float | None:
    meta_path = Path(output_dir) / "meta.json"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    eval_rows = meta.get("eval", [])
    if not eval_rows:
        return None
    return round(float(pd.DataFrame(eval_rows)["mape_pct"].mean()), 1)


def _chart_time_axis(span_h: float) -> tuple[int, int, int, float]:
    """Return (dtick_ms, tick_angle, top_margin, day_label_y) for the demand chart."""
    if span_h <= 40:
        dtick_ms = 6 * 3_600_000
    elif span_h <= 72:
        dtick_ms = 6 * 3_600_000
    else:
        dtick_ms = 12 * 3_600_000
    n_ticks = span_h / (dtick_ms / 3_600_000)
    # Day labels sit in the top margin above hour ticks (side=top on row 1).
    if n_ticks <= 10:
        return dtick_ms, 0, 92, 1.16
    return dtick_ms, -45, 152, 1.24


OVERRIDES_KEY = "forecast_overrides"
DEFAULT_TOTAL_DWELLINGS = 10_000


def _mape_period_status(window_mape: float, reference_mape: float) -> tuple[str, str]:
    """Label and colour for period MAPE vs the active model's average."""
    ref = reference_mape if reference_mape > 0 else TEST_MAPE
    ratio = window_mape / ref
    if ratio <= 1.10:
        return "On target", "#16a34a"
    if ratio <= 1.25:
        return "Acceptable", "#ca8a04"
    return "High error", "#dc2626"


def apply_forecast_overrides(
    forecast: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
    overrides: list[dict],
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Scale forecast and confidence band for each active override window."""
    fc = forecast.copy().astype(float)
    lo = lower.copy().astype(float)
    hi = upper.copy().astype(float)
    for ov in overrides:
        start = pd.Timestamp(ov["from_dt"])
        end = pd.Timestamp(ov["to_dt"])
        if start > end:
            continue
        mask = (fc.index >= start) & (fc.index <= end)
        factor = 1.0 + float(ov["pct_change"]) / 100.0
        fc.loc[mask] *= factor
        lo.loc[mask] *= factor
        hi.loc[mask] *= factor
    return fc, lo, hi


def _init_override_inputs(pfx: str = "") -> None:
    if f"{pfx}ov_total" not in st.session_state:
        st.session_state[f"{pfx}ov_total"] = DEFAULT_TOTAL_DWELLINGS
    if f"{pfx}ov_pct" not in st.session_state:
        st.session_state[f"{pfx}ov_pct"] = 0
    if f"{pfx}ov_dwell" not in st.session_state:
        st.session_state[f"{pfx}ov_dwell"] = 0


def _sync_pct_to_dwell(pfx: str = "") -> None:
    total = st.session_state[f"{pfx}ov_total"]
    st.session_state[f"{pfx}ov_dwell"] = int(round(st.session_state[f"{pfx}ov_pct"] / 100.0 * total))


def _sync_dwell_to_pct(pfx: str = "") -> None:
    total = st.session_state[f"{pfx}ov_total"]
    if total > 0:
        st.session_state[f"{pfx}ov_pct"] = int(round(st.session_state[f"{pfx}ov_dwell"] / total * 100.0))
    else:
        st.session_state[f"{pfx}ov_pct"] = 0


def _sync_total_to_dwell(pfx: str = "") -> None:
    _sync_pct_to_dwell(pfx)


# Override add/remove are handled via on_click callbacks (not inline + st.rerun).
# A callback mutates session_state BEFORE the script body re-runs, so the single
# widget-triggered rerun already reflects the change — no explicit st.rerun is
# needed. The explicit st.rerun caused a SECOND render that re-created (remounted)
# the Plotly chart and scrolled the page to the top.
def _add_override_cb(pfx: str, overrides_key: str) -> None:
    from_date = st.session_state[f"{pfx}ov_from_date"]
    to_date   = st.session_state[f"{pfx}ov_to_date"]
    from_time = st.session_state[f"{pfx}ov_from_time"]
    to_time   = st.session_state[f"{pfx}ov_to_time"]
    o_from = pd.Timestamp.combine(from_date, time(int(str(from_time)[:2]), 0))
    o_to   = pd.Timestamp.combine(to_date,   time(int(str(to_time)[:2]), 0))
    if o_to <= o_from:
        st.session_state[f"{pfx}ov_warn"] = "End time must be after start time."
        return
    if int(st.session_state[f"{pfx}ov_pct"]) == 0:
        st.session_state[f"{pfx}ov_warn"] = "Set a non-zero demand change or dwelling equivalents."
        return
    st.session_state.setdefault(overrides_key, []).append({
        "from_dt": o_from.isoformat(),
        "to_dt": o_to.isoformat(),
        "pct_change": int(st.session_state[f"{pfx}ov_pct"]),
        "dwellings_equiv": int(st.session_state[f"{pfx}ov_dwell"]),
        "total_dwellings": int(st.session_state[f"{pfx}ov_total"]),
    })
    st.session_state[f"{pfx}ov_warn"] = None
    st.session_state.pop(f"{pfx}ov_confirm_remove", None)


def _request_remove_override_cb(pfx: str, idx: int) -> None:
    st.session_state[f"{pfx}ov_confirm_remove"] = idx


def _confirm_remove_override_cb(pfx: str, overrides_key: str, idx: int) -> None:
    try:
        st.session_state[overrides_key].pop(idx)
    except (IndexError, KeyError):
        pass
    st.session_state.pop(f"{pfx}ov_confirm_remove", None)


def _cancel_remove_override_cb(pfx: str) -> None:
    st.session_state.pop(f"{pfx}ov_confirm_remove", None)


def _pick_date(default, min_dt, max_dt, key: str):
    return st.date_input(
        "Date",
        value=default.date(),
        min_value=min_dt.date(),
        max_value=max_dt.date(),
        key=key,
        label_visibility="collapsed",
    )


def _pick_hour(default_hour: int, key: str) -> time:
    hour_options = [f"{h:02d}:00" for h in range(24)]
    picked = st.selectbox(
        "Time",
        options=hour_options,
        index=default_hour,
        key=key,
        label_visibility="collapsed",
    )
    return time(int(picked[:2]), 0)


def _render_manual_override(
    forecast_base: pd.Series,
    data_end: pd.Timestamp,
    key_prefix: str = "",
    overrides_key: str = OVERRIDES_KEY,
    rerun_scope: str = "app",
) -> None:
    """Override form + periods table. key_prefix scopes all widget keys so the
    widget can be used independently on the aggregate tab and each substation tab.

    rerun_scope controls the scope of the reruns triggered by Add/Remove actions.
    The aggregate tab uses "app" (its chart renders at module level and needs a
    full rerun). The substation tab calls this function inline inside its own
    fragment and passes "fragment", so override changes redraw only the substation
    chart without recomputing the aggregate."""
    pfx = key_prefix
    n_active = len(st.session_state.get(overrides_key, []))
    expander_title = (
        "Manual forecast override · active" if n_active else "Manual forecast override"
    )
    with st.expander(expander_title, expanded=False):
        st.caption(
            "Adjust the displayed forecast for scenarios such as new connections or "
            "outages. Overrides apply wherever they overlap the forecast; future periods "
            "are saved and take effect when the chart window reaches them."
        )
        _init_override_inputs(pfx)

        fc_min = forecast_base.index[0].to_pydatetime()
        fc_max = forecast_base.index[-1].to_pydatetime()
        pick_max = data_end.to_pydatetime()

        c_pct, c_dwell, c_total = st.columns(3)
        c_pct.number_input(
            "Demand change %",
            step=1,
            format="%d",
            key=f"{pfx}ov_pct",
            on_change=_sync_pct_to_dwell,
            args=(pfx,),
            help="Positive = increase, negative = decrease.",
        )
        c_dwell.number_input(
            "Dwelling equivalents",
            step=1,
            format="%d",
            key=f"{pfx}ov_dwell",
            on_change=_sync_dwell_to_pct,
            args=(pfx,),
            help="Equivalent number of dwellings added (+) or removed (−).",
        )
        c_total.number_input(
            "Total dwellings",
            min_value=1,
            step=100,
            key=f"{pfx}ov_total",
            on_change=_sync_total_to_dwell,
            args=(pfx,),
            help="Network total — used to convert between % and dwelling equivalents.",
        )

        st.markdown("**Override period**")
        h_fd, h_ft, h_td, h_tt = st.columns([2, 1, 2, 1])
        h_fd.caption("From · date")
        h_ft.caption("From · time")
        h_td.caption("To · date")
        h_tt.caption("To · time")
        c_fd, c_ft, c_td, c_tt = st.columns([2, 1, 2, 1])
        with c_fd:
            from_date = _pick_date(fc_min, fc_min, pick_max, f"{pfx}ov_from_date")
        with c_ft:
            from_time = _pick_hour(fc_min.hour, f"{pfx}ov_from_time")
        with c_td:
            to_date = _pick_date(fc_max, fc_min, pick_max, f"{pfx}ov_to_date")
        with c_tt:
            to_time = _pick_hour(fc_max.hour, f"{pfx}ov_to_time")
        o_from_ts = pd.Timestamp.combine(from_date, from_time)
        o_to_ts = pd.Timestamp.combine(to_date, to_time)

        st.button(
            "Add override", key=f"{pfx}ov_add_btn",
            on_click=_add_override_cb, args=(pfx, overrides_key),
        )
        if st.session_state.get(f"{pfx}ov_warn"):
            st.warning(st.session_state[f"{pfx}ov_warn"])

        overrides = st.session_state[overrides_key]
        if overrides:
            st.divider()
            st.markdown("**Override periods**")
            confirm_idx = st.session_state.get(f"{pfx}ov_confirm_remove")
            for i, ov in enumerate(overrides):
                pending = confirm_idx == i
                summary = (
                    f"{pd.Timestamp(ov['from_dt']).strftime('%d %b %H:%M')} → "
                    f"{pd.Timestamp(ov['to_dt']).strftime('%d %b %H:%M')} · "
                    f"{int(ov['pct_change']):+d}% · "
                    f"{int(round(ov['dwellings_equiv'])):+d} dwellings"
                )
                if pending:
                    q_col, yes_col, no_col = st.columns([4, 1, 1])
                    q_col.markdown(f"Are you sure you want to remove **{summary}**?")
                    yes_col.button(
                        "Yes", key=f"{pfx}ov_confirm_yes", type="primary",
                        on_click=_confirm_remove_override_cb, args=(pfx, overrides_key, i),
                    )
                    no_col.button(
                        "No", key=f"{pfx}ov_confirm_no",
                        on_click=_cancel_remove_override_cb, args=(pfx,),
                    )
                else:
                    btn_col, txt_col = st.columns([1, 5])
                    btn_col.button(
                        "🗑️", key=f"{pfx}rm_ov_{i}", help="Remove this override",
                        on_click=_request_remove_override_cb, args=(pfx, i),
                    )
                    txt_col.markdown(
                        f"**{pd.Timestamp(ov['from_dt']).strftime('%d %b %H:%M')}** → "
                        f"**{pd.Timestamp(ov['to_dt']).strftime('%d %b %H:%M')}** · "
                        f"{int(ov['pct_change']):+d}% · "
                        f"{int(round(ov['dwellings_equiv'])):+d} dwellings"
                    )


if hasattr(st, "fragment"):
    _manual_override = st.fragment(_render_manual_override)
else:
    _manual_override = _render_manual_override


def _toggle_feature_from_chart(event, feat_order: list[str]) -> None:
    """Toggle one feature when the user clicks a bar in the probe chart."""
    if event is None:
        return
    sel = getattr(event, "selection", None)
    if sel is None:
        return
    pts = sel.get("points", []) if isinstance(sel, dict) else getattr(sel, "points", [])
    if not pts:
        return
    pt = pts[0]
    idx = pt.get("point_index", pt.get("pointIndex"))
    if idx is None or idx >= len(feat_order):
        return
    feat = feat_order[int(idx)]
    pending = st.session_state.get("retrain_pending")
    if not pending:
        return
    selected = set(pending["selected"])
    if feat in selected:
        selected.remove(feat)
    else:
        selected.add(feat)
    pending["selected"] = sorted(selected)
    st.session_state["retrain_pending"] = pending
    st.session_state["feat_chart_nonce"] = st.session_state.get("feat_chart_nonce", 0) + 1
    st.rerun()


def _norm_pkl_dir(d: str) -> str:
    return str(d).rstrip("/")


def _render_version_and_retrain_sidebar(
    df_raw: pd.DataFrame,
    mh_eval: "pd.DataFrame | None" = None,
    mh_features: "list[str] | None" = None,
) -> None:
    """Render the **Model version** selector + **Model specifications** + **🔁 Train model** expander.

    Shared by the ready-project dashboard sidebar and the ``awaiting_training``
    setup view so the retrain flow (feature probe, project-aware date bounds,
    label, subprocess streaming with progress) is byte-for-byte identical in
    both. Driven entirely by *df_raw* (the project's demand+weather data) and
    the project registry, so it works for a not-yet-trained project that has no
    PKLs: the version selector shows "no versions yet" and the retrain expander
    lets the user create the first model. Assumes it is called inside a
    ``with st.sidebar:`` block.
    """
    # ── Model version selector ────────────────────────────────────────────────
    st.markdown("**Model version**")
    registry = _make_registry()
    versions = registry.list_versions()

    if not versions:
        st.info("No model versions registered yet — use **Train model** in the sidebar to train your first model.")
    else:
        def _fmt_version(v: dict) -> str:
            label_part = f"  {v['label']}" if v.get("label") else ""
            return (
                f"{_format_date_range(v['date_range_start'], v['date_range_end'])}"
                f"{label_part}  (MAPE {v['mape_avg']:.1f}%)"
            )

        id_to_version = {v["id"]: v for v in versions}
        dir_to_id = {_norm_pkl_dir(v["pkl_dir"]): v["id"] for v in versions}
        version_ids = [v["id"] for v in versions]

        current_dir = st.session_state.get("active_pkl_dir", project.rel_default_pkl_dir)
        if "active_version_id" not in st.session_state:
            st.session_state["active_version_id"] = dir_to_id.get(
                _norm_pkl_dir(current_dir), version_ids[0]
            )
        if st.session_state["active_version_id"] not in id_to_version:
            st.session_state["active_version_id"] = version_ids[0]

        st.selectbox(
            "Select version",
            options=version_ids,
            format_func=lambda vid: _fmt_version(id_to_version[vid]),
            key="active_version_id",
            label_visibility="collapsed",
        )
        chosen_dir = id_to_version[st.session_state["active_version_id"]]["pkl_dir"]
        if _norm_pkl_dir(chosen_dir) != _norm_pkl_dir(current_dir):
            st.session_state["active_pkl_dir"] = chosen_dir
            st.cache_resource.clear()
            st.rerun()

        selected_id = st.session_state["active_version_id"]
        selected_summary = _fmt_version(id_to_version[selected_id])

        # ── Model specifications ──────────────────────────────────────────────
        _sel_v = id_to_version[selected_id]
        _spec_period = (
            f"{_sel_v.get('date_range_start','?')} → {_sel_v.get('date_range_end','?')}"
        )
        _spec_mape = (
            f"{round(float(mh_eval['mape_pct'].mean()), 1)}%"
            if mh_eval is not None else "—"
        )
        _spec_mae = (
            f"{round(float(mh_eval['mae_mw'].mean()) * _du_factor, 3)} {_display_unit}"
            if mh_eval is not None else "—"
        )
        _spec_nfeat = len(mh_features) if mh_features else "—"
        _spec_wo = (
            _model_is_weather_only(mh_features) if mh_features else False
        )
        _spec_type = "🌤️ Weather-only" if _spec_wo else "📊 Demand + weather"
        with st.expander("Model specifications", expanded=True):
            _spec_rows = [
                ("Period", _spec_period),
                ("Avg MAPE", _spec_mape),
                ("Avg MAE", _spec_mae),
                ("Features", str(_spec_nfeat)),
                ("Type", _spec_type),
            ]
            for _sk, _sv in _spec_rows:
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'font-size:0.82em;line-height:1.7;">'
                    f'<span style="color:{GRAY_MID};font-weight:500;">{_sk}</span>'
                    f'<span style="color:{TEXT};font-weight:600;">{_sv}</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

        if st.session_state.get("version_confirm_remove"):
            st.markdown(f"Remove **{selected_summary}**?")
            c_yes, c_no = st.columns(2)
            if c_yes.button("Yes", key="version_rm_yes", type="primary"):
                registry.remove_version(selected_id, delete_files=True)
                st.session_state.pop("version_confirm_remove", None)
                remaining = registry.list_versions()
                if remaining:
                    st.session_state["pending_version_switch"] = {
                        "id": remaining[0]["id"],
                        "pkl_dir": remaining[0]["pkl_dir"],
                    }
                else:
                    st.session_state["pending_version_switch"] = {
                        "id": None,
                        "pkl_dir": project.rel_default_pkl_dir,
                    }
                st.cache_resource.clear()
                st.rerun()
            if c_no.button("No", key="version_rm_no"):
                st.session_state.pop("version_confirm_remove", None)
                st.rerun()
        elif st.button(
            "Remove selected model",
            key="version_rm_btn",
            disabled=len(version_ids) <= 1,
        ):
            st.session_state["version_confirm_remove"] = True
            st.rerun()

    success_name = st.session_state.pop("retrain_success_name", None)
    if success_name:
        st.success(
            f"New model **{success_name}** ready — select it in the "
            "**Model version** dropdown above."
        )

    # ── Retrain expander ──────────────────────────────────────────────────────
    # Retrain date-picker bounds are derived from the ACTIVE project's available
    # data range (df_raw is already loaded above), so a user can retrain whatever
    # window the project actually covers:
    #   start: min = data_start, max = data_end, default = data_start
    #   end:   min = start + 90d, max = data_end,
    #          default = test_start − 1 day  (reproduces the train/test split),
    #                    falling back to data_end when test_start is missing.
    # For flensburg this defaults to 2020-01-01 → 2023-12-31 (its original range);
    # for aalborg it allows 2018-01-03 → 2019-12-31.
    _rt_data_start = df_raw.index.min()
    _rt_data_end = df_raw.index.max()
    _rt_default_start = _rt_data_start
    _rt_end_min = _rt_default_start + pd.Timedelta(days=90)
    if project.test_start:
        _rt_default_end = pd.Timestamp(project.test_start) - pd.Timedelta(days=1)
    else:
        _rt_default_end = _rt_data_end
    _rt_default_end = min(max(_rt_default_end, _rt_end_min), _rt_data_end)
    with st.expander("🔁 Train model"):
        with st.form("retrain_dates_form", clear_on_submit=False):
            rt_col1, rt_col2 = st.columns(2)
            with rt_col1:
                rt_start = st.date_input(
                    "Start date",
                    value=_rt_default_start.date(),
                    min_value=_rt_data_start.date(),
                    max_value=_rt_data_end.date(),
                )
            with rt_col2:
                rt_end = st.date_input(
                    "End date",
                    value=_rt_default_end.date(),
                    min_value=rt_start + timedelta(days=90),
                    max_value=_rt_data_end.date(),
                )
            rt_label = st.text_input(
                "Label (optional)",
                placeholder="e.g. summer-only",
            )
            rt_model_type = st.radio(
                "Model type",
                options=["Standard (demand lags + weather)", "Weather-only (no lags)"],
                horizontal=True,
                help=(
                    "**Standard**: trains with all 31 features including demand lags. "
                    "Best accuracy when SCADA data is available.\n\n"
                    "**Weather-only**: trains with weather + calendar features only (no demand lags). "
                    "Used by the Live Forecaster tab. "
                    "Also available in the version selector for MAPE comparison."
                ),
            )
            rt_speed = st.radio(
                "Training speed",
                options=["Full (500 trees — best accuracy)", "Fast (100 trees — ~5× quicker)"],
                horizontal=True,
                help=(
                    "**Full**: 500 estimators — recommended for final/production models. "
                    "Takes ~3–5 min locally, ~15–30 min on Streamlit Cloud.\n\n"
                    "**Fast**: 100 estimators — ~5× quicker, ~+0.5–1% higher MAPE. "
                    "Good for testing or training on slow cloud VMs."
                ),
            )
            analyse_btn = st.form_submit_button("Continue →", use_container_width=True)

        if analyse_btn:
            if rt_end <= rt_start + timedelta(days=90):
                st.warning("End date must be more than 90 days after start date.")
            else:
                _is_wo   = rt_model_type.startswith("Weather-only")
                _is_fast = rt_speed.startswith("Fast")
                if _is_wo:
                    st.session_state["retrain_pending"] = {
                        "start": str(rt_start),
                        "end": str(rt_end),
                        "label": rt_label.strip(),
                        "model_type": "weather-only",
                        "fast": _is_fast,
                        "importance": {},
                        "selected": [],
                    }
                else:
                    importance = probe_feature_importance(
                        active_project_id, str(rt_start), str(rt_end)
                    )
                    auto_selected = sorted(auto_select_features(importance))
                    st.session_state["retrain_pending"] = {
                        "start": str(rt_start),
                        "end": str(rt_end),
                        "label": rt_label.strip(),
                        "model_type": "standard",
                        "fast": _is_fast,
                        "importance": importance,
                        "selected": auto_selected,
                    }
                    st.session_state["feat_chart_nonce"] = 0
                st.rerun()

        pending = st.session_state.get("retrain_pending")
        training_job = st.session_state.get("retrain_job")

        if pending:
            _is_wo = pending.get("model_type") == "weather-only"

            if not _is_wo:
                with st.expander(
                    f"Feature relevance · {pending['start']} → {pending['end']}",
                    expanded=False,
                ):
                    is_locked = bool(training_job)
                    if is_locked:
                        st.caption("Training in progress — feature selection is locked.")
                    else:
                        st.caption(
                            "Click a bar to include or exclude. Selected = accent + **bold**."
                        )

                    imp_rows = _importance_rows(pending["importance"])
                    feat_order = [f for f, _ in imp_rows]
                    selected_set = set(pending["selected"])

                    fig_probe = _build_feature_chart(
                        imp_rows,
                        selected_set,
                        xaxis_title="Importance (probe h=24)",
                        disabled=is_locked,
                        max_height=360,
                    )
                    chart_key = f"feat_probe_{st.session_state.get('feat_chart_nonce', 0)}"
                    if is_locked:
                        st.plotly_chart(fig_probe, use_container_width=True, key=chart_key)
                    else:
                        chart_event = st.plotly_chart(
                            fig_probe,
                            use_container_width=True,
                            on_select="rerun",
                            selection_mode="points",
                            key=chart_key,
                        )
                        _toggle_feature_from_chart(chart_event, feat_order)

                    selected_feats = list(pending["selected"])
                    st.caption(
                        f"**{len(selected_feats)}** of {len(feat_order)} features selected"
                    )
            else:
                st.info(
                    "Weather-only model: trains with weather + calendar features only "
                    "(no demand lags). Saves to `models/live/` and appears in the version "
                    "selector with a **WO ·** label for MAPE comparison.",
                    icon="🌤️",
                )

            if not training_job:
                train_btn = st.button(
                    "Start training",
                    type="primary",
                    use_container_width=True,
                )
                if train_btn:
                    if _is_wo:
                        wo_label = f"WO · {pending['label']}" if pending["label"] else "WO"
                        output_dir = f"projects/{active_project_id}/models/live/"
                        cmd = [
                            sys.executable,
                            "train_multi_horizon.py",
                            "--project",    active_project_id,
                            "--start-date", pending["start"],
                            "--end-date",   pending["end"],
                            "--weather-only",
                            "--output-dir", str(REPO_ROOT / output_dir),
                            "--label",      wo_label,
                        ]
                        if pending.get("fast"):
                            cmd.append("--fast")
                        st.session_state["retrain_job"] = {
                            "cmd": cmd,
                            "output_dir": output_dir,
                            "start": pending["start"],
                            "end": pending["end"],
                            "label": wo_label,
                            "model_type": "weather-only",
                        }
                        st.rerun()
                    else:
                        selected_feats = list(pending["selected"])
                        if len(selected_feats) < 3:
                            st.warning("Select at least 3 features.")
                        else:
                            ver_label = pending["label"] or "v1"
                            output_dir = (
                                f"projects/{active_project_id}/models/versions/"
                                f"{pending['start']}_{pending['end']}_{ver_label}/"
                            )
                            cmd = [
                                sys.executable,
                                "train_multi_horizon.py",
                                "--project",    active_project_id,
                                "--start-date", pending["start"],
                                "--end-date",   pending["end"],
                                "--label",      pending["label"],
                                "--output-dir", output_dir,
                                "--features",   ",".join(selected_feats),
                            ]
                            if pending.get("fast"):
                                cmd.append("--fast")
                            st.session_state["retrain_job"] = {
                                "cmd": cmd,
                                "output_dir": output_dir,
                                "start": pending["start"],
                                "end": pending["end"],
                                "label": pending["label"] or "v1",
                                "model_type": "standard",
                            }
                            st.rerun()

        if training_job:
            lines: list[str] = []
            progress = st.progress(0, text="Starting training…")
            proc = subprocess.Popen(
                training_job["cmd"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:  # type: ignore[union-attr]
                lines.append(line)
                m = re.search(r"h=\s*(\d+)/(\d+)", line)
                if m:
                    h, total = int(m.group(1)), int(m.group(2))
                    progress.progress(
                        h / total,
                        text=f"Training model {h} of {total}…",
                    )
                elif "Evaluating on held-out" in line:
                    progress.progress(1.0, text="Evaluating on held-out test set…")
                elif "Saving models" in line:
                    progress.progress(1.0, text="Saving models…")
            proc.wait()

            st.session_state.pop("retrain_job", None)
            if proc.returncode == 0:
                progress.progress(1.0, text="Done")
                with st.expander("Training log", expanded=False):
                    st.code("".join(lines))
                mape_avg = _read_mape_avg(training_job["output_dir"])
                version_label = training_job["label"]
                st.session_state["retrain_success_name"] = _format_version_label(
                    training_job["start"],
                    training_job["end"],
                    version_label,
                    mape_avg,
                )
                del st.session_state["retrain_pending"]
                st.cache_resource.clear()
                st.rerun()
            else:
                progress.empty()
                with st.expander("Training log (error)", expanded=True):
                    st.code("".join(lines))
                del st.session_state["retrain_pending"]


# ── Sidebar: project selector + "New project" (always rendered first) ─────────
# The project selector lives at the top of the sidebar for every mode. The
# branches below decide what fills the main area: the builder wizard, the
# "setup incomplete" panel, or the full dashboard (ready projects only).
with st.sidebar:
    st.markdown("**City / location**")
    st.selectbox(
        "City / location",
        options=_PROJECT_IDS,
        format_func=lambda pid: _PROJECT_NAMES.get(pid, pid),
        key="active_project_id",
        label_visibility="collapsed",
    )
    if st.session_state["active_project_id"] != active_project_id:
        st.rerun()
    if st.button("＋ New forecaster…", use_container_width=True, key="new_project_btn"):
        st.session_state["builder_mode"] = True
        st.rerun()

    # ── Delete project (two-step confirm; mirrors the version-removal UX) ──────
    # Visually separated from the *Model version* "Remove" button below: this
    # destroys the whole project (config + data + trained models) on disk.
    # Guard rails: never delete the last remaining project, and never delete
    # while the setup wizard is open or a retrain job is running.
    _is_last_project = len(_PROJECT_IDS) <= 1
    _delete_blocked = bool(st.session_state.get("builder_mode")) or bool(
        st.session_state.get("retrain_job")
    )
    _active_name = _PROJECT_NAMES.get(active_project_id, active_project_id)

    if st.session_state.get("project_confirm_delete"):
        st.warning(
            f"Permanently delete **{_active_name}**? This erases the project's "
            "data **and** trained models from disk and cannot be undone."
        )
        _d_yes, _d_no = st.columns(2)
        if _d_yes.button(
            "Delete",
            key="project_del_yes",
            type="primary",
            use_container_width=True,
            disabled=_is_last_project or _delete_blocked,
        ):
            _deleted_id = active_project_id
            if Project.delete(_deleted_id):
                _remaining = [p for p in Project.list_all() if p != _deleted_id]
                st.session_state["pending_project_switch"] = (
                    _remaining[0] if _remaining else Project.default_id()
                )
                st.session_state.pop("project_confirm_delete", None)
                st.cache_resource.clear()
                st.rerun()
            else:
                st.session_state.pop("project_confirm_delete", None)
                st.error("Could not delete this project.")
        if _d_no.button("Cancel", key="project_del_no", use_container_width=True):
            st.session_state.pop("project_confirm_delete", None)
            st.rerun()
    elif _is_last_project:
        st.caption("🗑️ Delete project unavailable — at least one project must remain.")
    else:
        if st.button(
            "🗑️ Delete project",
            key="project_del_btn",
            use_container_width=True,
            disabled=_delete_blocked,
            help=(
                "Finish the setup wizard or wait for retraining to complete first."
                if _delete_blocked
                else "Permanently remove this project's data and models from disk."
            ),
        ):
            st.session_state["project_confirm_delete"] = True
            st.rerun()
    st.divider()

# ── Builder mode: render the setup wizard instead of the dashboard ────────────
if st.session_state.get("builder_mode"):
    from src.builder_ui import render_wizard

    render_wizard(header_fn=_render_page_header)
    st.stop()

# ── Incomplete project: friendly setup panel; skip init()/simulate_forecast ───
# A freshly created project has demand only (no weather columns, no models), so
# running the forecast path would crash. Show a graceful checklist instead.
if not project.is_ready():
    from src.builder_ui import render_incomplete_project

    # A project with its demand+weather dataset but no trained models yet
    # (``awaiting_training``) must be trainable — otherwise the user is stuck:
    # the Retrain UI is the only way to create the models, but it used to be
    # gated behind already having them. Render the SAME sidebar version
    # selector + 🔁 Retrain expander used by ready projects, driven by the
    # project's own data loaded directly via ``load_raw`` (NOT ``init()``,
    # which assumes the 48 PKLs exist). Done BEFORE ``st.stop()`` so the
    # expander is reachable. ``awaiting_weather`` projects (no dataset yet)
    # skip this — they must fetch weather before training is possible.
    if project.data_path.exists():
        df_raw = load_raw(project.data_path, target=project.target_column)
        with st.sidebar:
            _render_version_and_retrain_sidebar(df_raw, mh_eval=None, mh_features=None)
            st.divider()

    render_incomplete_project(project, header_fn=_render_page_header)
    st.stop()

# ── Active pkl_dir from session state ────────────────────────────────────────
if "pending_version_switch" in st.session_state:
    _ps = st.session_state.pop("pending_version_switch")
    if _ps.get("id") is None:
        st.session_state.pop("active_version_id", None)
    else:
        st.session_state["active_version_id"] = _ps["id"]
    st.session_state["active_pkl_dir"] = _ps["pkl_dir"]

pkl_dir = st.session_state.get("active_pkl_dir", project.rel_default_pkl_dir)
df_raw, mh_models, mh_eval, mh_features = init(active_project_id, pkl_dir)

# ── Display-time unit scaling (W / kW → MW) ───────────────────────────────────
# Projects that store raw data in W or kW receive a display scale factor applied
# at render time.  The CSV and models are NEVER modified — both history (from
# df_raw) and forecast (from model outputs) are multiplied by the same factor
# before being shown in any chart or KPI, so everything always displays in MW.
# MAPE calculations are unit-agnostic and remain on the original scale.
_DU_FACTORS: dict[str, float] = {"W": 1e-6, "kW": 1e-3, "kWh_per_hour": 1e-3}
_du_factor: float = _DU_FACTORS.get(project.demand_unit, 1.0)
_display_unit: str = "MW" if _du_factor != 1.0 else project.demand_unit

if "forecast_horizon" not in st.session_state:
    st.session_state.forecast_horizon = 48

# ── Substation helper functions (must be defined before sidebar/tab code) ──────

def _render_substation_retrain_sidebar(
    project: "Project",
    sub_id: str,
    df_raw: "pd.DataFrame",
) -> None:
    """Full training UI for the currently selected substation.

    Mirrors _render_version_and_retrain_sidebar but scoped to sub_id:
    version selector → model specs → retrain expander (with feature probe,
    fast/slow speed, weather-only option, and subprocess training).
    Call this inside a ``with st.sidebar:`` block.
    """
    sub_meta  = next((s for s in project.substations if s["id"] == sub_id), None)
    sub_col   = sub_meta.get("column", sub_id) if sub_meta else sub_id
    sub_name  = sub_meta["name"] if sub_meta else sub_id

    sub_models, sub_features, sub_eval = _load_sub_models(project, sub_id)
    has_sub_model = bool(sub_models)

    # ── Version selector ─────────────────────────────────────────────────────
    st.markdown(f"**Model version** · {sub_name}")
    sub_reg = ModelRegistry(
        registry_path=project.substation_registry_path(sub_id),
        default_meta_path=project.substation_default_meta_path(sub_id),
        default_pkl_dir=project.rel_substation_pkl_dir(sub_id),
    )
    sub_versions = sub_reg.list_versions()

    if not sub_versions:
        st.info("No model versions yet — use **Train** below to train the first model.")
    else:
        def _fmt_sub_version(v: dict) -> str:
            lbl = f"  {v['label']}" if v.get("label") else ""
            return (
                f"{_format_date_range(v['date_range_start'], v['date_range_end'])}"
                f"{lbl}  (MAPE {v['mape_avg']:.1f}%)"
            )

        _sv_ids = [v["id"] for v in sub_versions]
        _sv_map = {v["id"]: v for v in sub_versions}
        _sv_key = f"sub_active_version_{sub_id}"
        if _sv_key not in st.session_state or st.session_state[_sv_key] not in _sv_map:
            st.session_state[_sv_key] = _sv_ids[0]

        st.selectbox(
            "Select version",
            options=_sv_ids,
            format_func=lambda vid: _fmt_sub_version(_sv_map[vid]),
            key=_sv_key,
            label_visibility="collapsed",
        )

        _sv_sel = _sv_map[st.session_state[_sv_key]]
        # Use per-version data from the registry (not the loaded model) so the
        # specs update when the user switches versions in the dropdown.
        _spec_mape = (
            f"{_sv_sel['mape_avg']:.1f}%"
            if _sv_sel.get("mape_avg") is not None and not _is_nan_like(_sv_sel.get("mape_avg"))
            else (f"{round(float(sub_eval['mape_pct'].mean()), 1)}%" if sub_eval is not None else "—")
        )
        _spec_mae = (
            f"{round(float(sub_eval['mae_mw'].mean()) * _du_factor, 3)} {_display_unit}"
            if sub_eval is not None else "—"
        )
        _spec_nfeat = len(sub_features) if sub_features else "—"
        _spec_wo = _model_is_weather_only(sub_features) if sub_features else False
        _spec_type = "🌤️ Weather-only" if _spec_wo else "📊 Demand + weather"
        with st.expander("Model specifications", expanded=True):
            for _sk, _sv in [
                ("Period", f"{_sv_sel.get('date_range_start','?')} → {_sv_sel.get('date_range_end','?')}"),
                ("Avg MAPE", _spec_mape),
                ("Avg MAE",  _spec_mae),
                ("Features", str(_spec_nfeat)),
                ("Type",     _spec_type),
            ]:
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'font-size:0.82em;line-height:1.7;">'
                    f'<span style="color:{GRAY_MID};font-weight:500;">{_sk}</span>'
                    f'<span style="color:{TEXT};font-weight:600;">{_sv}</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # ── Retrain expander ─────────────────────────────────────────────────────
    _rt_data_start   = df_raw.index.min()
    _rt_data_end     = df_raw.index.max()
    _rt_default_end  = (
        pd.Timestamp(project.test_start) - pd.Timedelta(days=1)
        if project.test_start else _rt_data_end
    )
    _rt_default_end = min(
        max(_rt_default_end, _rt_data_start + pd.Timedelta(days=90)),
        _rt_data_end,
    )

    _spfx = f"sub_{sub_id}_rt_"  # session-state key prefix for this sub's retrain state
    _pending_key = f"{_spfx}pending"
    _job_key     = f"{_spfx}job"
    _nonce_key   = f"{_spfx}nonce"

    success_name = st.session_state.pop(f"{_spfx}success_name", None)
    if success_name:
        st.success(f"Model **{success_name}** ready — select it above.")

    with st.expander(f"🔁 Train · {sub_name}", expanded=not has_sub_model):
        with st.form(f"sub_rt_form_{sub_id}", clear_on_submit=False):
            _rc1, _rc2 = st.columns(2)
            with _rc1:
                _rt_start = st.date_input(
                    "Start date", value=_rt_data_start.date(),
                    min_value=_rt_data_start.date(), max_value=_rt_data_end.date(),
                    key=f"{_spfx}start",
                )
            with _rc2:
                _rt_end = st.date_input(
                    "End date", value=_rt_default_end.date(),
                    min_value=(_rt_data_start + pd.Timedelta(days=90)).date(),
                    max_value=_rt_data_end.date(),
                    key=f"{_spfx}end",
                )
            _rt_label = st.text_input(
                "Label (optional)", placeholder="e.g. summer-only",
                key=f"{_spfx}label",
            )
            _rt_model_type = st.radio(
                "Model type",
                options=["Standard (demand lags + weather)", "Weather-only (no lags)"],
                horizontal=True,
                key=f"{_spfx}model_type",
            )
            _rt_speed = st.radio(
                "Training speed",
                options=["Full (500 trees — best accuracy)", "Fast (100 trees — ~5× quicker)"],
                horizontal=True,
                key=f"{_spfx}speed",
            )
            _analyse_btn = st.form_submit_button("Continue →", use_container_width=True)

        if _analyse_btn:
            if _rt_end <= _rt_start + timedelta(days=90):
                st.warning("End date must be more than 90 days after start date.")
            else:
                _is_wo   = _rt_model_type.startswith("Weather-only")
                _is_fast = _rt_speed.startswith("Fast")
                if _is_wo:
                    st.session_state[_pending_key] = {
                        "start": str(_rt_start), "end": str(_rt_end),
                        "label": _rt_label.strip(), "model_type": "weather-only",
                        "fast": _is_fast, "importance": {}, "selected": [],
                    }
                else:
                    importance = probe_feature_importance(
                        active_project_id, str(_rt_start), str(_rt_end),
                        sub_column=sub_col,
                    )
                    auto_selected = sorted(auto_select_features(importance))
                    st.session_state[_pending_key] = {
                        "start": str(_rt_start), "end": str(_rt_end),
                        "label": _rt_label.strip(), "model_type": "standard",
                        "fast": _is_fast, "importance": importance,
                        "selected": auto_selected,
                    }
                    st.session_state[_nonce_key] = 0
                st.rerun()

        _pending  = st.session_state.get(_pending_key)
        _train_job = st.session_state.get(_job_key)

        if _pending:
            _is_wo = _pending.get("model_type") == "weather-only"

            if not _is_wo:
                with st.expander(
                    f"Feature relevance · {_pending['start']} → {_pending['end']}",
                    expanded=False,
                ):
                    _is_locked = bool(_train_job)
                    if _is_locked:
                        st.caption("Training in progress — feature selection is locked.")
                    else:
                        st.caption("Click a bar to include or exclude. Selected = accent + **bold**.")

                    _imp_rows = _importance_rows(_pending["importance"])
                    _feat_order = [f for f, _ in _imp_rows]
                    _selected_set = set(_pending["selected"])
                    _fig_probe = _build_feature_chart(
                        _imp_rows, _selected_set,
                        xaxis_title="Importance (probe h=24)",
                        disabled=_is_locked, max_height=360,
                    )
                    _ck = f"sub_feat_probe_{sub_id}_{st.session_state.get(_nonce_key, 0)}"
                    if _is_locked:
                        st.plotly_chart(_fig_probe, use_container_width=True, key=_ck)
                    else:
                        _ev = st.plotly_chart(
                            _fig_probe, use_container_width=True,
                            on_select="rerun", selection_mode="points", key=_ck,
                        )
                        _toggle_feature_from_chart(_ev, _feat_order)
                    st.caption(f"**{len(_pending['selected'])}** of {len(_feat_order)} features selected")
            else:
                st.info("Weather-only model: trains with weather + calendar features only.", icon="🌤️")

            if not _train_job:
                if st.button("Start training", type="primary",
                             use_container_width=True, key=f"sub_train_start_{sub_id}"):
                    _is_wo = _pending.get("model_type") == "weather-only"
                    _ver_label = _pending["label"] or "v1"
                    _out_dir = (
                        f"projects/{active_project_id}/models/substations/{sub_id}/versions/"
                        f"{_pending['start']}_{_pending['end']}_{_ver_label}/"
                    )
                    _cmd = [
                        sys.executable, "train_multi_horizon.py",
                        "--project",    active_project_id,
                        "--substation", sub_id,
                        "--start-date", _pending["start"],
                        "--end-date",   _pending["end"],
                        "--output-dir", str(REPO_ROOT / _out_dir),
                        "--label",      _ver_label,
                    ]
                    if _pending.get("fast"):
                        _cmd.append("--fast")
                    if not _is_wo and len(_pending.get("selected", [])) >= 3:
                        _cmd += ["--features", ",".join(sorted(_pending["selected"]))]
                    st.session_state[_job_key] = {
                        "cmd": _cmd, "output_dir": _out_dir,
                        "start": _pending["start"], "end": _pending["end"],
                        "label": _ver_label,
                    }
                    st.rerun()
            else:
                # ── Run the training subprocess (streaming output) ────────────
                with st.status(f"Training {sub_name}…", expanded=True) as _sts:
                    _log = st.empty()
                    _lines: list[str] = []
                    _proc = subprocess.Popen(
                        _train_job["cmd"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        cwd=str(REPO_ROOT),
                    )
                    for _line in _proc.stdout:
                        _line = _line.rstrip()
                        if _line:
                            _lines.append(_line)
                            _log.code("\n".join(_lines[-25:]), language=None)
                    _proc.wait()
                    if _proc.returncode == 0:
                        _sts.update(label=f"{sub_name} trained successfully ✅", state="complete")
                        # Version is already registered by train_multi_horizon.py itself;
                        # just clear the model cache and rerun.
                        st.session_state.pop(_job_key, None)
                        st.session_state.pop(_pending_key, None)
                        st.session_state[f"{_spfx}success_name"] = _train_job["label"]
                        st.cache_resource.clear()
                        st.rerun()
                    else:
                        _sts.update(label="Training failed", state="error")
                        st.session_state.pop(_job_key, None)
                        st.rerun()



@st.cache_resource(show_spinner=False)
def _load_models_from_dir(pkl_dir_str: str) -> tuple[dict, list[str], "pd.DataFrame | None"]:
    """Load PKL models + features from a directory (cached by path).

    Cached with @st.cache_resource so re-selecting a substation does not re-read
    up to 48 pickle files from disk every time.  The cache is cleared on retrain
    (st.cache_resource.clear()), so models stay fresh.
    """
    import pickle as _pickle, json as _json
    from pathlib import Path as _Path
    pkl_dir = _Path(pkl_dir_str)
    if not pkl_dir.is_dir():
        return {}, [], None
    meta_path = pkl_dir / "meta.json"
    if not meta_path.exists():
        return {}, [], None
    with open(meta_path) as _f:
        _meta = _json.load(_f)
    _features = _meta.get("features", list(MH_FEATURES))
    _eval_df = pd.DataFrame(_meta["eval"]) if "eval" in _meta else None
    _models: dict[int, "lgb.LGBMRegressor"] = {}
    for h in range(1, 49):
        p = pkl_dir / f"lgbm_h{h:02d}.pkl"
        if p.exists():
            try:
                with open(p, "rb") as _f:
                    _models[h] = _pickle.load(_f)
            except Exception:
                # Corrupted/truncated pkl — delete so it can be retrained
                try:
                    p.unlink()
                except OSError:
                    pass
    return _models, _features, _eval_df


def _load_sub_models(
    project: "Project", sub_id: str, pkl_dir_override: str | None = None
) -> tuple[dict, list[str], "pd.DataFrame | None"]:
    """Load PKL models + features for a substation (delegates to cached loader).

    Resolution order:
      1. Per-substation version chosen in the substation tab (pkl_dir_override)
      2. Default for this substation: substation_default_pkl_dir
    Returns (models, features, eval_df) or ({}, [], None) on missing/error.
    """
    if pkl_dir_override:
        pkl_dir = REPO_ROOT / pkl_dir_override.rstrip("/")
    else:
        pkl_dir = project.substation_default_pkl_dir(sub_id)
    return _load_models_from_dir(str(pkl_dir))


@st.cache_data(show_spinner=False)
def _sub_mape_for_paths(
    reg_path_str: str, meta_path_str: str, pkl_dir_str: str, _mtime: float
) -> float | None:
    """Cached MAPE lookup keyed by registry path + mtime.

    Caching avoids re-reading every substation's registry JSON on every render
    (90 tiles → 90 disk reads).  The ``_mtime`` arg invalidates the cache when the
    registry file changes after a retrain.
    """
    from pathlib import Path as _Path
    try:
        reg = ModelRegistry(
            registry_path=_Path(reg_path_str),
            default_meta_path=_Path(meta_path_str),
            default_pkl_dir=pkl_dir_str,
        )
        versions = reg.list_versions()
        if not versions:
            return None
        return versions[0].get("mape_avg")
    except Exception:
        return None


def _sub_mape(project: "Project", sub_id: str) -> float | None:
    """Return the average MAPE from the substation's latest registry entry, or None."""
    reg_path = project.substation_registry_path(sub_id)
    if not reg_path.exists():
        return None
    return _sub_mape_for_paths(
        str(reg_path),
        str(project.substation_default_meta_path(sub_id)),
        project.rel_substation_pkl_dir(sub_id),
        reg_path.stat().st_mtime,
    )


@st.cache_data(show_spinner=False)
def _sub_window_mape_cached(
    sub_id: str, sub_col: str, snapshot_iso: str, horizon: int, _mtime: float,
    _project: "Project", _df_raw: "pd.DataFrame",
) -> float | None:
    """Forecast-vs-actual MAPE over the displayed window for one substation.

    This is the same "MAPE of period displayed" metric shown above the substation
    chart, computed per substation so each tile reflects how that model did at THIS
    snapshot.  Cached on (sub_id, snapshot, horizon, model mtime) so the tile grid
    only recomputes a substation when one of those actually changes — ``_project``
    and ``_df_raw`` are excluded from the hash (leading underscore).
    """
    from src.model import simulate_forecast as _sf
    try:
        sub_models, sub_features, sub_eval = _load_sub_models(_project, sub_id)
        if not sub_models:
            return None
        horizon_sub = {h: m for h, m in sub_models.items() if h <= horizon}
        if not horizon_sub:
            return None
        res = _sf(
            _df_raw, horizon_sub, pd.Timestamp(snapshot_iso),
            history_hours=24, features=sub_features, mh_eval=sub_eval,
            target=sub_col,
        )
        if res is None:
            return None
        return compute_mape(res["actual_future"], res["forecast"])
    except Exception:
        return None


def _sub_window_mape(
    project: "Project", df_raw: "pd.DataFrame", sub_id: str, sub_col: str,
    snapshot_dt, horizon: int,
) -> float | None:
    """MAPE of the displayed period for one substation (snapshot-aware, per substation)."""
    reg_path = project.substation_registry_path(sub_id)
    if not reg_path.exists() or snapshot_dt is None:
        return None
    return _sub_window_mape_cached(
        sub_id, sub_col, pd.Timestamp(snapshot_dt).isoformat(), int(horizon),
        reg_path.stat().st_mtime, project, df_raw,
    )


def _aggregate_ref_mape(
    project: "Project",
    df_raw: "pd.DataFrame",
) -> float | None:
    """Aggregate reference MAPE for multi-substation projects.

    MAE is additive: aggregate_MAE = Σ substation_MAEs.
    aggregate_MAPE = aggregate_MAE / aggregate_mean_demand × 100.

    This is equivalent to comparing summed substation forecasts against summed
    actuals over the full test period — no re-running of forecasts needed.
    Returns None if no substation has eval data.
    """
    total_mae = 0.0
    total_mean_demand = 0.0
    found_any = False
    for sub in project.substations:
        sub_id  = sub["id"]
        sub_col = sub.get("column", sub_id)
        _, _, sub_eval = _load_sub_models(project, sub_id)
        if sub_eval is None or sub_eval.empty:
            continue
        sub_mae    = float(sub_eval["mae_mw"].mean())
        if sub_col in df_raw.columns:
            sub_mean = float(df_raw[sub_col].dropna().mean())
        else:
            sub_mean = float(sub_eval.get("mae_mw", pd.Series([0])).mean())  # fallback
        if sub_mean <= 0:
            continue
        total_mae         += sub_mae
        total_mean_demand += sub_mean
        found_any = True
    if not found_any or total_mean_demand <= 0:
        return None
    return round(total_mae / total_mean_demand * 100.0, 1)


def _mape_tile_style(mape: float | None) -> tuple[str, str]:
    """Return (bg_color, label_color) for a substation tile based on MAPE."""
    if mape is None:
        return "rgba(156,163,175,0.15)", GRAY_MID
    if mape < 8.0:
        return "rgba(22,163,74,0.12)", "#16a34a"
    if mape < 15.0:
        return "rgba(202,138,4,0.12)", "#ca8a04"
    return "rgba(220,38,38,0.12)", "#dc2626"


def _simulate_aggregate_from_substations(
    df_raw: "pd.DataFrame",
    project: "Project",
    snapshot_dt: "pd.Timestamp",
    horizon: int,
    wx_forecast,
    mc_samples: int,
) -> dict | None:
    """Sum individual substation forecasts to produce the aggregate result dict.

    This replaces the separate aggregate model for multi-mode projects.
    History and actual-future are derived by summing the substation columns
    directly from df_raw (no trained model needed for the aggregate).
    """
    from src.model import simulate_forecast as _sf

    agg_fc = agg_lo = agg_hi = None
    last_weather = None

    for sub in project.substations:
        sub_id = sub["id"]
        sub_col = sub.get("column", sub_id)
        sub_models, sub_features, sub_eval = _load_sub_models(project, sub_id)
        if not sub_models:
            continue
        horizon_sub = {h: m for h, m in sub_models.items() if h <= horizon}
        if not horizon_sub:
            continue
        sub_result = _sf(
            df_raw, horizon_sub, snapshot_dt,
            n_samples=mc_samples,
            wx_forecast=wx_forecast,
            history_hours=24,
            features=sub_features,
            mh_eval=sub_eval,
            target=sub_col,
        )
        if sub_result is None:
            continue

        # Apply per-substation manual overrides before summing into aggregate
        _sub_ovs = st.session_state.get(f"sub_overrides_{sub_id}", [])
        if _sub_ovs:
            _sfc, _slo, _shi = apply_forecast_overrides(
                sub_result["forecast"], sub_result["lower"], sub_result["upper"],
                _sub_ovs,
            )
        else:
            _sfc, _slo, _shi = sub_result["forecast"], sub_result["lower"], sub_result["upper"]

        if agg_fc is None:
            agg_fc = _sfc.copy()
            agg_lo = _slo.copy()
            agg_hi = _shi.copy()
        else:
            agg_fc = agg_fc.add(_sfc, fill_value=0)
            agg_lo = agg_lo.add(_slo, fill_value=0)
            agg_hi = agg_hi.add(_shi, fill_value=0)
        last_weather = sub_result.get("weather")

    if agg_fc is None:
        return None

    sub_cols = [
        s.get("column", s["id"])
        for s in project.substations
        if s.get("column", s["id"]) in df_raw.columns
    ]
    agg_series = df_raw[sub_cols].sum(axis=1)
    history = agg_series.loc[snapshot_dt - pd.Timedelta(hours=24) : snapshot_dt]
    actual_future = agg_series.loc[
        snapshot_dt + pd.Timedelta(hours=1) : snapshot_dt + pd.Timedelta(hours=horizon)
    ]

    return {
        "forecast": agg_fc,
        "lower": agg_lo,
        "upper": agg_hi,
        "history": history,
        "actual_future": actual_future,
        "weather": last_weather,
    }


# ── Shared chart helpers (must be defined before sidebar/tab code) ────────────

def _render_snapshot_picker(
    df_raw: "pd.DataFrame",
    project: "Project",
    key_prefix: str = "",
    live_key: str | None = None,
) -> "pd.Timestamp | tuple[pd.Timestamp, bool]":
    """Render the snapshot date+hour picker, return snapshot_dt (or (snapshot_dt, live_mode)).

    Pass ``live_key`` to add a live-mode toggle as the third column.  When live
    mode is active the date/hour inputs are disabled and the function returns
    ``(snapshot_dt, True)``.  Without ``live_key`` it returns just ``snapshot_dt``.
    """
    _data_start    = df_raw.index.min()
    _data_end      = df_raw.index.max()
    # Allow any snapshot that has at least 336 h (14 days) of history for lag
    # features.  Do NOT clamp to test_start — users should be able to replay
    # any date in the dataset, including the training period.
    _min_snap = _data_start + pd.Timedelta(hours=336)
    _max_snap = _data_end - pd.Timedelta(hours=48)
    if _min_snap > _max_snap:
        _min_snap = _max_snap
    # Default: one 48h forecast-window before the data ends (so ground truth is visible)
    _default_snap = _max_snap - pd.Timedelta(days=3)
    if _default_snap < _min_snap:
        _default_snap = _min_snap

    # Reset stored date when it falls outside the valid range for THIS project
    # (can happen when switching between projects with different data extents,
    # or when test_start was previously used as the lower bound).
    _date_key = f"{key_prefix}snap_date"
    _stored = st.session_state.get(_date_key)
    if _stored is not None:
        import datetime as _dt
        _stored_d = _stored if isinstance(_stored, _dt.date) else getattr(_stored, "date", lambda: _stored)()
        if not (_min_snap.date() <= _stored_d <= _max_snap.date()):
            del st.session_state[_date_key]

    # Read live-mode state from session state before rendering (needed to style label).
    _live_mode = st.session_state.get(live_key, False) if live_key else False

    _label_text  = (
        "Live mode active — snapshot controls disabled"
        if _live_mode
        else "🕐 Replay a historic forecast — select date & time"
    )
    _label_color = GRAY_MID if _live_mode else GRAY
    st.markdown(
        f'<span style="font-size:0.78em;font-weight:600;text-transform:uppercase;'
        f'letter-spacing:0.07em;color:{_label_color};">{_label_text}</span>',
        unsafe_allow_html=True,
    )

    if live_key:
        _snap_date_col, _snap_time_col, _snap_live_col = st.columns([2, 1, 1])
        _live_mode = _snap_live_col.toggle("Live forecast", key=live_key)
    else:
        _snap_date_col, _snap_time_col, _ = st.columns([2, 1, 3])

    import datetime as _dt
    # Streamlit's year-dropdown navigates to Jan 1 of the clicked year.  If
    # min_value is mid-year (e.g. 2025-09-15) that navigation is silently
    # blocked and the user appears "stuck".  Fix: open the calendar to Jan 1
    # of the earliest year so the year dropdown works, then clamp the result
    # in Python so the app never uses a date before the actual data window.
    _picker_min = _dt.date(_min_snap.year, 1, 1)
    snap_date_raw = _snap_date_col.date_input(
        "Snapshot date",
        value=_default_snap.date(),
        min_value=_picker_min,
        max_value=_max_snap.date(),
        help=f"Available range: {_min_snap.strftime('%d %b %Y')} → {_max_snap.strftime('%d %b %Y')}.",
        key=f"{key_prefix}snap_date",
        disabled=_live_mode,
    )
    snap_date = max(snap_date_raw, _min_snap.date())
    sel_hour_str = _snap_time_col.selectbox(
        "Hour",
        options=[f"{h:02d}:00" for h in range(24)],
        index=14,
        key=f"{key_prefix}snap_hour",
        disabled=_live_mode,
    )
    snapshot_dt = pd.Timestamp(snap_date) + pd.Timedelta(hours=int(sel_hour_str.split(":")[0]))
    if live_key:
        return snapshot_dt, _live_mode
    return snapshot_dt


def _add_demand_traces(
    fig: "go.Figure",
    history: "pd.Series",
    forecast: "pd.Series",
    lower: "pd.Series",
    upper: "pd.Series",
    actual_future: "pd.Series",
    snapshot_dt: "pd.Timestamp",
    forecast_base: "pd.Series",
    overrides: list,
    has_overrides: bool,
    demand_unit: str,
    row: "int | None" = None,
    col: "int | None" = None,
) -> None:
    """Add all demand-row traces to *fig* (subplots with row/col, or standalone go.Figure)."""
    _rc = dict(row=row, col=col) if row is not None else {}

    fig.add_vrect(
        x0=snapshot_dt.isoformat(),
        x1=forecast.index[-1].isoformat(),
        fillcolor=FORECAST_SHADE, line_width=0, layer="below",
        **_rc,
    )
    fig.add_trace(
        go.Scatter(
            x=history.index, y=history.values,
            name="Actual (history)", mode="lines",
            line=dict(color=PRIMARY, width=2.5),
            hovertemplate=f"%{{y:.0f}} {demand_unit}<extra></extra>",
        ),
        **_rc,
    )
    _anchor_ts  = history.index[-1]
    _anchor_val = float(history.iloc[-1])
    _upper_b = pd.concat([pd.Series([_anchor_val], index=[_anchor_ts]), upper])
    _lower_b = pd.concat([pd.Series([_anchor_val], index=[_anchor_ts]), lower])
    fig.add_trace(
        go.Scatter(
            x=list(_upper_b.index) + list(_lower_b.index[::-1]),
            y=list(_upper_b.values) + list(_lower_b.values[::-1]),
            name="80% CI (Monte Carlo)",
            fill="toself", fillcolor=BAND_COL,
            line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
        ),
        **_rc,
    )
    fig.add_trace(
        go.Scatter(
            x=[_anchor_ts, forecast.index[0]],
            y=[_anchor_val, float(forecast.iloc[0])],
            mode="lines", line=dict(color=FORECAST, width=2.5),
            showlegend=False, hoverinfo="skip",
        ),
        **_rc,
    )
    fig.add_trace(
        go.Scatter(
            x=forecast.index, y=forecast.values,
            name="Forecast (median)", mode="lines",
            line=dict(color=FORECAST, width=2, dash="dash"),
            hovertemplate=f"%{{y:.0f}} {demand_unit}<extra></extra>",
        ),
        **_rc,
    )
    # CONSTANT TRACE STRUCTURE — these two traces are ALWAYS added (in this fixed
    # order), regardless of whether overrides exist or ground truth is available.
    # Streamlit/Plotly re-renders the chart in place only when the number/order of
    # traces is unchanged; a varying trace count forces a full re-render that yanks
    # the page scroll to the top. So we keep them present and just toggle their
    # visibility / data instead of conditionally adding them.
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="markers",
            name="Override active",
            marker=dict(size=12, color=OVERRIDE_MARK, symbol="square"),
            showlegend=bool(has_overrides),
            visible=True if has_overrides else "legendonly",
        ),
        **_rc,
    )
    if has_overrides:
        for _ov in overrides:
            v0 = max(pd.Timestamp(_ov["from_dt"]), forecast_base.index[0])
            v1 = min(pd.Timestamp(_ov["to_dt"]),   forecast_base.index[-1])
            if v0 <= v1:
                fig.add_vrect(
                    x0=v0.isoformat(), x1=v1.isoformat(),
                    fillcolor=OVERRIDE_FILL, line_width=0,
                    **_rc,
                )
    _has_truth = actual_future is not None and not actual_future.empty
    fig.add_trace(
        go.Scatter(
            x=(actual_future.index if _has_truth else [None]),
            y=(actual_future.values if _has_truth else [None]),
            name="Actual (future, ground truth)", mode="lines",
            line=dict(color=PRIMARY, width=1.5, dash="dot"),
            opacity=0.40,
            showlegend=bool(_has_truth),
            hovertemplate=f"%{{y:.0f}} {demand_unit}<extra></extra>",
        ),
        **_rc,
    )
    fig.add_vline(
        x=snapshot_dt.isoformat(),
        line=dict(color=GRAY_MID, width=1, dash="dot"),
        **_rc,
    )


def _apply_demand_chart_layout(
    fig: "go.Figure",
    history: "pd.Series",
    forecast: "pd.Series",
    demand_unit: str,
    row: "int | None" = None,
    col: "int | None" = None,
    height: int = 320,
    add_day_labels: bool = False,
) -> None:
    """Apply the standard demand-chart layout to *fig*.

    Shared by both tabs.  Pass add_day_labels=True only for the aggregate tab,
    which has enough top-margin real-estate to show the floating day annotations
    above the tick marks without overlap.  The substation standalone chart is
    shorter (height≈320) so day labels are omitted there — the snapshot picker
    above the chart already provides full date context.
    """
    _rc_kw = dict(row=row, col=col) if row is not None else {}

    first_ts = history.index[0]
    last_ts  = forecast.index[-1]
    _span_h  = (last_ts - first_ts).total_seconds() / 3600
    _dtick_ms, _tick_angle, _top_margin, _day_label_y = _chart_time_axis(_span_h)

    # When day labels are requested always use diagonal ticks and enough top margin,
    # regardless of span (short substation windows can otherwise produce ≤10 ticks
    # which the threshold converts to tickangle=0 causing visual clutter).
    if add_day_labels and _tick_angle == 0:
        _tick_angle  = -45
        _top_margin  = 152
        _day_label_y = 1.24

    all_days = pd.date_range(start=first_ts.normalize(), end=last_ts.normalize(), freq="D")
    for midnight in all_days[1:]:
        fig.add_vline(
            x=midnight.isoformat(), line_width=1,
            line_color=GRAY_MID, line_dash="dot", opacity=0.35,
            **_rc_kw,
        )

    if add_day_labels:
        for day in all_days:
            seg_start = max(day, first_ts)
            seg_end   = min(day + pd.Timedelta(days=1), last_ts)
            midpoint  = seg_start + (seg_end - seg_start) / 2
            fig.add_annotation(
                x=midpoint.isoformat(), y=_day_label_y,
                xref="x", yref="paper",
                text=day.strftime("<b>%a %d %b</b>"),
                showarrow=False, font=dict(size=15, color=TEXT), xanchor="center",
            )

    _leg_style = dict(
        xanchor="left", yanchor="top", orientation="v",
        bgcolor="rgba(255,255,255,0.75)", borderwidth=0,
        font=dict(size=11, color=TEXT_MUT, family="Inter, sans-serif"),
    )
    fig.update_layout(
        height=height,
        margin=dict(t=_top_margin, b=20, l=52, r=185),
        showlegend=True,
        legend=dict(x=1.02, y=0.99, **_leg_style),
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        # Constant uirevision → Plotly.js treats each new figure as an in-place
        # update of the SAME plot (rather than a brand-new one), so switching
        # substations updates the existing chart instead of remounting the
        # component. Remounting is what yanked the page scroll to the top.
        uirevision="demand-chart",
    )
    fig.update_yaxes(
        title_text=f"Heat demand<br>{demand_unit}",
        gridcolor=GRID_COL, zeroline=False,
        title_font=dict(color=TEXT_MUT), tickfont=dict(color=TEXT_MUT),
        **_rc_kw,
    )
    first_midnight = first_ts.normalize().isoformat()
    fig.update_xaxes(
        showgrid=False,
        side="top", tickangle=_tick_angle, showticklabels=True,
        tick0=first_midnight, dtick=_dtick_ms, tickformat="%H:%M",
        tickfont=dict(size=12, color=TEXT_MUT), tickcolor=GRAY_LITE,
        ticks="outside", ticklen=4,
        **_rc_kw,
    )


# Single demand panel height so its plotting area matches the aggregate demand
# row (600 * 0.54 ≈ 324px) plus the shared top/bottom margins (152 + 20).
SUB_CHART_HEIGHT = round(600 * 0.54 + 172)  # ≈ 496


def build_demand_figure(
    history: "pd.Series",
    forecast: "pd.Series",
    lower: "pd.Series",
    upper: "pd.Series",
    actual_future: "pd.Series",
    snapshot_dt: "pd.Timestamp",
    forecast_base: "pd.Series",
    *,
    overrides: list,
    has_overrides: bool,
    demand_unit: str,
    show_weather: bool = False,
    weather: "pd.DataFrame | None" = None,
    df_raw: "pd.DataFrame | None" = None,
    height: "int | None" = None,
) -> "go.Figure":
    """Build the demand (+ optional weather) figure shared by both tabs.

    This is the single source of truth for the demand chart used in the Aggregate
    demand forecast tab and the Substation Detail tab.  When ``show_weather`` is
    True the temperature/wind and cloud/humidity rows are appended as a 3-row
    subplot (identical to the aggregate layout); otherwise a single demand panel
    is returned, sized so its plotting area matches the aggregate demand row.
    The weather section is "collapsible" via the ``show_weather`` flag, which the
    callers wire up to a per-tab ``st.toggle``.
    """
    _WX_COLS = ["temperature_c", "wind_speed_ms", "cloud_cover_pct", "humidity_pct"]

    if show_weather:
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.54, 0.26, 0.16], vertical_spacing=0.07,
            specs=[
                [{"secondary_y": False}],
                [{"secondary_y": True}],
                [{"secondary_y": True}],
            ],
        )
        _row, _col = 1, 1
    else:
        fig = go.Figure()
        _row, _col = None, None

    # Forecast-region shading on weather rows 2 & 3 (row 1 handled by _add_demand_traces)
    if show_weather:
        _fc_end_ts = forecast.index[-1]
        for _shade_row in (2, 3):
            fig.add_vrect(
                x0=snapshot_dt.isoformat(), x1=_fc_end_ts.isoformat(),
                fillcolor=FORECAST_SHADE, line_width=0, layer="below",
                row=_shade_row, col=1,
            )

    # All demand-row traces (shading, history, CI, handoff, forecast, ground truth, overrides, vline)
    _add_demand_traces(
        fig,
        history, forecast, lower, upper, actual_future,
        snapshot_dt, forecast_base,
        overrides=overrides,
        has_overrides=has_overrides,
        demand_unit=demand_unit,
        row=_row, col=_col,
    )

    if show_weather:
        hist_wx = df_raw.loc[history.index, _WX_COLS]
        all_wx = pd.concat([hist_wx, weather[_WX_COLS]]).sort_index()

        fig.add_trace(
            go.Scatter(
                x=all_wx.index, y=all_wx["temperature_c"],
                name="Temperature (°C)", mode="lines",
                line=dict(color=ACCENT, width=2),
                legend="legend2",
                hovertemplate="%{y:.1f} °C<extra></extra>",
            ),
            row=2, col=1, secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=all_wx.index, y=all_wx["wind_speed_ms"],
                name="Wind speed (m/s)", mode="lines",
                line=dict(color=ACCENT2, width=1.5, dash="dot"),
                legend="legend2",
                hovertemplate="%{y:.1f} m/s<extra></extra>",
            ),
            row=2, col=1, secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(
                x=all_wx.index, y=all_wx["cloud_cover_pct"],
                name="Cloud cover (%)", mode="lines",
                line=dict(color=GRAY, width=1.5, dash="dash"),
                legend="legend3",
                opacity=0.70,
                hovertemplate="%{y:.0f}%<extra></extra>",
            ),
            row=3, col=1, secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=all_wx.index, y=all_wx["humidity_pct"],
                name="Humidity (%)", mode="lines",
                line=dict(color=PURPLE, width=2.5),
                legend="legend3",
                hovertemplate="%{y:.0f}%<extra></extra>",
            ),
            row=3, col=1, secondary_y=True,
        )

    # NOW line (shape above fills so it is not broken by the CI band; spans full height)
    fig.add_shape(
        type="line",
        x0=snapshot_dt, x1=snapshot_dt,
        y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color=NOW_COL, width=2),
        layer="above",
    )

    _height = height if height is not None else (600 if show_weather else SUB_CHART_HEIGHT)
    _apply_demand_chart_layout(
        fig, history, forecast, demand_unit,
        row=_row, col=_col, height=_height, add_day_labels=True,
    )

    if show_weather:
        # Override top-margin to accommodate extra weather-row legend entries
        _span_h2 = (forecast.index[-1] - history.index[0]).total_seconds() / 3600
        _, _, _top_margin, _ = _chart_time_axis(_span_h2)
        fig.update_layout(
            height=_height,
            margin=dict(t=_top_margin, b=20, l=52, r=185),
            legend2=dict(x=1.02, y=0.52, xanchor="left", yanchor="top", orientation="v",
                         bgcolor="rgba(255,255,255,0.75)", borderwidth=0,
                         font=dict(size=11, color=TEXT_MUT, family="Inter, sans-serif")),
            legend3=dict(x=1.02, y=0.11, xanchor="left", yanchor="middle", orientation="v",
                         bgcolor="rgba(255,255,255,0.75)", borderwidth=0,
                         font=dict(size=11, color=TEXT_MUT, family="Inter, sans-serif")),
            barmode="overlay",
        )

        fig.add_hline(y=0, line_dash="dot", line_color=PRIMARY, opacity=0.20, row=2, col=1)
        _temp_pad = 5
        _t_min = float(all_wx["temperature_c"].min()) - _temp_pad
        _t_max = float(all_wx["temperature_c"].max()) + _temp_pad
        fig.update_yaxes(title_text="Temperature<br>°C", row=2, col=1, secondary_y=False,
                         gridcolor=GRID_COL,
                         zeroline=True, zerolinecolor=GRID_COL, zerolinewidth=1,
                         range=[_t_min, _t_max],
                         title_font=dict(color=ACCENT), tickfont=dict(color=ACCENT))
        fig.update_yaxes(title_text="Wind<br>m/s", row=2, col=1, secondary_y=True,
                         showgrid=False, zeroline=False,
                         range=[0, max(float(all_wx["wind_speed_ms"].max()) * 1.4, 8)],
                         title_font=dict(color=ACCENT2), tickfont=dict(color=ACCENT2))
        _h_pad = 5
        _h_min = float(all_wx["humidity_pct"].min()) - _h_pad
        _h_max = float(all_wx["humidity_pct"].max()) + _h_pad
        fig.update_yaxes(title_text="Cloud<br>%", row=3, col=1, secondary_y=False,
                         gridcolor=GRID_COL, zeroline=False,
                         range=[0, 100], dtick=25, fixedrange=True,
                         title_font=dict(color=TEXT_MUT), tickfont=dict(color=TEXT_MUT))
        fig.update_yaxes(title_text="Humidity<br>%", row=3, col=1, secondary_y=True,
                         showgrid=False, zeroline=False,
                         range=[max(0, _h_min), min(100, _h_max)],
                         title_font=dict(color=PURPLE), tickfont=dict(color=PURPLE))
        fig.update_xaxes(showgrid=False, showticklabels=False, row=2, col=1)
        fig.update_xaxes(showgrid=False, showticklabels=False, row=3, col=1)

    return fig


@st.fragment
def _render_substation_train_panel(project: "Project", df_raw: "pd.DataFrame") -> None:
    """Sidebar per-substation training panel (its own fragment).

    Wrapping this in a fragment means changing the "Train substation" selector
    reruns only this sidebar panel — it does NOT refresh the main/right screen.
    The selector uses the independent "sub_train" key so it is decoupled from the
    substation being *viewed* in Tab 2.
    """
    _train_ids   = [s["id"] for s in project.substations]
    _train_names = {s["id"]: s["name"] for s in project.substations}
    if st.session_state.get("sub_train") not in _train_ids:
        st.session_state["sub_train"] = _train_ids[0]

    st.divider()
    st.markdown("**Substation Detail**")
    st.caption("Pick a substation to train (independent of the one you're viewing)")
    _train_sub_id = st.selectbox(
        "Select substation",
        options=_train_ids,
        format_func=lambda _i: _train_names.get(_i, _i),
        key="sub_train",
    )
    _render_substation_retrain_sidebar(project, _train_sub_id, df_raw)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # For multi-mode projects the aggregate train widget is replaced entirely by
    # the batch controls below.  For single-mode projects keep the existing UI.
    if not project.is_multi():
        _render_version_and_retrain_sidebar(df_raw, mh_eval=mh_eval, mh_features=mh_features)

    # ── Multi-substation extras ────────────────────────────────────────────────
    if project.is_multi():
        st.divider()
        st.markdown("**Aggregate demand forecast**")
        st.caption("Controls for the aggregate / batch view (Tab 1)")

        # Show only batch_ prefixed versions so operators can pick a consistent
        # training run for all substations at once.
        # Read from the first substation's registry (all substations share the
        # same batch_ version IDs when trained together with Train all).
        _first_sub_for_reg = project.substations[0] if project.substations else None
        _batch_versions: list = []
        if _first_sub_for_reg:
            _batch_reg = ModelRegistry(
                registry_path=project.substation_registry_path(_first_sub_for_reg["id"]),
                default_meta_path=project.substation_default_meta_path(_first_sub_for_reg["id"]),
                default_pkl_dir=project.rel_substation_pkl_dir(_first_sub_for_reg["id"]),
            )
            _batch_versions = [v for v in _batch_reg.list_versions() if v.get("id", "").startswith("batch_")]

        if _batch_versions:
            _bv_options = [v["id"] for v in _batch_versions]
            _bv_selected = st.selectbox(
                "Batch training version",
                ["(individual per substation)"] + _bv_options,
                key="active_batch_version",
                help="Select a batch-trained version to use for all substations. "
                     "Individual overrides in Tab 2 still take precedence.",
            )
            st.caption(f"{len(_batch_versions)} batch version(s) available.")
        else:
            st.caption("No batch versions yet — run 'Train all' to create one.")

        with st.expander("🚀 Train all substations", expanded=False):
            st.caption(
                "Trains all substation models simultaneously using fast settings "
                "(fewer trees) and all features. The aggregate forecast is then "
                "automatically the sum of the individual substation forecasts. "
                "Version IDs are prefixed with `batch_`."
            )
            _batch_wo = st.checkbox(
                "Weather-only (enables live forecasting)",
                key="batch_weather_only",
                help=(
                    "Trains models without demand-lag features. "
                    "Required to use the live forecast toggle (no SCADA connection needed)."
                ),
            )
            if st.button("Train all", type="primary", key="train_all_btn",
                         use_container_width=True,
                         disabled=bool(st.session_state.get("train_all_running"))):
                st.session_state["train_all_running"] = True
                _all_cmds = []
                for _sub in project.substations:
                    _cmd = [
                        sys.executable, "train_multi_horizon.py",
                        "--project", project.id,
                        "--substation", _sub["id"],
                        "--fast", "--label-prefix", "batch_",
                    ]
                    if _batch_wo:
                        _cmd.append("--weather-only")
                    _all_cmds.append(_cmd)
                _ta_total = len(_all_cmds)
                _ta_errors = []
                _ta_progress = st.progress(0, text="Starting batch training…")
                _ta_status   = st.empty()
                for _ta_idx, _ta_cmd in enumerate(_all_cmds):
                    _ta_label = (
                        _ta_cmd[_ta_cmd.index("--substation") + 1]
                        if "--substation" in _ta_cmd
                        else "aggregate demand"
                    )
                    _ta_pct = _ta_idx / _ta_total
                    _ta_progress.progress(
                        _ta_pct,
                        text=f"Training {_ta_idx + 1}/{_ta_total}: **{_ta_label}**",
                    )
                    _ta_status.caption(f"⏳ Running model for {_ta_label}…")
                    _ta_res = subprocess.run(
                        _ta_cmd, capture_output=True, text=True, cwd=str(REPO_ROOT)
                    )
                    if _ta_res.returncode != 0:
                        _ta_errors.append(_ta_label)
                        _ta_status.caption(f"❌ Failed: {_ta_label}")
                    else:
                        _ta_status.caption(f"✅ Done: {_ta_label}")
                if _ta_errors:
                    _ta_progress.progress(1.0, text="Training finished with errors")
                    st.error(f"Failed models: {', '.join(_ta_errors)}")
                else:
                    _ta_progress.progress(1.0, text=f"All {_ta_total} models trained successfully ✓")
                    _ta_status.empty()
                st.session_state.pop("train_all_running", None)
                st.cache_resource.clear()
                st.rerun()

        # ── Per-substation training (its own fragment so the selector does not
        # refresh the main/right screen; decoupled from the Tab 2 view via the
        # independent "sub_train" key). ────────────────────────────────────────
        if project.substations:
            _render_substation_train_panel(project, df_raw)

    mc_samples = 200

# Initialise the selected substation early so the sidebar can render training
# controls for it without the user needing to visit Tab 2 first.
if project.is_multi() and project.substations:
    if (
        "sub_selected" not in st.session_state
        or st.session_state["sub_selected"] not in [s["id"] for s in project.substations]
    ):
        st.session_state["sub_selected"] = project.substations[0]["id"]

# Always use NWP forecast weather (realistic simulation); oracle weather is a
# research-only option that was confusing in a demo context.
use_wx_forecast = True

# ── Main area ─────────────────────────────────────────────────────────────────
_render_page_header(f"Heating Demand Forecast - {project.name}")

def _simulate_live_agg(
    project: "Project",
    live_wx: "pd.DataFrame",
    now_ts: "pd.Timestamp",
    n_samples: int = 200,
) -> "dict | None":
    """Sum individual substation live (weather-only) forecasts for the aggregate live view."""
    agg_fc = agg_lo = agg_hi = None
    last_weather = None
    for sub in project.substations:
        sub_id = sub["id"]
        sub_models, sub_features, sub_eval = _load_sub_models(project, sub_id)
        if not sub_models or not _model_is_weather_only(sub_features):
            continue
        sub_result = simulate_live_forecast(
            sub_models, live_wx, now_ts,
            n_samples=n_samples,
            features=sub_features,
            country_code=project.country_code,
            mh_eval=sub_eval,
        )
        if sub_result is None:
            continue
        if agg_fc is None:
            agg_fc = sub_result["forecast"].copy()
            agg_lo = sub_result["lower"].copy()
            agg_hi = sub_result["upper"].copy()
        else:
            agg_fc = agg_fc.add(sub_result["forecast"], fill_value=0)
            agg_lo = agg_lo.add(sub_result["lower"], fill_value=0)
            agg_hi = agg_hi.add(sub_result["upper"], fill_value=0)
        last_weather = sub_result.get("weather")
    if agg_fc is None:
        return None
    return {"forecast": agg_fc, "lower": agg_lo, "upper": agg_hi, "weather": last_weather}


def _render_live_agg_section(
    project: "Project",
    models: "dict | None",
    features: "list | None",
    mh_eval: "pd.DataFrame | None",
    key_suffix: str = "",
    enable_log: bool = True,
) -> None:
    """Render the live aggregate forecast view.

    Pass ``models=None`` for multi-mode projects (will sum substations).
    Pass ``models=<dict>`` for single-mode (uses that model directly).
    """
    _LIVE_REFRESH_INTERVAL = 300  # 5 minutes
    _rkey_ts   = f"live_{key_suffix}_last_refresh_ts"
    _rkey_mono = f"live_{key_suffix}_refresh_mono"

    # ── Model availability check (single-mode only) ───────────────────────────
    is_multi_live = project.is_multi() and models is None
    if not is_multi_live:
        if not models:
            st.info(
                "No model loaded. Train a model first via **Train model** in the sidebar.",
                icon="ℹ️",
            )
            return
        if not _model_is_weather_only(features):
            st.warning(
                "The selected model uses demand lag features which require a live SCADA "
                "data connection.\n\n"
                "Select a **Weather-only** model version in the sidebar, or train a "
                "batch weather-only model (**Train all** → ☑ Weather-only) to enable "
                "live forecasting.",
                icon="⚠️",
            )
            return

    # ── Auto-refresh every 5 minutes ─────────────────────────────────────────
    _now_ts = pd.Timestamp.utcnow().tz_localize(None).floor("H")
    _elapsed_s = 0
    if st.session_state.get(_rkey_ts) is not None:
        _elapsed_s = int(_time.monotonic() - st.session_state.get(_rkey_mono, _time.monotonic()))
    if _elapsed_s >= _LIVE_REFRESH_INTERVAL:
        st.session_state[_rkey_ts]   = _now_ts
        st.session_state[_rkey_mono] = _time.monotonic()
        st.rerun()
    if _rkey_ts not in st.session_state:
        st.session_state[_rkey_ts]   = _now_ts
        st.session_state[_rkey_mono] = _time.monotonic()

    _elapsed_s = int(_time.monotonic() - st.session_state.get(_rkey_mono, _time.monotonic()))
    _next_refresh_s = max(0, _LIVE_REFRESH_INTERVAL - _elapsed_s)
    _refresh_min = _next_refresh_s // 60
    _refresh_sec = _next_refresh_s % 60

    # ── LIVE header ────────────────────────────────────────────────────────────
    from datetime import datetime as _datetime
    _actual_now = _datetime.now()
    _lhdr_left, _lhdr_right = st.columns([5, 1])
    _lhdr_left.markdown(
        f'<span style="background:{PINK};color:white;font-weight:700;padding:3px 9px;'
        f'border-radius:4px;font-size:0.82em;letter-spacing:0.06em;margin-right:10px;">'
        f'● LIVE</span>'
        f'<span style="font-size:1.05em;font-weight:600;color:{NAVY};">'
        f'{_actual_now.strftime("%A %d %b %Y · %H:%M:%S")}</span>'
        f'<span style="color:{GRAY};font-size:0.83em;margin-left:12px;">'
        f'Real-time weather from Open-Meteo · auto-refresh in '
        f'{_refresh_min}m {_refresh_sec:02d}s</span>',
        unsafe_allow_html=True,
    )
    _rkey_btn = f"live_refresh_btn_{key_suffix}"
    if _lhdr_right.button("↺ Refresh", key=_rkey_btn, use_container_width=True):
        st.session_state[_rkey_ts]   = pd.Timestamp.utcnow().tz_localize(None).floor("H")
        st.session_state[_rkey_mono] = _time.monotonic()
        st.rerun()
    st.divider()

    # ── Fetch live weather ────────────────────────────────────────────────────
    with st.spinner("Fetching live weather forecast from Open-Meteo…"):
        _live_wx = fetch_open_meteo_forecast(
            _now_ts, horizon=48,
            lat=project.lat, lon=project.lon, history_hours=24,
        )
    if _live_wx is None:
        st.error("Could not fetch weather from Open-Meteo. Check network connection.")
        return

    # ── Run live forecast ─────────────────────────────────────────────────────
    with st.spinner("Running live forecast…"):
        if is_multi_live:
            _live_result = _simulate_live_agg(project, _live_wx, _now_ts)
        else:
            _live_result = simulate_live_forecast(
                models, _live_wx, _now_ts,
                n_samples=200,
                features=features,
                country_code=project.country_code,
                mh_eval=mh_eval,
            )
    if _live_result is None:
        if is_multi_live:
            st.warning(
                "No substation has a weather-only model yet. "
                "Train batch weather-only models first (**Train all** → ☑ Weather-only).",
                icon="⚠️",
            )
        else:
            st.error("Live forecast failed — not enough weather data for this snapshot.")
        return

    _live_fc      = _live_result["forecast"]
    _live_lower   = _live_result["lower"]
    _live_upper   = _live_result["upper"]
    _live_weather = _live_result["weather"]

    # ── Log 1h forecast to SQLite (aggregate only, not multi-sum) ────────────
    if not is_multi_live and enable_log:
        try:
            log_run(project, _now_ts, {h: float(v) for h, v in zip(
                range(1, len(_live_fc) + 1), _live_fc.values
            )})
        except Exception:
            pass

    # ── Fetch proxy actuals from log ──────────────────────────────────────────
    _proxy_actuals = pd.Series(dtype=float)
    if not is_multi_live:
        _proxy_from = _now_ts - pd.Timedelta(hours=24)
        _proxy_actuals = get_proxy_actuals(project, _proxy_from, _now_ts)

    # ── KPIs ──────────────────────────────────────────────────────────────────
    _live_peak_mw = float(_live_fc.max()) * _du_factor
    _live_peak_ts = _live_fc.idxmax()
    _lk_mape, _lk_demand = st.columns([1, 2])
    _lk_mape.metric(
        "MAPE", "–",
        delta="Real-time demand datastream required",
        delta_color="off",
        help="MAPE requires actual demand data (SCADA connection not yet available).",
    )
    _lk_cur, _lk_peak = _lk_demand.columns(2)
    _lk_cur.metric(
        "Current demand", "–",
        help="Real-time demand datastream required — no SCADA connection.",
    )
    _lk_peak.metric(
        "48h peak forecast",
        f"{_live_peak_mw:.2f} {_display_unit}",
        help=_live_peak_ts.strftime("%a %d %b · %H:%M"),
    )

    # ── Chart ─────────────────────────────────────────────────────────────────
    _live_fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.54, 0.26, 0.16], vertical_spacing=0.07,
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": True}],
            [{"secondary_y": True}],
        ],
    )

    _live_fc_end = _live_fc.index[-1]
    for _sr in (1, 2, 3):
        _live_fig.add_vrect(
            x0=_now_ts.isoformat(), x1=_live_fc_end.isoformat(),
            fillcolor=FORECAST_SHADE, line_width=0, layer="below",
            row=_sr, col=1,
        )

    # Scale live forecast series for display
    _live_fc_d    = _live_fc * _du_factor
    _live_lower_d = _live_lower * _du_factor
    _live_upper_d = _live_upper * _du_factor

    if not _proxy_actuals.empty:
        _live_fig.add_trace(
            go.Scatter(
                x=_proxy_actuals.index, y=_proxy_actuals.values * _du_factor,
                name="Logged 1h forecasts (proxy)",
                mode="lines",
                line=dict(color=PRIMARY, width=2, dash="dot"),
                opacity=0.60,
                hovertemplate=f"%{{y:.2f}} {_display_unit}<extra>proxy actual</extra>",
            ),
            row=1, col=1,
        )

    _live_anchor_ts  = _now_ts
    _live_anchor_val = float(_live_fc_d.iloc[0])
    _live_upper_b = pd.concat([pd.Series([_live_anchor_val], index=[_live_anchor_ts]), _live_upper_d])
    _live_lower_b = pd.concat([pd.Series([_live_anchor_val], index=[_live_anchor_ts]), _live_lower_d])
    _live_fc_b    = pd.concat([pd.Series([_live_anchor_val], index=[_live_anchor_ts]), _live_fc_d])

    _lx_band = list(_live_upper_b.index) + list(_live_lower_b.index[::-1])
    _ly_band = list(_live_upper_b.values) + list(_live_lower_b.values[::-1])
    _live_fig.add_trace(
        go.Scatter(
            x=_lx_band, y=_ly_band,
            name="80% CI (Monte Carlo)",
            fill="toself", fillcolor=BAND_COL,
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
        ),
        row=1, col=1,
    )
    _live_fig.add_trace(
        go.Scatter(
            x=_live_fc_b.index, y=_live_fc_b.values,
            name="Forecast (median)",
            mode="lines", line=dict(color=FORECAST, width=2, dash="dash"),
            hovertemplate=f"%{{y:.2f}} {_display_unit}<extra></extra>",
        ),
        row=1, col=1,
    )
    _live_fig.add_shape(
        type="line",
        x0=_now_ts, x1=_now_ts, y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color=NOW_COL, width=2),
        layer="above",
    )

    _WX_COLS = ["temperature_c", "wind_speed_ms", "cloud_cover_pct", "humidity_pct"]
    _live_wx_avail = [c for c in _WX_COLS if _live_weather is not None and c in _live_weather.columns]
    if "temperature_c" in _live_wx_avail:
        _live_fig.add_trace(
            go.Scatter(
                x=_live_weather.index, y=_live_weather["temperature_c"],
                name="Temperature (°C)", mode="lines",
                line=dict(color=ACCENT, width=2),
                legend="legend2",
                hovertemplate="%{y:.1f} °C<extra></extra>",
            ),
            row=2, col=1, secondary_y=False,
        )
    if "wind_speed_ms" in _live_wx_avail:
        _live_fig.add_trace(
            go.Scatter(
                x=_live_weather.index, y=_live_weather["wind_speed_ms"],
                name="Wind speed (m/s)", mode="lines",
                line=dict(color=ACCENT2, width=1.5, dash="dot"),
                legend="legend2",
                hovertemplate="%{y:.1f} m/s<extra></extra>",
            ),
            row=2, col=1, secondary_y=True,
        )
    if "cloud_cover_pct" in _live_wx_avail:
        _live_fig.add_trace(
            go.Scatter(
                x=_live_weather.index, y=_live_weather["cloud_cover_pct"],
                name="Cloud cover (%)", mode="lines",
                line=dict(color=GRAY, width=1.5, dash="dash"),
                legend="legend3", opacity=0.70,
                hovertemplate="%{y:.0f}%<extra></extra>",
            ),
            row=3, col=1, secondary_y=False,
        )
    if "humidity_pct" in _live_wx_avail:
        _live_fig.add_trace(
            go.Scatter(
                x=_live_weather.index, y=_live_weather["humidity_pct"],
                name="Humidity (%)", mode="lines",
                line=dict(color=PURPLE, width=2.5),
                legend="legend3",
                hovertemplate="%{y:.0f}%<extra></extra>",
            ),
            row=3, col=1, secondary_y=True,
        )

    _live_first_ts  = _now_ts - pd.Timedelta(hours=24) if not _proxy_actuals.empty else _now_ts
    _live_last_ts   = _live_fc_end
    _live_span_h    = (_live_last_ts - _live_first_ts).total_seconds() / 3600
    _ldtick_ms, _ltick_angle, _ltop_margin, _lday_label_y = _chart_time_axis(_live_span_h)
    _live_all_days  = pd.date_range(
        start=_live_first_ts.normalize(), end=_live_last_ts.normalize(), freq="D"
    )
    for _lmidnight in _live_all_days[1:]:
        _live_fig.add_vline(
            x=_lmidnight.isoformat(), line_width=1,
            line_color=GRAY_MID, line_dash="dot", opacity=0.35,
        )
    for _lday in _live_all_days:
        _lseg_start = max(_lday, _live_first_ts)
        _lseg_end   = min(_lday + pd.Timedelta(days=1), _live_last_ts)
        _lmidpoint  = _lseg_start + (_lseg_end - _lseg_start) / 2
        _live_fig.add_annotation(
            x=_lmidpoint.isoformat(), y=_lday_label_y,
            xref="x", yref="paper",
            text=_lday.strftime("<b>%a %d %b</b>"),
            showarrow=False, font=dict(size=15, color=TEXT), xanchor="center",
        )

    _live_leg = dict(
        xanchor="left", yanchor="top", orientation="v",
        bgcolor="rgba(255,255,255,0.75)", borderwidth=0,
        font=dict(size=11, color=TEXT_MUT, family="Inter, sans-serif"),
    )
    _live_fig.update_layout(
        height=600,
        margin=dict(t=_ltop_margin, b=20, l=52, r=185),
        legend=dict(x=1.02, y=0.99, **_live_leg),
        legend2=dict(x=1.02, y=0.52, **_live_leg),
        legend3={**_live_leg, "x": 1.02, "y": 0.11, "yanchor": "middle"},
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
    )
    _live_fig.update_yaxes(
        title_text=f"Heat demand<br>{_display_unit}", row=1, col=1,
        gridcolor=GRID_COL, zeroline=False,
        title_font=dict(color=TEXT_MUT), tickfont=dict(color=TEXT_MUT),
    )
    if "temperature_c" in _live_wx_avail:
        _lt_min = float(_live_weather["temperature_c"].min()) - 5
        _lt_max = float(_live_weather["temperature_c"].max()) + 5
        _live_fig.update_yaxes(
            title_text="Temperature<br>°C", row=2, col=1, secondary_y=False,
            gridcolor=GRID_COL, zeroline=True, zerolinecolor=GRID_COL, zerolinewidth=1,
            range=[_lt_min, _lt_max],
            title_font=dict(color=ACCENT), tickfont=dict(color=ACCENT),
        )
    if "wind_speed_ms" in _live_wx_avail:
        _live_fig.update_yaxes(
            title_text="Wind<br>m/s", row=2, col=1, secondary_y=True,
            showgrid=False, zeroline=False,
            range=[0, max(float(_live_weather["wind_speed_ms"].max()) * 1.4, 8)],
            title_font=dict(color=ACCENT2), tickfont=dict(color=ACCENT2),
        )
    if "cloud_cover_pct" in _live_wx_avail:
        _live_fig.update_yaxes(
            title_text="Cloud<br>%", row=3, col=1, secondary_y=False,
            gridcolor=GRID_COL, zeroline=False, range=[0, 100], dtick=25, fixedrange=True,
            title_font=dict(color=TEXT_MUT), tickfont=dict(color=TEXT_MUT),
        )
    if "humidity_pct" in _live_wx_avail:
        _lh_min = max(0, float(_live_weather["humidity_pct"].min()) - 5)
        _lh_max = min(100, float(_live_weather["humidity_pct"].max()) + 5)
        _live_fig.update_yaxes(
            title_text="Humidity<br>%", row=3, col=1, secondary_y=True,
            showgrid=False, zeroline=False, range=[_lh_min, _lh_max],
            title_font=dict(color=PURPLE), tickfont=dict(color=PURPLE),
        )
    _live_first_midnight = _live_first_ts.normalize().isoformat()
    _live_fig.update_xaxes(
        showgrid=False, row=1, col=1,
        side="top", tickangle=_ltick_angle, showticklabels=True,
        tick0=_live_first_midnight, dtick=_ldtick_ms, tickformat="%H:%M",
        tickfont=dict(size=12, color=TEXT_MUT), tickcolor=GRAY_LITE,
        ticks="outside", ticklen=4,
    )
    _live_fig.update_xaxes(showgrid=False, showticklabels=False, row=2, col=1)
    _live_fig.update_xaxes(showgrid=False, showticklabels=False, row=3, col=1)

    st.plotly_chart(_live_fig, use_container_width=True)

    # ── Greyed-out sections ───────────────────────────────────────────────────
    _LOCK_STYLE = (
        "background:#f1f3f7; border-radius:8px; padding:12px 16px; "
        "color:#9CA3AF; font-size:0.9em;"
    )
    _lock_icon = "🔒"
    st.markdown("---")
    _lg1, _lg2 = st.columns(2)
    with _lg1:
        st.markdown(
            f'<div style="{_LOCK_STYLE}">'
            f'{_lock_icon} <b>Actual demand history</b><br>'
            "Real-time demand datastream required — no SCADA connection.<br>"
            "Connect a live demand API to populate this section."
            "</div>",
            unsafe_allow_html=True,
        )
    with _lg2:
        st.markdown(
            f'<div style="{_LOCK_STYLE}">'
            f'{_lock_icon} <b>MAPE / forecast accuracy</b><br>'
            "Real-time demand datastream required — cannot compute MAPE without actual demand.<br>"
            "Historical model MAPE: "
            + (f"{round(mh_eval['mape_pct'].mean(), 1)}% (from test set)" if mh_eval is not None else "see Historic tab")
            + "</div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        f'<div style="{_LOCK_STYLE}; margin-top:8px;">'
        f'{_lock_icon} <b>Demand lags &amp; rolling demand features</b><br>'
        "Real-time demand datastream required — demand_lag_0h … demand_lag_336h, "
        "demand_roll_24h, demand_roll_168h require live SCADA data.<br>"
        "The live forecaster uses <b>weather + calendar features only</b>."
        "</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Live Forecaster · {project.name} · Weather: Open-Meteo NWP · "
        f"Model: LightGBM weather-only · Snapshot: {_now_ts.strftime('%Y-%m-%d %H:%M')} UTC"
    )


def _render_demand_chart(fig, *, show_weather: bool, slot: str, key: str | None = None) -> None:
    """Render a demand chart inside a min-height 'slot'.

    A rerun (full or fragment) momentarily unmounts the Plotly component; without
    a reserved height the chart's container collapses to 0px, the content below
    jumps up by one chart-height, and the browser reports that as a scroll-to-top.

    Reserving a min-height on the surrounding vertical block keeps the slot at the
    chart's height even while the component is briefly gone, so nothing shifts and
    the scroll position is naturally preserved — no JS, no scroll locking. We use
    min-height (not a fixed height) so the weather-expanded chart can grow without
    an inner scrollbar. The marker span lets a CSS :has() rule target this block.
    """
    _min_h = (600 if show_weather else SUB_CHART_HEIGHT) + 8
    _cls = f"chart-slot-{slot}"
    st.markdown(
        f"<style>div[data-testid='stVerticalBlock']:has(> div span.{_cls})"
        f"{{min-height:{_min_h}px;}}</style>",
        unsafe_allow_html=True,
    )
    with st.container():
        st.markdown(f"<span class='{_cls}'></span>", unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, key=key)


def _render_chart_controls(*, weather_key: str, weather_default: bool, weather_help: str) -> None:
    """Render the 'Show weather' toggle and the 24/48h horizon toggle side by side,
    placed BELOW the chart. Both persist to session_state and are read at the top
    of the render pass (before the forecast/figure are built). The horizon uses the
    shared global key, so it stays unified across views even though it's shown here."""
    _cw, _ch, _ = st.columns([1.3, 1.6, 4])
    with _cw:
        st.toggle("Show weather", value=weather_default, key=weather_key, help=weather_help)
    with _ch:
        st.radio(
            "Forecast horizon",
            options=[24, 48],
            horizontal=True,
            format_func=lambda h: f"{h}h",
            key="forecast_horizon",
            help="Forecast length — applies to every chart and tile.",
        )


def _render_hist_tab(project, df_raw, mh_models, mh_features, mh_eval, mc_samples, use_wx_forecast, snapshot_dt=None):
    """Historic demand forecast analysis tab content."""

    # ── Snapshot picker (only when not already provided by the caller) ─────────
    if snapshot_dt is None:
        snapshot_dt = _render_snapshot_picker(df_raw, project, key_prefix="")

    horizon = int(st.session_state.get("forecast_horizon", 48))

    # ── Fetch weather forecast (if enabled) ──────────────────────────────────────
    # Open-Meteo's historical-forecast archive only covers ~2022 onward. For older
    # snapshots (e.g. Aalborg 2020) — or if the API can't be reached / returns
    # partial data — we fall back to observed weather. simulate_forecast performs
    # the per-cell fallback, so the forecast never crashes regardless of coverage.
    FORECAST_ARCHIVE_YEAR = 2022
    wx_forecast = None
    if use_wx_forecast:
        if snapshot_dt.year < FORECAST_ARCHIVE_YEAR:
            st.info(
                "Live forecast weather isn't available before ~2022 for this period, "
                "so observed weather (with NWP-like noise) is used for this snapshot.",
                icon="ℹ️",
            )
        else:
            with st.spinner("Fetching weather forecast from Open-Meteo…"):
                wx_forecast = fetch_open_meteo_forecast(
                    snapshot_dt, horizon=horizon,
                    lat=project.lat, lon=project.lon,
                    history_hours=24,
                )
            if wx_forecast is None:
                st.info(
                    "Forecast weather is unavailable for this snapshot — falling back to "
                    "observed weather (with NWP-like noise). This is expected before "
                    "~2022 or if Open-Meteo can't be reached.",
                    icon="ℹ️",
                )
            else:
                # Detect partial coverage over the forecast window; any missing cell is
                # transparently backfilled with observed weather inside simulate_forecast.
                _fc_index = pd.DatetimeIndex(
                    [snapshot_dt + pd.Timedelta(hours=h) for h in range(1, horizon + 1)]
                )
                _wx_cols = [c for c in MH_WEATHER_COLS if c in wx_forecast.columns]
                _missing_cols = len(_wx_cols) < len(MH_WEATHER_COLS)
                _has_gaps = False
                if _wx_cols:
                    _aligned = wx_forecast.reindex(_fc_index)[_wx_cols]
                    _has_gaps = bool(_aligned.isna().any().any())
                if _missing_cols or _has_gaps:
                    st.caption(
                        "Some forecast-weather values are unavailable for this window; "
                        "observed weather is used to fill the gaps."
                    )

    # ── Conformal / Monte Carlo forecast ─────────────────────────────────────
    if project.is_multi():
        # Aggregate forecast = sum of all trained substation forecasts.
        # No separate aggregate model is needed or trained.
        with st.spinner("Running aggregate forecast (summing all substations)…"):
            result = _simulate_aggregate_from_substations(
                df_raw, project, snapshot_dt, horizon, wx_forecast, mc_samples,
            )
    else:
        horizon_models = {h: m for h, m in mh_models.items() if h <= horizon}
        with st.spinner("Running forecast…"):
            result = simulate_forecast(
                df_raw, horizon_models, snapshot_dt,
                n_samples=mc_samples,
                wx_forecast=wx_forecast,
                history_hours=24,
                features=mh_features,
                mh_eval=mh_eval,
                target=project.target_column,
            )

    if result is None:
        st.error(
            "Not enough data for this snapshot — choose a different date/time "
            "in the snapshot selector above."
        )
        st.stop()

    history       = result["history"]
    actual_future = result["actual_future"]
    forecast_base = result["forecast"].copy()
    lower_base    = result["lower"].copy()
    upper_base    = result["upper"].copy()
    weather       = result["weather"]

    if OVERRIDES_KEY not in st.session_state:
        st.session_state[OVERRIDES_KEY] = []

    forecast, lower, upper = apply_forecast_overrides(
        forecast_base, lower_base, upper_base, st.session_state[OVERRIDES_KEY]
    )
    has_overrides = bool(st.session_state[OVERRIDES_KEY])

    # ── Run context (for sidebar footer) ──────────────────────────────────────────
    model_label = f"48-model ensemble (h=1…{horizon})"
    wx_label = (
        "weather: Open-Meteo NWP forecast ✓" if wx_forecast is not None
        else "weather: observed + synthetic noise"
    )

    # ── KPIs ──────────────────────────────────────────────────────────────────────
    # MAPE is unit-agnostic — compute on raw (native-unit) series.
    window_mape = compute_mape(actual_future, forecast_base)
    # Reference MAPE: for multi-mode compute aggregate (Σ MAE / Σ mean demand);
    # for single-mode use the model's test-set eval, falling back to TEST_MAPE.
    if project.is_multi():
        ref_mape = _aggregate_ref_mape(project, df_raw) or TEST_MAPE
    else:
        ref_mape = round(mh_eval["mape_pct"].mean(), 1) if mh_eval is not None else TEST_MAPE
    mape_status, _ = _mape_period_status(window_mape, ref_mape)
    # Apply display scale for absolute values shown in KPIs and charts.
    current_mw  = float(history.iloc[-1]) * _du_factor
    peak_mw     = float(forecast.max()) * _du_factor
    peak_ts     = forecast.idxmax()

    k_mape, k_demand = st.columns([1, 2])
    _mape_delta_color = (
        "normal" if mape_status == "On target"
        else "off" if mape_status == "Acceptable"
        else "inverse"
    )
    _ref_label = "aggregate test MAPE" if project.is_multi() else "model avg"
    k_mape.metric(
        "MAPE of period displayed",
        f"{window_mape:.1f}%",
        delta=f"{mape_status} · {_ref_label} {ref_mape:.1f}%",
        delta_color=_mape_delta_color,
        help=(
            "Mean absolute percentage error for the displayed forecast window.\n\n"
            "Reference is the aggregate MAPE over the full test period: "
            "Σ substation MAEs / Σ substation mean demands × 100."
            if project.is_multi() else
            "Mean absolute percentage error for the forecast window in the chart."
        ),
    )
    k_cur, k_peak = k_demand.columns(2)
    k_cur.metric("Current demand", f"{current_mw:.2f} {_display_unit}")
    k_peak.metric(
        f"{horizon}h peak forecast",
        f"{peak_mw:.2f} {_display_unit}",
        help=peak_ts.strftime("%a %d %b · %H:%M")
        + (" · includes manual overrides" if has_overrides else ""),
    )

    # ── Demand + Weather chart (shared builder; weather collapsible) ──────────────
    # Weather state is read here (before the figure) but its toggle is rendered
    # BELOW the chart, next to the horizon toggle (see _render_chart_controls).
    _agg_show_wx = st.session_state.get("hist_show_weather", True)
    # Scale demand series to display unit before passing to the chart builder.
    _d_history = history * _du_factor
    _d_forecast = forecast * _du_factor
    _d_lower = lower * _du_factor
    _d_upper = upper * _du_factor
    _d_actual_future = actual_future * _du_factor if actual_future is not None and len(actual_future) > 0 else actual_future
    _d_forecast_base = forecast_base * _du_factor
    fig = build_demand_figure(
        _d_history, _d_forecast, _d_lower, _d_upper, _d_actual_future,
        snapshot_dt, _d_forecast_base,
        overrides=st.session_state.get(OVERRIDES_KEY, []),
        has_overrides=has_overrides,
        demand_unit=_display_unit,
        show_weather=_agg_show_wx,
        weather=weather,
        df_raw=df_raw,
    )
    _render_demand_chart(fig, show_weather=_agg_show_wx, slot="agg")
    _render_chart_controls(
        weather_key="hist_show_weather", weather_default=True,
        weather_help="Show the temperature/wind and cloud/humidity rows below the demand chart.",
    )

    _manual_override(forecast_base, df_raw.index.max())

    _render_sidebar_footer(
        snapshot_dt=snapshot_dt,
        horizon=horizon,
        model_label=model_label,
        wx_label=wx_label,
        city=project.name,
        data_start=df_raw.index.min(),
        data_end=df_raw.index.max(),
        test_start_ts=pd.Timestamp(project.test_start or DEFAULT_TEST_START),
    )

    # ── Footer ────────────────────────────────────────────────────────────────────
    st.caption(
        f"Data: {project.name} district heating network 2020–2024 · "
        "Source: [Zenodo 17177421](https://zenodo.org/records/17177421) · "
        "Model: LightGBM · Confidence bands: Monte Carlo with NWP σ(h) noise"
    )


# ── Substation Detail tab (multi-mode projects only) ─────────────────────────

def _select_substation(sub_id: str) -> None:
    """on_click callback for substation tiles.

    Using a callback (instead of checking the button return value and calling
    st.rerun) sets the selection BEFORE the fragment re-executes, so the chart —
    rendered above the tiles — reflects the new substation in the same
    fragment-scoped rerun.  This gives reliable single-click switching with no
    full-app reload.
    """
    st.session_state["sub_selected"] = sub_id


@st.fragment
def _render_substation_tab(project: "Project", df_raw: "pd.DataFrame") -> None:
    """Render the Substation Detail tab for multi-mode projects.

    Layout: snapshot picker → KPIs → demand chart (same style as aggregate tab,
    demand row only) → manual override → substation selector grid.
    """
    from src.model import simulate_forecast as _sim_fc

    substations = project.substations
    if not substations:
        st.info("No substations configured for this project.")
        return

    sel_id   = st.session_state["sub_selected"]
    sel_sub  = next((s for s in substations if s["id"] == sel_id), substations[0])
    sel_col  = sel_sub.get("column", sel_id)

    # ── Snapshot, live mode and horizon all come from the GLOBAL controls ────
    # rendered above the tab bar, so every chart and tile shares one timing.
    sub_snapshot_dt = st.session_state.get("_g_snapshot_dt")
    _sub_live_mode  = st.session_state.get("_g_live_mode", False)
    horizon = int(st.session_state.get("forecast_horizon", 48))

    if _sub_live_mode:
        _slhas_model = project.substation_default_pkl_dir(sel_id).is_dir()
        _slmodels, _slfeatures, _sleval = (
            _load_sub_models(project, sel_id) if _slhas_model else ({}, [], None)
        )
        if not _slmodels:
            st.info("No model trained yet — use **🔁 Train** in the sidebar.", icon="ℹ️")
        elif not _model_is_weather_only(_slfeatures):
            st.warning(
                "No weather-only model found for this substation. "
                "Train with the **Weather-only** option to enable live forecasting.",
                icon="⚠️",
            )
        else:
            # STABLE key_suffix ("sub", not per-sel_id) — same reasoning as the
            # weather toggle / override widgets: reuse the live widgets across
            # substations so switching never destroys + recreates them.
            _render_live_agg_section(
                project, _slmodels, _slfeatures, _sleval,
                key_suffix="sub",
                enable_log=False,
            )

    else:
        # ── 1. Historic substation forecast ───────────────────────────────────

        # Per-substation overrides key (isolated from aggregate)
        _sub_ov_key = f"sub_overrides_{sel_id}"
        if _sub_ov_key not in st.session_state:
            st.session_state[_sub_ov_key] = []

        # ── 2. Run forecast ───────────────────────────────────────────────────
        st.markdown(
            f'<p style="font-size:1.05em;font-weight:700;margin:8px 0 2px 0;">'
            f'{sel_sub["name"]} — demand forecast</p>',
            unsafe_allow_html=True,
        )

        if sel_col not in df_raw.columns:
            st.warning(f"Column `{sel_col}` not found in the dataset.")
            _sub_result = None
        else:
            has_model = project.substation_default_pkl_dir(sel_id).is_dir()
            sub_models, sub_features, sub_eval = _load_sub_models(project, sel_id) if has_model else ({}, [], None)

            _sub_result = None
            if sub_models:
                _horizon_sub = {h: m for h, m in sub_models.items() if h <= horizon}
                with st.spinner("Running substation forecast…"):
                    _sub_result = _sim_fc(
                        df_raw, _horizon_sub, sub_snapshot_dt,
                        features=sub_features, mh_eval=sub_eval,
                        history_hours=24,   # match the aggregate tab (default is 48)
                        target=sel_col,
                    )
            if not has_model:
                st.info("No model trained yet — use **🔁 Train** in the sidebar.", icon="ℹ️")
            elif _sub_result is None:
                st.warning("Not enough data for this snapshot — try a different date.")

        if _sub_result is not None:
            _history      = _sub_result["history"]
            _actual_future = _sub_result["actual_future"]
            _forecast_base = _sub_result["forecast"].copy()
            _lower_base    = _sub_result["lower"].copy()
            _upper_base    = _sub_result["upper"].copy()

            _sub_ov = st.session_state[_sub_ov_key]
            _forecast, _lower, _upper = apply_forecast_overrides(
                _forecast_base, _lower_base, _upper_base, _sub_ov
            )
            _has_ov = bool(_sub_ov)

            # ── KPIs (identical to aggregate tab) ──────────────────────────────
            # MAPE on native-unit series; display scale for absolute KPI values.
            _win_mape = compute_mape(_actual_future, _forecast_base)
            _ref_mape = round(sub_eval["mape_pct"].mean(), 1) if sub_eval is not None else None
            _cur_mw   = float(_history.iloc[-1]) * _du_factor
            _peak_mw  = float(_forecast.max()) * _du_factor
            _peak_ts  = _forecast.idxmax()

            k_mape, k_demand = st.columns([1, 2])
            if _ref_mape is not None:
                _mape_status, _ = _mape_period_status(_win_mape, _ref_mape)
                _mape_delta_color = (
                    "normal" if _mape_status == "On target"
                    else "off" if _mape_status == "Acceptable"
                    else "inverse"
                )
                k_mape.metric(
                    "MAPE of period displayed",
                    f"{_win_mape:.1f}%",
                    delta=f"{_mape_status} · model avg {_ref_mape:.1f}%",
                    delta_color=_mape_delta_color,
                )
            else:
                k_mape.metric("MAPE (window)", f"{_win_mape:.1f}%" if _win_mape is not None else "—")
            k_cur, k_peak = k_demand.columns(2)
            k_cur.metric("Current demand", f"{_cur_mw:.2f} {_display_unit}")
            k_peak.metric(
                f"{horizon}h peak forecast",
                f"{_peak_mw:.2f} {_display_unit}",
                help=_peak_ts.strftime("%a %d %b · %H:%M")
                + (" · includes manual overrides" if _has_ov else ""),
            )

            # ── Demand chart — identical shared builder; weather collapsible ────
            # Weather state is read here (before the figure); its toggle + the
            # horizon toggle are rendered BELOW the chart (see _render_chart_controls).
            _sub_show_wx = st.session_state.get("sub_show_weather", False)
            _sd_history = _history * _du_factor
            _sd_forecast = _forecast * _du_factor
            _sd_lower = _lower * _du_factor
            _sd_upper = _upper * _du_factor
            _sd_actual_future = _actual_future * _du_factor if _actual_future is not None and len(_actual_future) > 0 else _actual_future
            _sd_forecast_base = _forecast_base * _du_factor
            _fig = build_demand_figure(
                _sd_history, _sd_forecast, _sd_lower, _sd_upper, _sd_actual_future,
                sub_snapshot_dt, _sd_forecast_base,
                overrides=_sub_ov,
                has_overrides=_has_ov,
                demand_unit=_display_unit,
                show_weather=_sub_show_wx,
                weather=_sub_result["weather"],
                df_raw=df_raw,
            )
            # STABLE chart key + reserved min-height slot so switching substations
            # (a fragment rerun that unmounts the Plotly component) can't collapse
            # the chart container and yank the page scroll.
            _render_demand_chart(
                _fig, show_weather=_sub_show_wx, slot="sub", key="sub_demand_chart",
            )
            _render_chart_controls(
                weather_key="sub_show_weather", weather_default=False,
                weather_help="Show the shared weather rows below the demand chart "
                             "(same single weather series used across all substations).",
            )

            # ── Manual override ────────────────────────────────────────────────
            # Call the UNWRAPPED function inline so it becomes part of this
            # fragment (no nested fragment). rerun_scope="fragment" means Add/Remove
            # reruns only the substation fragment — redrawing this chart and the DO
            # badge — without a full app rerun / aggregate recompute.
            #
            # STABLE widget key_prefix ("sub_") — NOT per-sel_id. The *saved*
            # overrides stay isolated per substation via overrides_key
            # (sub_overrides_{sel_id}); only the input widgets are shared, so the
            # widget tree stays structurally identical across substation switches
            # and the fragment updates in place instead of remounting/scrolling.
            _render_manual_override(
                _forecast_base,
                df_raw.index.max(),
                key_prefix="sub_",
                overrides_key=_sub_ov_key,
                rerun_scope="fragment",
            )

    # ── 3. Substation selector grid ───────────────────────────────────────────
    st.divider()
    _lgd_left, _lgd_right = st.columns([3, 5])
    _lgd_left.markdown(
        f'<span style="font-size:0.78em;font-weight:600;text-transform:uppercase;'
        f'letter-spacing:0.07em;color:{GRAY};">Select substation</span>',
        unsafe_allow_html=True,
    )
    _lgd_right.markdown(
        '<span style="display:inline-flex;gap:6px;align-items:center;flex-wrap:wrap;font-size:0.76em;">'
        f'<span style="background:rgba(22,163,74,0.18);color:#16a34a;'
        f'padding:1px 7px;border-radius:4px;border:1px solid rgba(22,163,74,0.35);">&lt;8% MAPE</span>'
        f'<span style="background:rgba(202,138,4,0.18);color:#ca8a04;'
        f'padding:1px 7px;border-radius:4px;border:1px solid rgba(202,138,4,0.35);">8–15%</span>'
        f'<span style="background:rgba(220,38,38,0.18);color:#dc2626;'
        f'padding:1px 7px;border-radius:4px;border:1px solid rgba(220,38,38,0.35);">&gt;15%</span>'
        f'<span style="background:rgba(156,163,175,0.15);color:#6b7280;'
        f'padding:1px 7px;border-radius:4px;border:1px solid #e5e7eb;">no model</span>'
        f'&nbsp;·&nbsp;'
        f'<span style="background:#7f1d1d;color:white;padding:1px 7px;'
        f'border-radius:4px;font-weight:700;letter-spacing:0.04em;">DO</span>'
        f'<span style="color:{GRAY};">= demand override active</span>'
        f'&nbsp;·&nbsp;'
        f'<span style="color:{GRAY};">tile colour = period MAPE · <em>m:</em> = model test MAPE</span>'
        '</span>',
        unsafe_allow_html=True,
    )

    # One global CSS reset for all tile buttons (makes `\n` render as a line-break
    # and strips the default Streamlit button chrome so our per-tile styles show).
    st.markdown(
        """<style>
        /* Allow newlines in button labels to render as visual line breaks */
        div[data-testid="stButton"] > button p {
            white-space: pre-line !important;
            text-align: center !important;
            margin: 0 !important;
        }
        </style>""",
        unsafe_allow_html=True,
    )

    # Scale cols per row by dataset size so tiles stay compact for large networks.
    COLS_PER_ROW = min(10, max(6, len(substations) // 9 + 6))
    _all_tile_css: list[str] = []
    for _row_subs in [substations[i:i + COLS_PER_ROW] for i in range(0, len(substations), COLS_PER_ROW)]:
        _row_cols = st.columns(len(_row_subs))
        for _col_widget, _sub in zip(_row_cols, _row_subs):
            _sub_col  = _sub.get("column", _sub["id"])

            # Window MAPE (current period, snapshot-aware) → drives tile colour
            _win_mape = _sub_window_mape(
                project, df_raw, _sub["id"], _sub_col, sub_snapshot_dt, horizon,
            )
            # Model MAPE (test-set average from registry)
            _mdl_mape = _sub_mape(project, _sub["id"])

            # Colour by window MAPE; fall back to model MAPE if window unavailable
            _colour_mape = _win_mape if _win_mape is not None else _mdl_mape
            _bg, _label_col = _mape_tile_style(_colour_mape)
            _selected = st.session_state["sub_selected"] == _sub["id"]

            # Demand + trend at snapshot time
            _trend = "→"
            _cur = "—"
            if _sub_col in df_raw.columns:
                _upto = df_raw[_sub_col].loc[:sub_snapshot_dt].dropna()
                if not _upto.empty:
                    _cur = f"{float(_upto.iloc[-1]) * _du_factor:.2f}"
                _recent = _upto.tail(3)
                if len(_recent) >= 2:
                    _d = float(_recent.iloc[-1]) - float(_recent.iloc[-2])
                    _m = float(abs(_recent.mean()) or 1)
                    _trend = "↑" if _d > 0.05 * _m else ("↓" if _d < -0.05 * _m else "→")

            _win_str = f"{_win_mape:.0f}%" if _win_mape is not None else "—"
            if _mdl_mape is None:
                _mdl_css_color = GRAY_MID
                _mdl_str = "m:—"
            elif _mdl_mape < 8:
                _mdl_css_color = "#2e7d32"   # green
                _mdl_str = f"m:{_mdl_mape:.0f}%"
            elif _mdl_mape < 15:
                _mdl_css_color = "#e65100"   # orange
                _mdl_str = f"m:{_mdl_mape:.0f}%"
            else:
                _mdl_css_color = "#c62828"   # red
                _mdl_str = f"m:{_mdl_mape:.0f}%"
            _has_sub_ov = bool(st.session_state.get(f"sub_overrides_{_sub['id']}"))

            _mc = "sub-tile-" + re.sub(r"[^a-zA-Z0-9]", "-", _sub["id"])
            _sel_extra = (
                f"filter:brightness(0.90)!important;"
                f"outline:2.5px solid {NAVY}!important;outline-offset:-2px!important;"
                if _selected else ""
            )
            _do_after = (
                f"""div[data-testid='stColumn']>div[data-testid='stVerticalBlock']:has(span.{_mc})
                    div[data-testid='stButton']>button::after{{
                        content:"DO";position:absolute;top:3px;right:4px;
                        background:{NAVY};color:white;
                        font-size:8px;font-weight:700;
                        padding:1px 4px;border-radius:3px;line-height:1.4;
                    }}"""
                if _has_sub_ov else ""
            )
            _tile_rules = f"""
            div[data-testid='stColumn']>div[data-testid='stVerticalBlock']:has(span.{_mc})
                div[data-testid='stButton']>button{{
                    background:{_bg}!important;
                    border:1px solid rgba(17,34,77,0.15)!important;
                    border-radius:6px!important;
                    padding:4px 5px!important;
                    min-height:52px!important;
                    width:100%!important;
                    font-family:Inter,sans-serif!important;
                    font-size:10px!important;
                    color:{_label_col}!important;
                    text-align:left!important;
                    position:relative!important;
                    line-height:1.35!important;
                    {_sel_extra}
                }}
            div[data-testid='stColumn']>div[data-testid='stVerticalBlock']:has(span.{_mc})
                div[data-testid='stButton']>button p{{color:{_label_col}!important;text-align:left!important;}}
            div[data-testid='stColumn']>div[data-testid='stVerticalBlock']:has(span.{_mc})
                div[data-testid='stButton']>button p::first-line{{
                    font-weight:700!important;color:{TEXT}!important;
                }}
            div[data-testid='stColumn']>div[data-testid='stVerticalBlock']:has(span.{_mc})
                div[data-testid='stButton']>button:hover{{
                    filter:brightness(0.93)!important;
                    box-shadow:0 2px 6px rgba(0,0,0,0.10)!important;
                }}
            {_do_after}
            div[data-testid='stColumn']>div[data-testid='stVerticalBlock']:has(span.{_mc})
                div[data-testid='stButton']>button::before{{
                    content:"{_mdl_str}";position:absolute;bottom:3px;right:4px;
                    background:{_mdl_css_color};color:white;
                    font-size:9px;font-weight:400;line-height:1.4;
                    padding:1px 4px;border-radius:3px;
                }}"""
            _all_tile_css.append(_tile_rules)

            _col_widget.markdown(f'<span class="{_mc}"></span>', unsafe_allow_html=True)

            # Line 1: substation name + trend arrow
            # Line 2: current demand
            # Line 3: window MAPE (period) + model MAPE reference
            _label = f"{_sub['name']}\n{_cur} {_display_unit} {_trend}\n{_win_str}"
            _col_widget.button(
                _label,
                key=f"sub_btn_{_sub['id']}",
                use_container_width=True,
                type="secondary",
                on_click=_select_substation,
                args=(_sub["id"],),
            )

    st.markdown(f"<style>{''.join(_all_tile_css)}</style>", unsafe_allow_html=True)


# ── Render substation tab ─────────────────────────────────────────────────────
# (_render_substation_tab is wrapped in @st.fragment so tile selection triggers a
# fragment-scoped rerun — instant chart swap, no full-app reload or scroll-to-top.
# The sidebar training controls are decoupled via the independent "sub_train"
# selector, so they never depend on the viewed substation.)


@st.fragment
def _render_workspace_inner(project, df_raw, mh_models, mh_features, mh_eval, mc_samples, use_wx_forecast):
    """Global controls + view selector + active view content.

    This is a *nested* fragment (called from the thin ``_render_workspace`` shell
    fragment). That matters for scrolling: Streamlit scrolls a TOP-LEVEL fragment's
    anchor into view on every rerun (yanking the page to the top), but leaves a
    NESTED fragment's scroll position untouched. Rerunning here on a snapshot /
    horizon / view change therefore preserves the scroll position, and the
    reserved-height chart slots keep the layout from shifting. The substation view
    keeps its own further-nested fragment for snappy single-tile switches.
    """
    # ── Global snapshot selector (drives every chart and tile) ───────────────
    # The 24/48h horizon toggle now lives BELOW each chart, next to the
    # "Show weather" toggle (see _render_chart_controls), but stays unified via
    # the shared "forecast_horizon" session key.
    _g_snapshot_dt, _g_live_mode = _render_snapshot_picker(
        df_raw, project, key_prefix="", live_key="live_mode",
    )
    st.session_state["_g_snapshot_dt"] = _g_snapshot_dt
    st.session_state["_g_live_mode"]   = _g_live_mode
    st.divider()

    # ── View selector ────────────────────────────────────────────────────────
    # A segmented control (NOT st.tabs): st.tabs remounts its whole panel — and
    # scrolls it into view — every time this fragment reruns, which yanks the page
    # to the top whenever the global snapshot/horizon changes. A segmented control
    # is a lightweight widget that doesn't remount the content panel, so combined
    # with the reserved-height chart slots the scroll position is preserved on
    # every snapshot/horizon change.
    _AGG_VIEW, _SUB_VIEW = "Aggregate demand forecast", "Substation Detail"
    if project.is_multi():
        st.markdown(
            f"""<style>
            button[data-testid="stBaseButton-segmented_control"],
            button[data-testid="stBaseButton-segmented_controlActive"] {{
                border-radius: 8px !important;
                font-weight: 600 !important;
                padding: 4px 16px !important;
            }}
            button[data-testid="stBaseButton-segmented_controlActive"] {{
                background-color: {NAVY} !important;
                color: #ffffff !important;
                border-color: {NAVY} !important;
            }}
            button[data-testid="stBaseButton-segmented_controlActive"] p {{
                color: #ffffff !important;
            }}
            </style>""",
            unsafe_allow_html=True,
        )
        _view = st.segmented_control(
            "View",
            options=[_AGG_VIEW, _SUB_VIEW],
            default=st.session_state.get("_view_sel", _AGG_VIEW),
            key="_view_sel",
            label_visibility="collapsed",
        ) or _AGG_VIEW
    else:
        _view = _AGG_VIEW

    if _view == _SUB_VIEW and project.is_multi():
        _render_substation_tab(project, df_raw)
    else:
        if _g_live_mode:
            _live_models_arg   = None if project.is_multi() else mh_models
            _live_features_arg = None if project.is_multi() else mh_features
            _live_eval_arg     = None if project.is_multi() else mh_eval
            _render_live_agg_section(
                project, _live_models_arg, _live_features_arg, _live_eval_arg,
                key_suffix="hist",
            )
        else:
            _render_hist_tab(
                project, df_raw, mh_models, mh_features, mh_eval, mc_samples,
                use_wx_forecast, snapshot_dt=_g_snapshot_dt,
            )


@st.fragment
def _render_workspace(project, df_raw, mh_models, mh_features, mh_eval, mc_samples, use_wx_forecast):
    """Thin top-level shell fragment.

    It holds NO widgets of its own, so it never reruns on a control change — only
    the nested ``_render_workspace_inner`` does. Keeping the real content in a
    nested fragment is what preserves the scroll position on every rerun.
    """
    _render_workspace_inner(
        project, df_raw, mh_models, mh_features, mh_eval, mc_samples, use_wx_forecast,
    )


_render_workspace(project, df_raw, mh_models, mh_features, mh_eval, mc_samples, use_wx_forecast)

