# Architecture — District Heating Demand Forecaster

## Overview

This project forecasts hourly district heating demand for the Flensburg network (2020–2024) using a LightGBM ensemble of 48 direct multi-horizon models. A Streamlit dashboard provides interactive time-travel inspection of forecast quality, with live weather integration via Open-Meteo.

**Data source:** [Zenodo 17177421](https://zenodo.org/records/17177421)  
**Target variable:** `heat_demand_mw`  
**Forecast horizons:** h = 1 … 48 hours ahead  
**Test set:** 2024 hold-out (training: 2020–2023)

---

## File & Module Structure

```
district-heating-forecaster/
├── app.py                           # Streamlit dashboard entry point
├── train_multi_horizon.py           # CLI: train and persist 48 LightGBM models
├── requirements.txt                 # Python dependencies (unpinned)
├── .cursorrules                     # Agent behaviour rules
├── ARCHITECTURE.md                  # This file
├── data/
│   ├── raw/                         # gitignored — Zenodo Excel source
│   │   └── flensburg_heat_network_2020_2024.xlsx
│   └── processed/
│       └── demand_with_weather.csv  # 43 843 hourly rows · 22 columns · 5.9 MB
├── models/
│   └── multi_horizon/
│       ├── meta.json                # feature lists + per-horizon eval (committed)
│       └── lgbm_h01.pkl … h48.pkl  # gitignored — 48 pickled LGBMRegressors
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_weather_features.ipynb    # builds demand_with_weather.csv
│   └── archive/                     # dev artifacts — superseded
│       ├── 03_baseline_model.ipynb
│       ├── 04_hyperparameter_tuning.ipynb  # BEST_PARAMS baked into src/model.py
│       └── 05_prophet_model.ipynb
└── src/
    ├── __init__.py
    ├── data.py                      # load_raw, build_horizon_dataset, all column/path constants
    ├── weather.py                   # fetch_open_meteo_forecast, lat/lon constants
    └── model.py                     # LightGBM training, eval, persistence, inference
```

### Module responsibilities

| Module | Responsibility |
|---|---|
| `src/project.py` | `Project` abstraction + `PROJECTS_DIR`/`REPO_ROOT`; resolves per-project paths, config and coordinates (`Project.load`, `Project.list_all`, `Project.default_id`) |
| `src/data.py` | `TARGET`, `_DATA_PATH`, `MH_MODELS_DIR` (default to the Flensburg project), all `MH_*` column lists; `load_raw(path, target)`, `build_horizon_dataset` |
| `src/weather.py` | `FLENSBURG_LAT/LON` defaults; `fetch_open_meteo_forecast(lat, lon, …)` (Open-Meteo HTTP calls + derived feature computation) |
| `src/model.py` | `BEST_PARAMS`, `TEST_*` fallback metrics; `compute_mape`, `train_multi_horizon`, `eval_multi_horizon`, `save/load_multi_horizon_models`, `load_mh_eval`, `get_mh_forecast_window` |
| `src/model_registry.py` | `ModelRegistry(registry_path, default_meta_path, default_pkl_dir)` — project-scoped JSON registry of trained versions |

### Dependency graph (horizontal flow)

```
Excel → nb01 → nb02 → CSV → train_multi_horizon.py → PKLs + meta.json
                       CSV → src/data.py ─┐
                                          ├── src/model.py → app.py
                       Open-Meteo API ────┘ (via src/weather.py, optional)
```

---

## Multi-project layer (Phase A)

The app was originally hardcoded to a single city (Flensburg). Phase A introduced
a **multi-project layer**: every city/network is a self-contained *project* under
`projects/<id>/`, with its own dataset, models, registry and configuration. No
forecasting math, feature engineering, hyperparameters or Monte-Carlo / NWP-noise
logic changed — this was purely a structural / path-resolution refactor plus a
data migration. **Flensburg has been migrated into `projects/flensburg/`.**

### Directory structure

```
projects/
└── flensburg/
    ├── config.json
    ├── data/
    │   └── processed/
    │       └── demand_with_weather.csv      # migrated (git mv, history preserved)
    └── models/
        ├── registry.json                    # committed; pkl_dir now project-relative
        ├── multi_horizon/
        │   ├── meta.json                     # committed
        │   └── lgbm_h01.pkl … h48.pkl       # gitignored
        └── versions/                         # gitignored PKLs; retrained versions
```

Each project is fully independent — adding a new city is a matter of creating a
new `projects/<id>/` folder with a `config.json` and a dataset. This is the
foundation for the upcoming **"forecast builder" wizard** (a later phase) that
will create projects from uploaded data.

### `config.json` schema

```json
{
  "id": "flensburg",
  "name": "Flensburg",
  "city": "Flensburg, Germany",
  "lat": 54.79,
  "lon": 9.44,
  "timezone": "Europe/Berlin",
  "demand_unit": "MW",
  "target_column": "heat_demand_mw",
  "created_at": "2026-05-31T06:15:53Z"
}
```

### `Project` abstraction (`src/project.py`)

`Project` resolves the repo root robustly (`Path(__file__).resolve().parent.parent`)
and exposes `PROJECTS_DIR = repo_root / "projects"`.

| Member | Description |
|---|---|
| `Project.load(id)` | Reads `projects/<id>/config.json`; exposes `id, name, city, lat, lon, timezone, demand_unit, target_column, created_at` |
| `.data_path` | `projects/<id>/data/processed/demand_with_weather.csv` |
| `.models_dir` | `projects/<id>/models` |
| `.default_pkl_dir` | `projects/<id>/models/multi_horizon` |
| `.registry_path` | `projects/<id>/models/registry.json` |
| `.default_meta_path` | `projects/<id>/models/multi_horizon/meta.json` |
| `.versions_dir` | `projects/<id>/models/versions` |
| `.rel_default_pkl_dir` | repo-relative string stored in the registry (e.g. `projects/flensburg/models/multi_horizon/`) |
| `Project.list_all()` | Scans `projects/*/config.json`; returns ids (default project first, then alphabetical) |
| `Project.default_id()` | Sensible default active project id (`flensburg`) |

### How project-awareness is wired

- **`app.py`** resolves the active project from a top-of-sidebar `Project` selector
  (`st.session_state["active_project_id"]`, default `flensburg`). `init()` is cached
  per `(project_id, pkl_dir)` and loads that project's CSV + PKLs; the `ModelRegistry`,
  Open-Meteo `lat/lon`, page header, captions and the retrain subprocess (`--project <id>`)
  all derive from the active project.
- **`train_multi_horizon.py`** takes `--project` (default `flensburg`); it resolves the
  data path, default output dir and registry through the `Project`. The existing
  `--start-date/--end-date/--label/--output-dir/--features` flags still work, scoped to
  the chosen project. Registry `pkl_dir` values are written repo-relative.
- **`src/data.py`**, **`src/weather.py`**, **`src/model_registry.py`** keep
  backward-compatible defaults (pointing at the Flensburg project) but accept caller-supplied
  paths/coords so they are project-agnostic.

### `.gitignore`

Because models moved under `projects/`, the new ignore patterns are:

```
projects/**/models/**/*.pkl
projects/**/models/**/*.joblib
projects/**/models/**/*.lgb
projects/**/models/**/*.txt
projects/**/data/raw/
```

`meta.json` and `registry.json` remain **tracked** (no binary weights are committed).

---

## Key Architectural Decisions

### 1. Multi-horizon direct strategy (48 models)

Each horizon h=1…48 has its own dedicated LightGBM model trained directly on `demand(t+h)` as the target, with features observed at snapshot time `t` plus `fc_*`-prefixed weather/calendar features at `t+h`.

**Why:** The single-model iterative (recursive) strategy was found to be deceptively accurate at short horizons due to data leakage — pre-computed lag features referenced future demand values. The direct strategy eliminates this leakage by construction.

**Single-model fallback:** `get_forecast_window()` implements an iterative recursive approach (building a `demand_buf` and recomputing lags step-by-step) as a fallback when PKLs are missing. It is not used in production once models are trained.

### 2. Feature set for multi-horizon models (30 features)

| Group | Count | Features |
|---|---|---|
| Demand lags at t | 9 | lag_1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h, 336h |
| Demand rolls at t | 2 | roll_24h, roll_168h |
| Weather fc at t+h | 11 | temperature_c, heating_degrees, temp_change_3h, wind_speed_ms, wind_sin, wind_cos, solar_radiation_wm2, humidity_pct, snowfall_cm, snow_depth_m, cloud_cover_pct |
| Calendar fc at t+h | 8 | hour, day_of_week, month, day_of_year, is_weekend, is_holiday, is_school_holiday, heating_season_day |

**Short-term lags (2h–12h) were added later** after discovering that the initial models lacked granular recent-trend information for near-horizon predictions. LightGBM down-weights them automatically for longer horizons.

**fc_* prefix convention:** Weather and calendar features for the target time `t+h` are prefixed `fc_` to distinguish them from the corresponding observed values at snapshot time `t`.

### 3. Weather forecast integration (Open-Meteo)

Real weather forecasts replace historical observed weather for future steps, via the [Open-Meteo API](https://open-meteo.com/):
- **Current/future dates:** `/v1/forecast` endpoint
- **Historical/time-travel dates:** `/v1/historical-forecast` archive endpoint
- **Derived features computed from raw API output:** `heating_degrees`, `temp_change_3h`, `wind_sin`, `wind_cos`
- **Optional at runtime:** controlled by a sidebar checkbox in the dashboard

Without the API, the dashboard falls back to oracle weather (historical actuals from the CSV), which inflates short-horizon accuracy.

### 4. Uncertainty quantification

80% prediction interval: `±1.28 × σ`

**σ per horizon** = per-horizon RMSE from the 2024 hold-out evaluation, stored in `meta.json`.

**NWP widening:** if the Open-Meteo API is active, an additional component is added in quadrature: `σ_nwp = temp_forecast_error × demand_sensitivity`. This accounts for the fact that forecast temperature errors compound into demand forecast errors.

**Band visualisation:** the band is bridged to the last observed history point so it fans out from zero at "Now", giving an honest visual of growing uncertainty with horizon.

### 5. `@st.cache_resource` caching

`init()` in `app.py` loads the CSV and all 48 PKL models into memory on first call. This is cached for the lifetime of the Streamlit server process.

**Important:** if models are retrained, the Streamlit server must be **restarted** to clear the cache and reload new models. Hot-reload does not update `@st.cache_resource` values.

### 6. Time-travel inspection

The dashboard sidebar exposes a date picker and hour slider. Any date in 2024 can be selected as the "snapshot" (current time). The dashboard then:
- Shows 24h of history up to the snapshot
- Shows the model's forecast as it would have been made at that moment (using only data available up to that point)
- Overlays the actual future demand as ground truth for performance comparison

---

## Data Pipeline

### Offline / one-time

```
Zenodo Excel + Open-Meteo archive
        ↓  (Notebook 02)
demand_with_weather.csv
        ↓  (train_multi_horizon.py)
lgbm_h01-48.pkl + meta.json
```

Run training: `python3 train_multi_horizon.py`

### Runtime (Streamlit)

```
demand_with_weather.csv + PKLs + meta.json
        ↓  init() @cached
        ↓  get_mh_forecast_window(snapshot_dt, horizon, wx_forecast)
        ↓  Streamlit charts + KPIs + operator log
```

Run dashboard: `streamlit run app.py`

---

## Inference Flow (`get_mh_forecast_window`)

1. **Slice history** — extract rows from `df_raw` up to `snapshot_dt`
2. **Compute lags at t** — read all 11 lag/roll features from history (no prediction feedback needed)
3. **For h = 1 … horizon:**
   - Load `lgbm_hNN` from in-memory cache
   - Build `fc_feats` at `t+h`: weather from `wx_forecast` if available, else `df_raw` fallback; plus calendar features
   - Call `model_h.predict([lag_feats | fc_feats])` → point estimate
   - Compute `σ_h = RMSE_h (± NWP widening)`; `upper/lower = forecast ± 1.28σ`
4. **Return** dict: `history`, `forecast`, `upper`, `lower`, `actual_future`, `weather`

---

## Model Performance (2024 hold-out)

| Metric | h=1 | h=24 | h=48 | Average |
|---|---|---|---|---|
| MAPE | ~6.6% | ~7.9% | ~8.3% | ~7.8% |

The relatively flat MAPE curve across horizons reflects that temperature and calendar features dominate over demand lags — the forecast skill does not degrade sharply at longer horizons.

---

## Dashboard (app.py)

- **Framework:** Streamlit with `st.set_page_config(layout="wide")`
- **Charts:** Plotly with two subplots (demand + weather), shared x-axis at the top, day-band annotations
- **Colour palette:** Gradyent.ai-inspired blue palette; Open Sans font
- **Legends:** positioned outside the chart area on the right, split into demand and weather legends
- **KPIs:** current demand, forecast MAPE vs actuals, peak forecast, 80% band coverage
- **Operator log:** free-text maintenance/event notes stored in Streamlit session state (not fed to the model)
- **Bottom panels:** per-horizon MAPE chart, feature importance from one of the MH models

### Chart x-axis convention
- Shared x-axis shown at the **top** of the first subplot only
- Time ticks every 3 hours (`dtick = 10_800_000` ms)
- Day-band annotations (`<b>Sat 27 May</b>`) drawn above the axis at `y = 1.13`

---

## Model Versioning

### Overview

Model versioning allows multiple trained configurations to coexist and be selected at runtime without restarting the Streamlit server.

### registry.json schema

Stored at `models/registry.json` (committed to git — no binary weights).

```json
{
  "versions": [
    {
      "id": "2020-2023_default",
      "label": "default",
      "date_range_start": "2020-01-01",
      "date_range_end": "2023-12-31",
      "trained_at": "2026-05-30T17:52:00",
      "mape_avg": 7.8,
      "mape_per_horizon": {"h1": 6.6, "h24": 7.9, "h48": 8.3},
      "pkl_dir": "models/multi_horizon/",
      "is_default": true
    }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique key: `{start}_{end}[_{label}]` |
| `label` | string | Human-readable tag (empty for default) |
| `date_range_start/end` | ISO date | Training window used |
| `trained_at` | ISO datetime | UTC timestamp of training run |
| `mape_avg` | float | Average MAPE across h=1…48 on 2024 test set |
| `mape_per_horizon` | object | Per-horizon MAPE keyed `h1`…`h48` |
| `pkl_dir` | string | Relative path to the directory holding PKLs + meta.json |
| `is_default` | bool | True for exactly one entry (the fallback version) |

### Folder structure

```
models/
├── registry.json               # committed — version metadata only, no weights
├── multi_horizon/              # default version (gitignored PKLs)
│   ├── meta.json               # committed
│   └── lgbm_h01.pkl … h48.pkl # gitignored
└── versions/                   # gitignored — custom trained versions
    └── 2021-01-01_2022-12-31_2yr-short/
        ├── meta.json
        └── lgbm_h01.pkl … h48.pkl
```

`models/versions/` should be added to `.gitignore` alongside `models/multi_horizon/*.pkl`.

### New CLI arguments on train_multi_horizon.py

| Argument | Default | Description |
|---|---|---|
| `--start-date YYYY-MM-DD` | earliest row in CSV | Training start date |
| `--end-date YYYY-MM-DD` | `2023-12-31` | Training end date |
| `--label TEXT` | `""` | Version label stored in registry |
| `--output-dir PATH` | `models/multi_horizon/` (legacy) | Where to save PKLs + meta.json |

When `--output-dir` is supplied the script saves to that directory and calls
`ModelRegistry.register_version(...)` before exiting. The version id is printed as
`Version registered: {id}`.

When no arguments are given, behaviour is **identical to the pre-versioning script** —
PKLs go to `models/multi_horizon/` and the registry is not touched.

### src/model_registry.py

`ModelRegistry` is a lightweight class backed by `models/registry.json`.

| Method | Description |
|---|---|
| `load()` / `save()` | Read/write JSON; `save()` creates parent dirs |
| `list_versions()` | Returns versions sorted by `trained_at` descending |
| `get_version(id)` | Single entry dict, or `None` |
| `register_version(...)` | Upsert entry and persist |
| `auto_register_existing_default()` | Seeds registry from `models/multi_horizon/meta.json` on first app load; silent no-op if registry already exists or meta.json is missing |

### Version selector + retrain UI (Dashboard)

**Model version selectbox** — rendered near the top of the sidebar.

- Lists all registered versions formatted as `{start[:4]}–{end[:4]}  {label}  (MAPE N.N%)`.
- Versions are sorted newest-first.
- Switching version calls `st.cache_resource.clear()` so the new PKLs are loaded fresh.
- Each distinct `pkl_dir` gets its own `@st.cache_resource` slot, so versions load in the background after first use without evicting each other.

**🔁 Retrain model expander** — below the version selector.

- Date range inputs + optional label text input.
- Validates that end date > start date + 90 days.
- Spawns `train_multi_horizon.py` as a subprocess and streams stdout line-by-line into `st.status(…)`.
- On success: offers "Switch to new version" (updates `session_state["active_pkl_dir"]`, clears cache, reruns) or "Keep current version".
- On failure: shows last 20 lines of output in a code block.

---

## Simplification (completed on branch `refactor/simplify-model-modules`)

The following cleanup was completed:

1. **Archived notebooks 03, 04, 05** → moved to `notebooks/archive/`; `prophet` removed from `requirements.txt`
2. **Removed single-model fallback** — `load_and_prepare`, `train_model`, `monthly_mape`, `get_forecast_window` deleted from `src/model.py`; `USE_MH` flag and fallback init/dispatch logic removed from `app.py`
3. **Split `src/model.py`** into three focused modules (`src/data.py`, `src/weather.py`, `src/model.py`)

Total line reduction: ~1162 → 990 lines across all Python source files.

---

## Environment Notes

- Python 3.x; `streamlit run app.py` from project root
- No `.streamlit/` config on disk (gitignored); Streamlit uses defaults
- Model PKLs are gitignored — must run `train_multi_horizon.py` after cloning
- `data/raw/` is gitignored — download from Zenodo 17177421
- `meta.json` **is** committed — it contains eval metrics but no binary model weights
