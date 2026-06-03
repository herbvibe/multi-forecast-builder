"""
SQLite-backed forecast log for the Live Forecaster tab.

Each time the live tab runs a forecast, the predicted values for all horizons
are logged to a per-project SQLite database. The h=1 forecast is later used
as a proxy "actual" line on the chart (labelled "Logged 1h forecasts").

Table schema:
    forecast_log(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id TEXT NOT NULL,
        logged_at  TEXT NOT NULL,   -- ISO-8601 UTC
        forecast_for TEXT NOT NULL, -- ISO-8601 UTC (= snapshot_dt + h hours)
        horizon_h  INTEGER NOT NULL,
        forecast_value REAL NOT NULL
    )

Index on (project_id, forecast_for, horizon_h) for efficient proxy-actuals queries.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.project import Project


def _db_path(project: Project) -> Path:
    return project.root / "forecast_log.db"


def _connect(project: Project) -> sqlite3.Connection:
    """Open (and initialise if new) the forecast-log database for *project*."""
    path = _db_path(project)
    con = sqlite3.connect(path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS forecast_log (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id   TEXT    NOT NULL,
            logged_at    TEXT    NOT NULL,
            forecast_for TEXT    NOT NULL,
            horizon_h    INTEGER NOT NULL,
            forecast_value REAL  NOT NULL
        )
    """)
    con.execute("""
        CREATE INDEX IF NOT EXISTS idx_fc_log
        ON forecast_log (project_id, forecast_for, horizon_h)
    """)
    con.commit()
    return con


def log_run(
    project: Project,
    snapshot_dt: pd.Timestamp,
    forecasts: dict[int, float],
) -> None:
    """Persist all horizon forecasts from one live run.

    ``forecasts`` maps horizon_h → predicted_value_mw.
    ``snapshot_dt`` is the "now" moment the forecast was issued.
    """
    logged_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []
    for h, val in forecasts.items():
        fc_ts = (snapshot_dt + pd.Timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%S")
        rows.append((project.id, logged_at, fc_ts, int(h), float(val)))

    con = _connect(project)
    con.executemany(
        "INSERT INTO forecast_log (project_id, logged_at, forecast_for, horizon_h, forecast_value) "
        "VALUES (?,?,?,?,?)",
        rows,
    )
    con.commit()
    con.close()


def get_proxy_actuals(
    project: Project,
    from_ts: pd.Timestamp,
    to_ts: pd.Timestamp,
) -> pd.Series:
    """Return the most recently logged h=1 forecast for each hour in [from_ts, to_ts].

    This series is used as the "proxy actuals" line on the live chart.
    Returns an empty Series if no logged data exists for the window.
    """
    con = _connect(project)
    rows = con.execute(
        """
        SELECT forecast_for, forecast_value
        FROM   forecast_log
        WHERE  project_id  = ?
          AND  horizon_h   = 1
          AND  forecast_for >= ?
          AND  forecast_for <= ?
        GROUP BY forecast_for
        -- "most recently logged" = maximum logged_at for each forecast_for
        HAVING logged_at = MAX(logged_at)
        ORDER BY forecast_for
        """,
        (
            project.id,
            from_ts.strftime("%Y-%m-%dT%H:%M:%S"),
            to_ts.strftime("%Y-%m-%dT%H:%M:%S"),
        ),
    ).fetchall()
    con.close()

    if not rows:
        return pd.Series(dtype=float)

    index = pd.to_datetime([r[0] for r in rows])
    values = [r[1] for r in rows]
    return pd.Series(values, index=index, name="proxy_actual")
