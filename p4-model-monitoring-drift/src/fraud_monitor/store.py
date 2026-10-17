"""SQLite prediction log. One row per request; labels arrive later via /feedback."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions(
  request_id TEXT PRIMARY KEY, ts TEXT NOT NULL, day INTEGER NOT NULL, model_version TEXT NOT NULL,
  score REAL NOT NULL, features TEXT NOT NULL, label INTEGER, label_ts TEXT);
CREATE INDEX IF NOT EXISTS ix_day ON predictions(day);
"""


@contextmanager
def conn(db: Path | None = None):
    c = sqlite3.connect(db or config.path("db"), timeout=30)
    c.execute("PRAGMA journal_mode=WAL")
    c.executescript(SCHEMA)
    try:
        yield c
        c.commit()
    finally:
        c.close()


def log_predictions(rows: list[tuple], db: Path | None = None) -> None:
    """rows: (request_id, ts, day, model_version, score, features_dict)."""
    with conn(db) as c:
        c.executemany(
            "INSERT OR REPLACE INTO predictions(request_id,ts,day,model_version,score,features) VALUES(?,?,?,?,?,?)",
            [(r[0], r[1], r[2], r[3], r[4], json.dumps(r[5])) for r in rows],
        )


def log_labels(rows: list[tuple[str, int, str]], db: Path | None = None) -> int:
    """rows: (request_id, label, label_ts). Returns number of matched requests."""
    with conn(db) as c:
        n = 0
        for rid, lab, ts in rows:
            n += c.execute("UPDATE predictions SET label=?, label_ts=? WHERE request_id=?", (lab, ts, rid)).rowcount
        return n


def reset(db: Path | None = None) -> None:
    with conn(db) as c:
        c.execute("DELETE FROM predictions")


def read(db: Path | None = None) -> pd.DataFrame:
    """Prediction log with features expanded to columns."""
    with conn(db) as c:
        df = pd.read_sql_query("SELECT * FROM predictions ORDER BY ts", c)
    if df.empty:
        return df
    feats = pd.DataFrame([json.loads(s) for s in df.pop("features")])
    return pd.concat([df.reset_index(drop=True), feats], axis=1)
