"""In-process demo pipeline used by the dashboard buttons: simulate -> monitor -> retrain (no separate server).

The FastAPI app is driven through Starlette's TestClient, so the click-driven demo works with a single process
(same code path as the real API: /predict/batch + /feedback/batch + SQLite log).
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import pandas as pd

from . import config, data, model, monitor, retrain

DEMO_DIR = "demo"


def demo_db() -> Path:
    return config.ROOT / ".run" / "ui_demo.db"


def demo_out() -> Path:
    d = config.path("reports") / DEMO_DIR
    (d / "windows").mkdir(parents=True, exist_ok=True)
    return d


@contextmanager
def _env_db(db: Path):
    old = os.environ.get("P4_DB")
    os.environ["P4_DB"] = str(db)
    db.parent.mkdir(parents=True, exist_ok=True)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("P4_DB", None)
        else:
            os.environ["P4_DB"] = old


def simulate(scenario: str | None, start_day: int | None, on_day: Callable[[int, int], None] | None = None) -> dict:
    from fastapi.testclient import TestClient

    import simulate as sim  # scripts/simulate.py (scripts dir is on sys.path for run.py / ui.py)
    from . import api

    ref = pd.read_parquet(config.path("data") / "reference.parquet")
    stream = pd.read_parquet(config.path("stream"))
    scenario = None if scenario in (None, "", "none") else scenario
    with _env_db(demo_db()), TestClient(api.app) as c:
        return sim.run(c, stream, ref, data.feature_cols(ref), scenario, start_day, reset=True, on_day=on_day)


def run_monitor() -> list[dict]:
    with _env_db(demo_db()):
        return monitor.run(demo_db(), demo_out())


def run_retrain(promote: bool = False) -> dict:
    with _env_db(demo_db()):
        return retrain.run(demo_db(), demo_out(), promote_alias=promote)


def reset_champion() -> str:
    return model.reset_champion()
