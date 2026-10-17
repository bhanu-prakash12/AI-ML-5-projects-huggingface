"""Phase 3: replay the stream as simulated days through the API; labels arrive `label_lag_days` later.

usage: simulate.py [--scenario A|B|C|none] [--start-day N] [--days N] [--reset] [--url URL]
"""
import argparse
import sys
import time
from datetime import datetime, timedelta
from typing import Callable

sys.path.insert(0, "src")
import httpx  # noqa: E402
import pandas as pd  # noqa: E402

from fraud_monitor import config, console, data, scenarios  # noqa: E402

BASE = datetime(2026, 1, 1)
CHUNK = 1000


def run(client, stream: pd.DataFrame, ref: pd.DataFrame, feats: list[str], scenario: str | None,
        start_day: int | None, days: int | None = None, lag: int | None = None, reset: bool = False,
        on_day: Callable[[int, int], None] | None = None) -> dict:
    cfg = config.load()
    lag = cfg["label_lag_days"] if lag is None else lag
    days = days or int(stream.day.max())
    if reset:
        client.delete("/predictions").raise_for_status()
    ids: dict[int, list[tuple[str, int]]] = {}
    sent = fed = 0

    def feedback(day: int) -> int:
        n = 0
        items = [{"request_id": r, "label": y} for r, y in ids.pop(day, [])]
        for i in range(0, len(items), 5000):
            n += client.post("/feedback/batch", json={"items": items[i:i + 5000]}).json()["matched"]
        return n

    for d in range(1, days + 1):
        day_df = scenarios.apply(stream[stream.day == d].reset_index(drop=True), d, ref, feats, scenario, start_day)
        ts0 = BASE + timedelta(days=d - 1)
        rows = [{"request_id": f"d{d:02d}-{i:05d}", "day": d,
                 "ts": (ts0 + timedelta(seconds=i * 86400 // max(len(day_df), 1))).isoformat(),
                 "features": {f: float(v) for f, v in zip(feats, vals)}}
                for i, vals in enumerate(day_df[feats].to_numpy())]
        for i in range(0, len(rows), CHUNK):
            r = client.post("/predict/batch", json={"items": rows[i:i + CHUNK]})
            r.raise_for_status()
        sent += len(rows)
        ids[d] = [(row["request_id"], int(y)) for row, y in zip(rows, day_df.Class)]
        if d - lag >= 1:
            fed += feedback(d - lag)
        if on_day:
            on_day(d, days)
    return {"sent": sent, "labels_fed": fed, "unlabelled_tail": sum(len(v) for v in ids.values())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Replay the held-out stream as simulated days through the running API; labels arrive 1 day late.",
        epilog="Inputs: data/stream.parquet, data/reference.parquet, a running API (python run.py serve).\n"
               "Outputs: the API's SQLite prediction log (artifacts/predictions.db, or $P4_DB).\n"
               "Example: python scripts/simulate.py --reset --scenario B --start-day 6",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default=None, help="A (amount x3), B (new fraud cluster), C (concept flip), none")
    ap.add_argument("--start-day", type=int, default=None, help="first day the scenario is applied (default from config)")
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--reset", action="store_true", help="wipe the prediction log first")
    ap.add_argument("--url", default=f"http://127.0.0.1:{config.load()['ports']['api']}")
    a = ap.parse_args()
    console.setup_logging()
    console.banner("simulate: replay traffic", f"Send each simulated day to {a.url}/predict/batch, then the previous day's "
                   f"labels to /feedback/batch. Scenario: {a.scenario or 'none (clean)'}"
                   + (f" from day {a.start_day}" if a.scenario and a.start_day else ""),
                   "one line per day, then a summary of rows sent and labels fed back")
    console.step(1, 3, "loading reference + stream parquet")
    ref = pd.read_parquet(config.path("data") / "reference.parquet")
    stream = pd.read_parquet(config.path("stream"))
    console.step(2, 3, "replaying days through the API")
    t0 = time.time()
    with httpx.Client(base_url=a.url, timeout=120) as c:
        res = run(c, stream, ref, data.feature_cols(ref), a.scenario, a.start_day, a.days, reset=a.reset,
                  on_day=lambda d, n: console.hint(f"day {d}/{n} sent ({time.time() - t0:.0f}s)"))
    console.step(3, 3, "summary")
    console.kv_table({**res, "wall_time_s": round(time.time() - t0, 1)})
    console.note("every prediction is logged with its features, so any window can be re-analysed later",
                 "labels lag by design: the newest day has no ground truth yet (unlabelled_tail)",
                 "a Kafka topic replaces the HTTP replay; labels come from a chargeback pipeline")
