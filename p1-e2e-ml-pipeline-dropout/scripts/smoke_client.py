"""Real HTTP requests against a running API: 3 smoke calls, then p50/p95 latency -> reports/latency.md."""
from __future__ import annotations

import _boot  # noqa: F401  (adds src/ to sys.path)
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

from dropout_risk import console

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8000"
N = 200


def call(path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(
        BASE + path, data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def main(base: str | None = None) -> None:
    global BASE
    BASE = base or (sys.argv[1] if len(sys.argv) > 1 else BASE)
    console.banner("Smoke client", "Real HTTP calls: /health, /predict, /predict/batch, then 200 timed calls.",
                   "a tier + probability for the first test student, then p50/p95 latency.")
    meta = json.loads((ROOT / "artifacts/model/preprocess.json").read_text())
    df = pd.read_parquet(ROOT / "data/processed/test.parquet")
    rows = df[meta["columns"]].astype(float).head(5).to_dict("records")
    h = call("/health")
    assert h["status"] == "ok", h
    p = call("/predict", rows[0])
    assert 0 <= p["probability"] <= 1 and len(p["top_features"]) == 3, p
    b = call("/predict/batch", {"rows": rows})
    assert len(b["predictions"]) == 5, b
    console.ok(f"smoke ok: {h['model_version']} tier={p['risk_tier']} p={p['probability']:.3f}")
    console.table([(f["feature"], f"{f['contribution']:+.3f}", f["direction"]) for f in p["top_features"]],
                  headers=("top feature", "SHAP (log-odds)", "direction"))
    times = []
    for i in range(N):
        t = time.perf_counter()
        call("/predict", rows[i % len(rows)])
        times.append((time.perf_counter() - t) * 1000)
    q = statistics.quantiles(times, n=100)
    out = (f"# Latency (single /predict, local uvicorn, {N} sequential requests)\n\n"
           f"| p50 ms | p95 ms | max ms |\n|---|---|---|\n"
           f"| {statistics.median(times):.2f} | {q[94]:.2f} | {max(times):.2f} |\n\n"
           f"Model: {h['model_version']}. Measured by scripts/smoke_client.py over loopback HTTP.\n")
    (ROOT / "reports/latency.md").write_text(out)
    console.ok("wrote reports/latency.md")
    console.hint(out)


if __name__ == "__main__":
    main()
