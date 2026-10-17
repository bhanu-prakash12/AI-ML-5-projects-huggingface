"""Champion vs challenger on the latest labelled window; promote only if clearly better."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, "src")
import httpx  # noqa: E402

from fraud_monitor import console, narrate, retrain  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Train a challenger on reference + labelled traffic, compare with the champion on the latest window.",
        epilog="Inputs: the prediction log (--db) with labels, data/reference.parquet, the MLflow registry.\n"
               "Outputs: <out>/retrain.json, a new MLflow model version, and (if promoted) the champion alias moves.\n"
               "Example: python scripts/retrain.py --db .run/demo.db --out reports/demo --no-promote",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-promote", action="store_true", help="dry run: evaluate the gate but keep the current champion")
    ap.add_argument("--reload-url", default=None, help="e.g. http://127.0.0.1:8004 -> POST /reload after promotion")
    a = ap.parse_args()
    console.setup_logging()
    console.banner("retrain: champion / challenger gate",
                   "Fit a challenger on days before the latest window, score both models on that window, "
                   "promote only if the gain is >= retrain.min_improvement and the old pattern is not forgotten.",
                   "champion vs challenger PR-AUC and a promote / reject verdict (~1 min)")
    console.step(1, 2, "training challenger and comparing")
    r = retrain.run(a.db, a.out, promote_alias=not a.no_promote)
    console.step(2, 2, "verdict")
    for line in narrate.explain_retrain(r):
        (console.ok if r["promoted"] else console.warn)(line)
    console.kv_table({"eval_window": r["eval_window"], "gate_passed": r["promoted"], "alias_updated": r["alias_updated"],
                      "champion_after": r["new_champion_version"]})
    if r["alias_updated"] and a.reload_url:
        console.ok(f"reloaded API: {httpx.post(a.reload_url + '/reload').json()}")
    console.note("a gate (min gain + no regression on the old pattern) stops noisy promotions",
                 "a few hundred labelled frauds only; the synthetic cluster is easy, real gains are smaller",
                 "needs shadow scoring, approval steps and a rollback alias in a real registry")
