"""Print saved test metrics (reports/metrics.json)."""
import _boot  # noqa: F401  (adds src/ to sys.path)
import argparse
import json

from dropout_risk import console
from dropout_risk.config import project_path


def cli() -> None:
    argparse.ArgumentParser(
        description="Show the champion's saved threshold and held-out test metrics.",
        epilog="Input: reports/metrics.json (written by train). Output: console table only. "
               "Sample: python scripts/evaluate.py").parse_args()
    console.setup_logging()
    console.banner("Evaluate champion", "Reads reports/metrics.json; no retraining.",
                   "ROC-AUC / PR-AUC / F1 / recall / precision on the untouched test split.")
    m = json.loads((project_path("reports") / "metrics.json").read_text())
    console.step(1, 2, "threshold chosen on validation")
    console.kv_table({"threshold": f"{m['threshold']['value']:.3f}", "rule": m["threshold"]["rule"]})
    console.step(2, 2, "test metrics")
    console.table([(k, f"{v:.3f}") for k, v in m["test"].items()], headers=("metric", "test"))
    console.ok("flag students at/above the threshold; see reports/fairness.md for per-group rates")
    console.note(
        "recall-first threshold catches most true dropouts at precision >= 0.6.",
        "precision is only ~0.6-0.65: about 1 in 3 flags is a false alarm.",
        "re-evaluate per cohort/term; drift in the base rate shifts the best threshold.")


if __name__ == "__main__":
    cli()
