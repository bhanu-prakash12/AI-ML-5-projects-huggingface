"""Train, track, register, export. Run via `python run.py train` (or `make train`)."""
import _boot  # noqa: F401  (adds src/ to sys.path)
import argparse
import logging
import sys
import time

from dropout_risk import console
from dropout_risk.config import project_path
from dropout_risk.train import main


def _max_rss_bytes() -> int | None:
    """Peak RSS in bytes where the stdlib `resource` module exists (macOS/Linux); None on Windows."""
    try:
        import resource
    except ImportError:
        return None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024  # Linux reports KiB


def cli() -> None:
    ap = argparse.ArgumentParser(
        description="Train 12 LightGBM/XGBoost runs, log them to MLflow, register the champion "
                    "and export artifacts/model/.",
        epilog="Input: data/processed/*.parquet (run `python run.py data` first). "
               "Output: MLflow runs (mlflow.db), artifacts/model/, reports/metrics.json, "
               "reports/fairness.md, reports/train_time.txt. "
               "Sample: python scripts/train.py")
    ap.parse_args()
    console.setup_logging()
    for noisy in ("alembic", "mlflow"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    console.banner("Train + track + register",
                   "Grid of LightGBM/XGBoost runs, each logged to MLflow; best LightGBM by "
                   "validation PR-AUC becomes the 'champion' and is exported for serving.",
                   "about 10-20 s on a laptop CPU, then a metrics table and the tuned threshold.")
    console.step(1, 3, "training the grid (one MLflow run per config)")
    t0 = time.perf_counter()
    s = main()
    wall = time.perf_counter() - t0
    console.step(2, 3, "champion + metrics")
    console.kv_table({"champion": f"{s['champion']['kind']} {s['champion']['params']}",
                      "registry version": s["champion"]["registry_version"],
                      "threshold": f"{s['threshold']['value']:.3f} ({s['threshold']['rule']})"})
    console.table([(k, f"{s['val'][k]:.3f}", f"{s['test'][k]:.3f}")
                   for k in ("roc_auc", "pr_auc", "f1", "recall", "precision")],
                  headers=("metric", "validation", "test"))
    console.step(3, 3, "recording wall time / peak RAM")
    rss = _max_rss_bytes()
    line = f"wall_s={wall:.2f} max_rss_bytes={rss if rss is not None else 'n/a (Windows)'}"
    (project_path("reports") / "train_time.txt").write_text(line + "\n")
    console.ok(line + "  -> reports/train_time.txt")
    console.hint("Next: python run.py eval   |   python run.py mlflow-ui  (http://127.0.0.1:5001)")
    console.note(
        "fast, reproducible (seed 42), every run is tracked and comparable in MLflow.",
        "small tabular data and a late-stage feature set; threshold tuned on validation only.",
        "grid search -> scheduled retraining job + registry alias promotion; same MLflow API.")


if __name__ == "__main__":
    cli()
