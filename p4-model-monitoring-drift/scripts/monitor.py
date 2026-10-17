"""Evidently report per window + alert rule -> <out>/{windows/*.html,status.json,alerts.jsonl}."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
from fraud_monitor import config, console, monitor, narrate  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Compare each simulated day (window) with the reference, apply the alert rule, write reports.",
        epilog="Inputs: the SQLite prediction log (--db, default artifacts/predictions.db) + reports/baseline.json.\n"
               "Outputs: <out>/status.json, <out>/alerts.jsonl, <out>/windows/window_NN.html (Evidently).\n"
               "Example: python scripts/monitor.py --db .run/demo.db --out reports/demo",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=None, help="prediction log (default: config paths.db / $P4_DB)")
    ap.add_argument("--out", type=Path, default=None, help="output dir (default: reports/)")
    a = ap.parse_args()
    console.setup_logging()
    cfg = config.load()["monitor"]
    console.banner("monitor: drift + performance per window",
                   f"Alert when drift share > {cfg['drift_share_threshold']} OR PR-AUC drops > "
                   f"{cfg['pr_auc_drop_threshold']:.0%} vs baseline (needs >= {cfg['min_frauds_for_perf']} labelled frauds).",
                   "one line per window (about 9 s each: Evidently renders an HTML report per window)")
    console.step(1, 2, "running Evidently on every window")
    t0 = time.time()
    rows = monitor.run(a.db, a.out)
    console.step(2, 2, "results")
    for r in rows:
        level, text = narrate.window_line(r)
        getattr(console, level)(text)
    console.table([{"window": r["window"], "n": r["n"], "frauds": r["n_fraud"], "drift_share": f"{r['drift_share']:.2f}",
                    "psi": f"{r['psi_score']:.3f}", "pr_auc": "n/a" if r["pr_auc"] is None else f"{r['pr_auc']:.3f}",
                    "alert": r["alert"]} for r in rows], title="windows", highlight=lambda x: "red" if x["alert"] else None)
    console.kv_table({"windows": len(rows), "alerts": sum(r["alert"] for r in rows), "wall_time_s": round(time.time() - t0, 1),
                      "reports": str((a.out or config.path("reports")) / "windows")})
    console.note("drift share alerts the same day, before labels exist; PR-AUC confirms real damage later",
                 "PR-AUC on ~10 frauds is noisy (guard skips windows with too few); labels arrive late",
                 "per-feature drift x many models needs a scheduled job + metrics store (Prometheus/Grafana), not HTML files")
