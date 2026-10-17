"""Train the baseline LightGBM on the reference part, write reports/baseline.json, register the champion."""
import argparse
import sys
import time

sys.path.insert(0, "src")
import pandas as pd  # noqa: E402

from fraud_monitor import config, console, data, model  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__,
        epilog="Inputs: data/reference.parquet (python run.py data).\n"
               "Outputs: reports/baseline.json, MLflow run + registered model `fraud` (alias champion).\n"
               "Example: python run.py train",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    console.setup_logging()
    console.banner("train baseline", "LightGBM on the first 80% of the reference rows, validated on the last 20% (time order).",
                   "PR-AUC (the primary metric) well above accuracy-for-free; takes under a minute on CPU")
    t0 = time.time()
    cfg = config.load()
    console.step(1, 3, "loading reference data")
    ref = pd.read_parquet(config.path("data") / "reference.parquet")
    feats = data.feature_cols(ref)
    n_val = int(len(ref) * cfg["model"]["val_frac"])
    tr, va = ref.iloc[:-n_val], ref.iloc[-n_val:]
    console.step(2, 3, f"fitting on {len(tr)} rows ({int(tr.Class.sum())} frauds)")
    m = model.fit(tr[feats], tr.Class)
    met = model.metrics(va.Class.to_numpy(), m.predict_proba(va[feats])[:, 1])
    naive_acc = float(1 - va.Class.mean())  # predict "never fraud"
    out = {"metrics_val": met, "naive_always_legit_accuracy": naive_acc, "primary_metric": "pr_auc",
           "n_train": len(tr), "n_val": len(va), "train_frauds": int(tr.Class.sum()), "features": feats,
           "split": "time-ordered: first 80% of reference = train, last 20% = validation"}
    console.step(3, 3, "registering the model in MLflow and setting alias `champion`")
    ver = model.register(m, cfg["model"]["params"], met, "baseline")
    model.set_champion(ver)
    out["champion_version"] = ver
    model.dump(config.path("reports") / "baseline.json", out)
    console.kv_table({"PR-AUC (val)": f"{met['pr_auc']:.3f}", "accuracy (val)": f"{met['accuracy']:.4f}",
                      "always-legit accuracy": f"{naive_acc:.4f}", "recall": f"{met['recall']:.3f}",
                      "precision": f"{met['precision']:.3f}", "champion version": ver,
                      "wall_time_s": round(time.time() - t0, 1)})
    console.note("PR-AUC ignores the huge legit majority, so it exposes what accuracy hides",
                 "one fixed split and ~100 training frauds: the number is indicative, not a benchmark",
                 "add a feature store, time-based CV and a registry approval step")
