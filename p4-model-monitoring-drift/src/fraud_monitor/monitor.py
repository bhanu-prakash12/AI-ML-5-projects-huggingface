"""On-demand monitoring: one Evidently report per window (= simulated day) + alert rule.

Rule (spec): alert if drift share > 0.3 OR PR-AUC drop > 10% vs baseline.
Guard (DECISIONS.md): the PR-AUC rule needs >= min_frauds_for_perf labelled frauds in the window.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from . import config, model, store

log = logging.getLogger("fraud_monitor.monitor")


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index on quantile bins of the reference (rule of thumb: >0.25 = major shift)."""
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def decide(drift_share: float | None, pr_auc: float | None, n_fraud: int, baseline: float) -> dict:
    cfg = config.load()["monitor"]
    reasons, perf_skipped = [], None
    if drift_share is not None and drift_share > cfg["drift_share_threshold"]:
        reasons.append(f"drift_share {drift_share:.2f} > {cfg['drift_share_threshold']}")
    if pr_auc is None:
        perf_skipped = "labels pending"
    elif n_fraud < cfg["min_frauds_for_perf"]:
        perf_skipped = f"only {n_fraud} labelled frauds (< {cfg['min_frauds_for_perf']})"
    else:
        drop = (baseline - pr_auc) / baseline
        if drop > cfg["pr_auc_drop_threshold"]:
            reasons.append(f"pr_auc {pr_auc:.3f} dropped {drop:.0%} vs baseline {baseline:.3f}")
    return {"alert": bool(reasons), "reasons": reasons, "perf_skipped": perf_skipped}


def _reference(feats: list[str], version: str) -> pd.DataFrame:
    cfg = config.load()
    ref = pd.read_parquet(config.path("data") / "reference.parquet")
    ref = ref.sample(n=min(cfg["ref_sample"], len(ref)), random_state=cfg["seed"]).sort_index()
    booster, _ = model.load_booster(version)
    out = ref[feats].copy()
    out["prediction"] = booster.predict(out[feats])
    out["target"] = ref["Class"].to_numpy()
    return out


def _evidently(ref: pd.DataFrame, cur: pd.DataFrame, feats: list[str], labelled: bool, html: Path) -> dict:
    from evidently import ColumnMapping
    from evidently.metric_preset import ClassificationPreset, DataDriftPreset, TargetDriftPreset
    from evidently.report import Report
    from evidently.test_suite import TestSuite
    from evidently.tests import TestShareOfDriftedColumns

    cm = ColumnMapping(target="target", prediction="prediction", numerical_features=feats, pos_label=1)
    thr = config.load()["monitor"]["drift_share_threshold"]
    metrics = [DataDriftPreset(columns=feats, drift_share=thr)]
    if labelled:
        metrics += [TargetDriftPreset(), ClassificationPreset(probas_threshold=config.load()["model"]["threshold"])]
        cols = feats + ["prediction", "target"]
    else:
        cols = feats + ["prediction"]
        cm = ColumnMapping(prediction="prediction", numerical_features=feats)
    rep = Report(metrics=metrics)
    rep.run(reference_data=ref[cols], current_data=cur[cols], column_mapping=cm)
    rep.save_html(str(html))
    d = rep.as_dict()["metrics"][0]["result"]
    ts = TestSuite(tests=[TestShareOfDriftedColumns(columns=feats, lt=thr)])
    ts.run(reference_data=ref[cols], current_data=cur[cols], column_mapping=cm)
    return {"drift_share": float(d["share_of_drifted_columns"]), "n_drifted": int(d["number_of_drifted_columns"]),
            "tests_passed": bool(ts.as_dict()["summary"]["all_passed"])}


def run(db: Path | None = None, out: Path | None = None) -> list[dict]:
    cfg = config.load()
    out = out or config.path("reports")
    (out / "windows").mkdir(parents=True, exist_ok=True)
    base = json.loads((config.path("reports") / "baseline.json").read_text())
    baseline, version = base["metrics_val"]["pr_auc"], base["champion_version"]
    log_df = store.read(db)
    feats = base["features"]
    n_ref = cfg["monitor"]["ref_days"]
    if n_ref:  # drift reference = first clean days of the stream (training period is itself shifted, see DECISIONS)
        r = log_df[log_df.day <= n_ref]
        ref = r[feats].assign(prediction=r["score"], target=r["label"].fillna(0).astype(int)).reset_index(drop=True)
    else:
        ref = _reference(feats, version)
    status, alerts = [], []
    for day, cur in log_df[log_df.day > n_ref].groupby("day"):
        lab = cur[cur.label.notna()]
        labelled = len(lab) == len(cur) and len(cur) > 0
        cur = cur.assign(prediction=cur["score"], target=cur["label"].astype("Int64").astype("float").fillna(-1).astype(int) if labelled else cur["label"])
        cur_eval = cur if labelled else cur.drop(columns="target")
        html = out / "windows" / f"window_{int(day):02d}.html"
        ev = _evidently(ref, cur_eval, feats, labelled, html)
        pr = float(average_precision_score(lab.label, lab.score)) if labelled and lab.label.sum() > 0 else None
        nf = int(lab.label.sum()) if labelled else 0
        dec = decide(ev["drift_share"], pr, nf, baseline)
        row = {"window": int(day), "n": len(cur), "n_fraud": nf, "labelled": labelled, **ev,
               "psi_score": psi(ref["prediction"].to_numpy(), cur["score"].to_numpy()),
               "mean_score": float(cur.score.mean()), "pr_auc": pr, "baseline_pr_auc": baseline,
               "model_versions": sorted(cur.model_version.unique().tolist()),
               "report": f"windows/{html.name}", **dec}
        status.append(row)
        if dec["alert"]:
            alerts.append({"window": int(day), "reasons": dec["reasons"], "drift_share": ev["drift_share"], "pr_auc": pr,
                           "labels_available_at_window": int(day) + cfg["label_lag_days"] if pr is not None else None})
            log.warning("ALERT window %s: %s", day, "; ".join(dec["reasons"]))
    (out / "status.json").write_text(json.dumps(status, indent=2))
    (out / "alerts.jsonl").write_text("".join(json.dumps(a) + "\n" for a in alerts))
    return status


