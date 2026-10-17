"""Champion/challenger: train on reference + recent labelled traffic, compare on the latest labelled window."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from . import config, data, model, store

log = logging.getLogger("fraud_monitor.retrain")


def decide(champ_pr: float, chall_pr: float, champ_guard: float, chall_guard: float) -> tuple[bool, str]:
    """Promote only if the challenger is better on the latest window by min_improvement AND does not
    regress by more than 0.05 PR-AUC on the old-pattern validation set (guardrail against forgetting)."""
    gain = config.load()["retrain"]["min_improvement"]
    if chall_pr < champ_pr + gain:
        return False, f"window PR-AUC {chall_pr:.3f} < champion {champ_pr:.3f} + {gain}"
    if chall_guard < champ_guard - 0.05:
        return False, f"regression on old-pattern validation {chall_guard:.3f} vs {champ_guard:.3f}"
    return True, f"window PR-AUC {chall_pr:.3f} vs champion {champ_pr:.3f}"


def run(db: Path | None = None, out: Path | None = None, promote_alias: bool = True) -> dict:
    cfg = config.load()
    out = out or config.path("reports")
    ref = pd.read_parquet(config.path("data") / "reference.parquet")
    feats = data.feature_cols(ref)
    n_val = int(len(ref) * cfg["model"]["val_frac"])
    tr, va = ref.iloc[:-n_val], ref.iloc[-n_val:]
    lg = store.read(db)
    lg = lg[lg.label.notna()]
    days = sorted(lg.day.unique())
    latest, past = days[-1], lg[lg.day < days[-1]]
    cur = lg[lg.day == latest]
    X = pd.concat([tr[feats], past[feats]])
    y = pd.concat([tr.Class, past.label.astype(int)])
    chall = model.fit(X, y)
    champ, champ_v = model.load_booster()
    yc = cur.label.astype(int).to_numpy()
    m_champ = model.metrics(yc, champ.predict(cur[feats]))     # re-score so versions are comparable
    m_chall = model.metrics(yc, chall.predict_proba(cur[feats])[:, 1])
    g_champ = model.metrics(va.Class.to_numpy(), champ.predict(va[feats]))["pr_auc"]
    g_chall = model.metrics(va.Class.to_numpy(), chall.predict_proba(va[feats])[:, 1])["pr_auc"]
    ok, why = decide(m_champ["pr_auc"], m_chall["pr_auc"], g_champ, g_chall)
    model.setup_mlflow()
    with mlflow.start_run(run_name="challenger"):
        mlflow.log_params({"trained_on_windows": f"1..{latest - 1}" if latest > 1 else "none", "eval_window": int(latest),
                           "n_train": len(X), "champion_version": champ_v})
        mlflow.log_metrics({"window_pr_auc": m_chall["pr_auc"], "champion_window_pr_auc": m_champ["pr_auc"],
                            "val_pr_auc": g_chall, "champion_val_pr_auc": g_champ, "promoted": float(ok)})
        mlflow.set_tag("decision", why)
        info = mlflow.lightgbm.log_model(chall.booster_, "model", registered_model_name=cfg["model"]["name"])
    ver = str(info.registered_model_version)
    promoted = ok and promote_alias
    if promoted:
        model.set_champion(ver)
    res = {"eval_window": int(latest), "n_new_labelled_rows": len(past), "n_new_frauds": int(past.label.sum()),
           "champion": {"version": champ_v, "window": m_champ, "val_pr_auc": g_champ},
           "challenger": {"version": ver, "window": m_chall, "val_pr_auc": g_chall},
           "promoted": ok, "alias_updated": promoted, "reason": why,
           "new_champion_version": ver if promoted else champ_v}
    (out / "retrain.json").write_text(json.dumps(res, indent=2, default=lambda o: None if np.isnan(o) else o))
    log.info("retrain: promoted=%s (%s)", ok, why)
    return res
