"""LightGBM/XGBoost training, MLflow logging, threshold selection."""
from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)

from dropout_risk import fairness  # noqa: E402
from dropout_risk.config import ROOT, load_config, project_path  # noqa: E402
from dropout_risk.features import FEATURES, build_preprocess, write_preprocess, xy  # noqa: E402

log = logging.getLogger("train")


def choose_threshold(y: np.ndarray, p: np.ndarray, min_precision: float) -> dict:
    """Max recall subject to precision >= min_precision; fallback to best-F1."""
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]  # align with thresholds
    ok = prec >= min_precision
    if ok.any():
        i = int(np.argmax(np.where(ok, rec, -1)))
        return {"value": float(thr[i]), "rule": f"max recall at precision>={min_precision}",
                "fallback": False}
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    i = int(np.argmax(f1))
    return {"value": float(thr[i]), "rule": "best F1 (precision floor infeasible)", "fallback": True}


def metrics_at(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    pred = (p >= thr).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    return {
        "roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
        "f1": float(f1_score(y, pred)), "recall": float(recall_score(y, pred)),
        "precision": float(tp / max(int(pred.sum()), 1)), "base_rate": float(y.mean()),
    }


def grid(max_runs: int) -> list[tuple[str, dict]]:
    lgbm = [("lightgbm", dict(num_leaves=nl, learning_rate=lr, n_estimators=n))
            for nl, lr, n in itertools.product((15, 31), (0.03, 0.1), (200, 400))][:8]
    xg = [("xgboost", dict(max_depth=d, learning_rate=lr, n_estimators=300))
          for d, lr in itertools.product((3, 6), (0.03, 0.1))]
    return (lgbm + xg)[:max_runs]


def fit(kind: str, params: dict, Xtr, ytr, n_jobs: int, seed: int):
    spw = float((ytr == 0).sum() / (ytr == 1).sum())  # class weight for imbalance
    if kind == "lightgbm":
        m = lgb.LGBMClassifier(**params, scale_pos_weight=spw, n_jobs=n_jobs,
                               random_state=seed, verbose=-1)
    else:
        m = xgb.XGBClassifier(**params, scale_pos_weight=spw, n_jobs=n_jobs, random_state=seed,
                              eval_metric="logloss", tree_method="hist")
    return m.fit(Xtr, ytr)


def _plots(y, p, thr, model, out: Path) -> tuple[Path, Path]:
    cm = out / "confusion_matrix.png"
    ConfusionMatrixDisplay(confusion_matrix(y, (p >= thr).astype(int)),
                           display_labels=["stay", "dropout"]).plot(values_format="d")
    plt.title("Test confusion matrix")
    plt.savefig(cm, dpi=120, bbox_inches="tight")
    plt.close()
    imp = pd.Series(model.booster_.feature_importance("gain"), index=FEATURES).nlargest(12)[::-1]
    fi = out / "feature_importance.png"
    imp.plot.barh(figsize=(7, 5), title="Top-12 features (gain)")
    plt.savefig(fi, dpi=120, bbox_inches="tight")
    plt.close()
    return cm, fi


def _tracking() -> None:
    cfg = load_config()["mlflow"]
    uri = cfg["tracking_uri"]
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        uri = f"sqlite:///{ROOT / uri[len('sqlite:///'):]}"
    mlflow.set_tracking_uri(uri)
    art = (ROOT / cfg["artifact_root"]).as_uri()
    if mlflow.get_experiment_by_name(cfg["experiment"]) is None:
        mlflow.create_experiment(cfg["experiment"], artifact_location=art)
    mlflow.set_experiment(cfg["experiment"])


def main() -> dict:
    cfg = load_config()
    seed, nj = cfg["seed"], cfg["runtime"]["n_jobs"]
    proc = project_path("data") / "processed"
    tr, va, te = (pd.read_parquet(proc / f"{s}.parquet") for s in ("train", "val", "test"))
    Xtr, ytr = xy(tr)
    Xva, yva = xy(va)
    Xte, yte = xy(te)
    reports, arts = project_path("reports"), project_path("artifacts")
    minp = cfg["model"]["threshold_min_precision"]
    _tracking()
    results = []
    for kind, params in grid(cfg["model"]["max_grid_runs"]):
        with mlflow.start_run(run_name=f"{kind}-{len(results)}") as run:
            model = fit(kind, params, Xtr, ytr, nj, seed)
            pva = model.predict_proba(Xva)[:, 1]
            thr = choose_threshold(yva.to_numpy(), pva, minp)
            m = metrics_at(yva.to_numpy(), pva, thr["value"])
            mlflow.log_params({"model": kind, **params, "seed": seed, "threshold": thr["value"]})
            mlflow.log_metrics({f"val_{k}": v for k, v in m.items()})
            results.append({"kind": kind, "params": params, "model": model, "thr": thr,
                            "val": m, "run_id": run.info.run_id})
            log.info("%s %s val pr_auc=%.3f", kind, params, m["pr_auc"])
    best = max((r for r in results if r["kind"] == "lightgbm"), key=lambda r: r["val"]["pr_auc"])
    model, thr = best["model"], best["thr"]
    pte = model.predict_proba(Xte)[:, 1]
    test = metrics_at(yte.to_numpy(), pte, thr["value"])
    cm, fi = _plots(yte.to_numpy(), pte, thr["value"], model, reports)
    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.log_metrics({f"test_{k}": v for k, v in test.items()})
        mlflow.log_artifact(str(cm))
        mlflow.log_artifact(str(fi))
        sig = infer_signature(Xtr.head(50), model.predict_proba(Xtr.head(50))[:, 1])
        info = mlflow.lightgbm.log_model(
            model.booster_, "model", signature=sig, input_example=Xtr.head(3),
            registered_model_name=cfg["mlflow"]["model_name"])
    client = mlflow.MlflowClient()
    version = str(info.registered_model_version)
    client.set_registered_model_alias(cfg["mlflow"]["model_name"], cfg["mlflow"]["alias"], version)
    # serving export
    (arts / "model").mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(arts / "model" / "model.txt"))
    write_preprocess(build_preprocess(thr, f"{cfg['mlflow']['model_name']}-v{version}"),
                     arts / "model" / "preprocess.json")
    summary = {
        "champion": {"kind": best["kind"], "params": best["params"], "registry_version": version,
                     "run_id": best["run_id"]},
        "threshold": thr, "val": best["val"], "test": test,
        "comparison": [{"kind": r["kind"], "params": r["params"], "val_pr_auc": r["val"]["pr_auc"],
                        "val_roc_auc": r["val"]["roc_auc"]} for r in results],
    }
    (reports / "metrics.json").write_text(json.dumps(summary, indent=2))
    sl = fairness.slice_metrics(te, pte, thr["value"])
    (reports / "fairness.md").write_text(fairness.to_markdown(sl, thr["value"]))
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
