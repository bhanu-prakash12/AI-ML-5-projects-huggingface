"""LightGBM fraud model, metrics and MLflow registry helpers (alias `champion`)."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

from . import config

log = logging.getLogger(__name__)


def sqlite_uri(p) -> str:
    """sqlite URI that also works for Windows drive paths (forward slashes)."""
    return f"sqlite:///{(p if hasattr(p, 'as_posix') else Path(p)).as_posix()}"


def setup_mlflow() -> MlflowClient:
    mlflow.set_tracking_uri(sqlite_uri(config.path("mlflow_db")))
    mlflow.set_experiment("fraud-monitoring")
    return MlflowClient()


def fit(X: pd.DataFrame, y: pd.Series) -> lgb.LGBMClassifier:
    cfg = config.load()
    m = lgb.LGBMClassifier(random_state=cfg["seed"], subsample_freq=1, **cfg["model"]["params"])
    return m.fit(X, y)


def metrics(y: np.ndarray, score: np.ndarray, thr: float | None = None) -> dict[str, float]:
    thr = config.load()["model"]["threshold"] if thr is None else thr
    pred = (score >= thr).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return {
        "pr_auc": float(average_precision_score(y, score)) if y.sum() else float("nan"),
        "roc_auc": float(roc_auc_score(y, score)) if 0 < y.sum() < len(y) else float("nan"),
        "precision": float(p), "recall": float(r), "f1": float(f),
        "accuracy": float((pred == y).mean()), "base_rate": float(y.mean()),
        "n": int(len(y)), "n_fraud": int(y.sum()),
    }


def register(model: lgb.LGBMClassifier, params: dict[str, Any], m: dict[str, float], tag: str) -> str:
    """Log to MLflow, register a new version of `fraud`; return the version string."""
    setup_mlflow()
    name = config.load()["model"]["name"]
    with mlflow.start_run(run_name=tag):
        mlflow.log_params({k: v for k, v in params.items() if k != "verbose"})
        mlflow.log_metrics({k: v for k, v in m.items() if v == v})
        info = mlflow.lightgbm.log_model(model.booster_, "model", registered_model_name=name)
        mlflow.set_tag("role", tag)
    return str(info.registered_model_version)


def set_champion(version: str) -> None:
    client = setup_mlflow()
    client.set_registered_model_alias(config.load()["model"]["name"], "champion", version)


def reset_champion() -> str:
    """Point the `champion` alias back at the baseline version recorded in reports/baseline.json."""
    v = str(json.loads((config.path("reports") / "baseline.json").read_text())["champion_version"])
    set_champion(v)
    return v


def champion_version() -> str:
    client = setup_mlflow()
    return str(client.get_model_version_by_alias(config.load()["model"]["name"], "champion").version)


def load_booster(version: str | None = None) -> tuple[lgb.Booster, str]:
    setup_mlflow()
    v = version or champion_version()
    b = mlflow.lightgbm.load_model(f"models:/{config.load()['model']['name']}/{v}")
    return b, v


def dump(path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2))
