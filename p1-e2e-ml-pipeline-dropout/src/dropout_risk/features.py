"""Column order, encodings, preprocess.json export."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dropout_risk.data import BINARY_TARGET, RANGES, TARGET

# D3: sensitive / protected attributes are excluded from the model; they stay in the
# splits so fairness.py can slice metrics by them.
SENSITIVE: tuple[str, ...] = (
    "gender", "age_at_enrollment", "nacionality", "international", "marital_status",
    "educational_special_needs",
)
FEATURES: list[str] = [c for c in RANGES if c not in SENSITIVE]
AGE_BINS = [0, 19, 24, 29, 200]
AGE_LABELS = ["<20", "20-24", "25-29", "30+"]


def xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Model matrix (float64, fixed column order) and binary label."""
    return df[FEATURES].astype("float64"), df[BINARY_TARGET].astype(int)


def age_band(age: pd.Series) -> pd.Series:
    return pd.cut(age, bins=AGE_BINS, labels=AGE_LABELS).astype(str)


def build_preprocess(threshold: dict, model_version: str) -> dict:
    """Everything serving needs besides model.txt (no pandas/sklearn/MLflow at serve time)."""
    return {
        "model_version": model_version,
        "columns": FEATURES,
        "ranges": {c: list(RANGES[c]) for c in FEATURES},
        "encodings": "none: integer category codes are passed through as numeric",
        "excluded_sensitive": list(SENSITIVE),
        "target": BINARY_TARGET,
        "source_target": TARGET,
        "threshold": threshold,
    }


def write_preprocess(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
