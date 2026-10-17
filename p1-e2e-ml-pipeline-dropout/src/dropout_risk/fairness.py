"""Metric slices by gender and age band."""
from __future__ import annotations

import numpy as np
import pandas as pd

from dropout_risk.data import BINARY_TARGET
from dropout_risk.features import age_band

MIN_GROUP_N = 30


def slice_metrics(df: pd.DataFrame, proba: np.ndarray, thr: float) -> pd.DataFrame:
    d = df.assign(
        pred=(proba >= thr).astype(int),
        gender_name=df["gender"].map({0: "female", 1: "male"}),
        age_band=age_band(df["age_at_enrollment"]),
    )
    rows = []
    for col in ("gender_name", "age_band"):
        for grp, g in d.groupby(col):
            y, p = g[BINARY_TARGET], g["pred"]
            tp = int(((y == 1) & (p == 1)).sum())
            fp = int(((y == 0) & (p == 1)).sum())
            fn = int(((y == 1) & (p == 0)).sum())
            rows.append({
                "slice": col, "group": grp, "n": len(g),
                "dropout_rate": round(float(y.mean()), 3),
                "flag_rate": round(float(p.mean()), 3),
                "recall": round(tp / (tp + fn), 3) if tp + fn else None,
                "precision": round(tp / (tp + fp), 3) if tp + fp else None,
                "reported": len(g) >= MIN_GROUP_N,
            })
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame, thr: float) -> str:
    out = ["# Fairness slices (test split)", "",
           f"Threshold {thr:.3f}. Sensitive attributes are NOT model inputs (D3); "
           f"groups with n < {MIN_GROUP_N} are suppressed.", ""]
    cols = ["slice", "group", "n", "dropout_rate", "flag_rate", "recall", "precision"]
    out += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        if r.reported:
            out.append("| " + " | ".join(str(getattr(r, c)) for c in cols) + " |")
        else:
            out.append(f"| {r.slice} | {r.group} | {r.n} | suppressed (n<{MIN_GROUP_N}) | | | |")
    out += ["", "Large recall/flag-rate gaps between groups indicate disparate impact; "
            "dropout_rate differences are base-rate differences, not model error.", ""]
    return "\n".join(out)
