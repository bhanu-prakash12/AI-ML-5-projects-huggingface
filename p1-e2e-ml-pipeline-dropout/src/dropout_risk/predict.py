"""Pandas-free booster inference + pred_contrib explanations."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

TOP_K = 3


@dataclass
class Predictor:
    booster: lgb.Booster
    meta: dict[str, Any]

    @classmethod
    def load(cls, model_dir: Path) -> Predictor:
        meta = json.loads((model_dir / "preprocess.json").read_text())
        return cls(lgb.Booster(model_file=str(model_dir / "model.txt")), meta)

    @property
    def columns(self) -> list[str]:
        return self.meta["columns"]

    @property
    def version(self) -> str:
        return self.meta["model_version"]

    @property
    def threshold(self) -> float:
        return float(self.meta["threshold"]["value"])

    def tier(self, p: float) -> str:
        """high: at/above the tuned threshold; medium: above half of it; else low."""
        if p >= self.threshold:
            return "high"
        return "medium" if p >= self.threshold / 2 else "low"

    def predict(self, rows: list[dict[str, float]]) -> list[dict[str, Any]]:
        """Score rows (dicts keyed by feature name). Contributions are log-odds (TreeSHAP)."""
        x = np.array([[float(r[c]) for c in self.columns] for r in rows], dtype="float64")
        probs = self.booster.predict(x)
        contrib = self.booster.predict(x, pred_contrib=True)[:, :-1]  # drop bias column
        out = []
        for p, c in zip(probs, contrib, strict=True):
            top = np.argsort(-np.abs(c))[:TOP_K]
            out.append({
                "probability": float(p),
                "risk_tier": self.tier(float(p)),
                "top_features": [{"feature": self.columns[i], "contribution": float(c[i]),
                                  "direction": "increases_risk" if c[i] > 0 else "decreases_risk"}
                                 for i in top],
                "model_version": self.version,
            })
        return out
