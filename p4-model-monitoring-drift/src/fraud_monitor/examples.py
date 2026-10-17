"""Example /predict payloads derived from the model's feature list and real reference rows (no invented values)."""
from __future__ import annotations

import json
from functools import lru_cache

import pandas as pd

from . import config

DEFAULT_FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]  # documented in DECISIONS.md; last-resort fallback


@lru_cache(maxsize=1)
def feature_names() -> tuple[str, ...]:
    """Feature list the champion expects: reports/baseline.json (written at training), else the live booster."""
    p = config.path("reports") / "baseline.json"
    try:
        return tuple(json.loads(p.read_text())["features"])
    except (OSError, KeyError, ValueError):
        pass
    try:
        from . import model

        return tuple(model.load_booster()[0].feature_name())
    except Exception:  # noqa: BLE001 - no model yet: schema-only fallback
        return tuple(DEFAULT_FEATURES)


@lru_cache(maxsize=1)
def sample_rows() -> dict[str, dict[str, float]]:
    """One real legit row and one real fraud row from data/reference.parquet (zeros if the file is absent)."""
    feats = list(feature_names())
    p = config.path("data") / "reference.parquet"
    try:
        df = pd.read_parquet(p)
        out = {}
        for name, cls in (("legit", 0), ("fraud", 1)):
            r = df[df.Class == cls].iloc[0]
            out[name] = {f: float(r[f]) for f in feats}
        return out
    except Exception:  # noqa: BLE001 - parquet missing: same structure, zero values
        return {"legit": {f: 0.0 for f in feats}, "fraud": {f: 0.0 for f in feats}}


def predict_examples() -> list[dict]:
    rows = sample_rows()
    return [{"request_id": f"demo-{k}-1", "features": v} for k, v in rows.items()]
