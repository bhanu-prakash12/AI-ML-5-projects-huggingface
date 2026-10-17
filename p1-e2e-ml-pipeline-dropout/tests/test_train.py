import numpy as np
import pandas as pd

from dropout_risk import fairness, features
from dropout_risk.data import RANGES
from dropout_risk.train import choose_threshold, fit, grid


def test_threshold_meets_precision_floor():
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.4 + rng.random(500) * 0.6, 0, 1)
    t = choose_threshold(y, p, 0.6)
    assert not t["fallback"]
    pred = p >= t["value"]
    assert (y[pred] == 1).mean() >= 0.6


def test_threshold_fallback_when_infeasible():
    y = np.array([0, 1, 0, 1, 0, 0])
    p = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.6])  # anti-correlated
    assert choose_threshold(y, p, 0.99)["fallback"] is True


def test_grid_capped_at_12():
    assert len(grid(12)) <= 12 and len(grid(3)) == 3


def test_sensitive_excluded_from_features():
    assert not set(features.SENSITIVE) & set(features.FEATURES)
    assert set(features.FEATURES) <= set(RANGES)


def _frame(n=200, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.integers(int(lo), int(min(hi, lo + 5)) + 1, n)
                       for c, (lo, hi) in RANGES.items()})
    df["dropout"] = (df["curricular_units_2nd_sem_approved"] < 2).astype(int)
    return df


def test_fit_and_fairness_slices():
    df = _frame()
    X, y = features.xy(df)
    m = fit("lightgbm", dict(num_leaves=7, learning_rate=0.1, n_estimators=20), X, y, 1, 42)
    p = m.predict_proba(X)[:, 1]
    assert ((p >= 0) & (p <= 1)).all()
    assert m.booster_.predict(X.head(2), pred_contrib=True).shape == (2, len(features.FEATURES) + 1)
    df["gender"] = np.arange(len(df)) % 2
    df["age_at_enrollment"] = 18 + np.arange(len(df)) % 20
    sl = fairness.slice_metrics(df, p, 0.5)
    assert set(sl["slice"]) == {"gender_name", "age_band"}
    assert "n<" not in fairness.to_markdown(sl[sl["n"] >= 30], 0.5)
