import numpy as np

from fraud_monitor import monitor


def test_psi_zero_for_same_and_large_for_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    assert monitor.psi(a, rng.normal(size=5000)) < 0.05
    assert monitor.psi(a, rng.normal(2, 1, 5000)) > 0.25


def test_decide_rules():
    ok = monitor.decide(0.1, 0.80, 20, 0.80)
    assert not ok["alert"]
    assert monitor.decide(0.4, 0.80, 20, 0.80)["alert"]                       # drift share rule
    d = monitor.decide(0.1, 0.60, 20, 0.80)                                   # 25% PR-AUC drop
    assert d["alert"] and "pr_auc" in d["reasons"][0]
    s = monitor.decide(0.1, 0.30, 4, 0.80)                                    # too few frauds -> guard
    assert not s["alert"] and "labelled frauds" in s["perf_skipped"]
    assert monitor.decide(0.1, None, 0, 0.8)["perf_skipped"] == "labels pending"
