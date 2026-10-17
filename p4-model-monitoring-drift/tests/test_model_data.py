import numpy as np

from fraud_monitor import data, model


def test_make_stream_keeps_all_frauds_and_days(tiny):
    s = data.make_stream(tiny)
    assert s.Class.sum() == tiny.Class.sum()
    assert s.day.min() == 1 and s.day.max() == 10
    assert s.day.is_monotonic_increasing


def test_split_is_time_ordered(tiny):
    ref, stream = data.split(tiny)
    assert len(ref) == int(len(tiny) * 0.6)
    assert ref.index[-1] + 1 == len(ref)


def test_metrics_accuracy_useless_at_low_base_rate():
    y = np.zeros(1000, int)
    y[:2] = 1
    m = model.metrics(y, np.zeros(1000))
    assert m["accuracy"] > 0.99 and m["recall"] == 0 and m["pr_auc"] < 0.01


def test_fit_beats_chance(tiny):
    f = data.feature_cols(tiny)
    m = model.fit(tiny[f], tiny.Class)
    assert model.metrics(tiny.Class.to_numpy(), m.predict_proba(tiny[f])[:, 1])["pr_auc"] > 0.9
