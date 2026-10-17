import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import simulate
from fraud_monitor import api, config, data, model, scenarios, store


@pytest.fixture
def setup(tmp_path, monkeypatch, tiny):
    f = data.feature_cols(tiny)
    m = model.fit(tiny[f], tiny.Class)
    monkeypatch.setattr(api.model, "load_booster", lambda: (m.booster_, "T1"))
    real = config.path
    monkeypatch.setattr(config, "path", lambda k: tmp_path / "p.db" if k == "db" else real(k))
    stream = data.make_stream(tiny)
    return stream, tiny, f


def test_scenario_B_adds_frauds_only_after_start(setup):
    stream, ref, f = setup
    d = stream[stream.day == 8].reset_index(drop=True)
    assert scenarios.apply(d, 3, ref, f, "B", 6).equals(d)
    o = scenarios.apply(d, 8, ref, f, "B", 6)
    assert o.Class.sum() > d.Class.sum() and not np.allclose(o[f].to_numpy(), d[f].to_numpy())


def test_scenarios_A_C(setup):
    stream, ref, f = setup
    d = stream[stream.day == 8].reset_index(drop=True)
    assert scenarios.apply(d, 8, ref, f, "A", 6).Amount.mean() > 2 * d.Amount.mean()
    assert scenarios.apply(d, 8, ref, f, "C", 6).Class.sum() == d.Class.sum()
    with pytest.raises(ValueError):
        scenarios.apply(d, 8, ref, f, "Z", 1)


def test_replay_logs_all_days_with_lagged_labels(setup):
    stream, ref, f = setup
    with TestClient(api.app) as c:
        res = simulate.run(c, stream, ref, f, "B", 6, reset=True)
        log = store.read()
    assert res["sent"] == len(stream) == len(log)
    assert set(log.day) == set(range(1, 11))
    last = log[log.day == 10]
    assert last.label.isna().all() and log[log.day == 9].label.notna().all()  # lag 1 day
    assert isinstance(pd.Timestamp(log.ts.iloc[0]), pd.Timestamp)
