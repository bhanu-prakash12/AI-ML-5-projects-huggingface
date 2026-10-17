import pytest
from fastapi.testclient import TestClient

from fraud_monitor import api, config, store


@pytest.fixture
def client(tmp_path, monkeypatch, tiny):
    # tiny in-memory model, temp DB; no MLflow
    from fraud_monitor import data, model

    f = data.feature_cols(tiny)
    m = model.fit(tiny[f], tiny.Class)
    monkeypatch.setattr(api.model, "load_booster", lambda: (m.booster_, "T1"))
    monkeypatch.setattr(config, "path", lambda k: tmp_path / "p.db" if k == "db" else tmp_path)
    return TestClient(api.app), f, tiny


def test_predict_logs_and_feedback(client):
    with client[0] as c:
        _, f, df = client
        row = {k: float(df.iloc[0][k]) for k in f}
        assert c.get("/health").json() == {"status": "ok", "model_version": "T1"}
        r = c.post("/predict", json={"request_id": "r1", "features": row, "day": 2, "ts": "2026-01-02T00:00:00"}).json()
        assert r["request_id"] == "r1" and 0 <= r["score"] <= 1 and r["model_version"] == "T1"
        assert c.post("/feedback", json={"request_id": "r1", "label": 1}).json() == {"matched": 1}
        assert c.post("/feedback", json={"request_id": "nope", "label": 1}).status_code == 404
        log = store.read()
        assert len(log) == 1 and log.loc[0, "label"] == 1 and log.loc[0, "day"] == 2 and log.loc[0, "V1"] == row["V1"]


def test_validation_and_batch(client):
    with client[0] as c:
        _, f, df = client
        assert c.post("/predict", json={"features": {"V1": 1.0}}).status_code == 422
        items = [{"features": {k: float(df.iloc[i][k]) for k in f}, "day": 1} for i in range(5)]
        assert len(c.post("/predict/batch", json={"items": items}).json()) == 5
        assert len(store.read()) == 5
        assert c.post("/feedback", json={"request_id": "x", "label": 7}).status_code == 422


def test_openapi_examples_validate_and_cover_all_features(client, monkeypatch, tmp_path):
    from fraud_monitor import examples

    _, f, df = client
    monkeypatch.setattr(examples, "feature_names", lambda: tuple(f))
    examples.sample_rows.cache_clear()
    monkeypatch.setattr(examples.config, "path", lambda k: tmp_path / k)  # no parquet -> zero-valued rows
    schema = client[0].get("/openapi.json").json()
    ex = schema["components"]["schemas"]["PredictIn"]["examples"]
    assert len(ex) == 2
    for e in ex:
        assert set(e["features"]) == set(f)
        api.PredictIn.model_validate(e)
    assert schema["info"]["description"] and all("summary" in op for p in schema["paths"].values() for op in p.values())
    examples.sample_rows.cache_clear()
