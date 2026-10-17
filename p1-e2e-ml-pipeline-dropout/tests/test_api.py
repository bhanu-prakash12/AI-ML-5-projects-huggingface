import json

import lightgbm as lgb
import numpy as np
import pytest
from fastapi.testclient import TestClient

from dropout_risk import features
from dropout_risk.api import create_app
from dropout_risk.data import RANGES
from dropout_risk.samples import SAMPLES


def _row(i: int = 0) -> dict[str, float]:
    return {c: float(RANGES[c][0]) for c in features.FEATURES} | {
        "curricular_units_2nd_sem_approved": float(i % 6)}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    d = tmp_path_factory.mktemp("model")
    rng = np.random.default_rng(42)
    x = rng.random((300, len(features.FEATURES)))
    k = features.FEATURES.index("curricular_units_2nd_sem_approved")
    x[:, k] = np.arange(300) % 6
    y = (x[:, k] < 2).astype(int)
    b = lgb.train({"objective": "binary", "verbose": -1, "seed": 42, "min_data_in_leaf": 5},
                  lgb.Dataset(x, y), num_boost_round=20)
    b.save_model(str(d / "model.txt"))
    (d / "preprocess.json").write_text(json.dumps(features.build_preprocess(
        {"value": 0.5, "fallback": False}, "dropout-risk-vtest")))
    with TestClient(create_app(d)) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok", "model_version": "dropout-risk-vtest"}


def test_predict_shape(client):
    r = client.post("/predict", json=_row(0))
    assert r.status_code == 200
    j = r.json()
    assert 0 <= j["probability"] <= 1 and j["risk_tier"] in {"low", "medium", "high"}
    assert len(j["top_features"]) == 3 and j["model_version"] == "dropout-risk-vtest"


def test_risk_tiers_ordered(client):
    hi = client.post("/predict", json=_row(0)).json()
    lo = client.post("/predict", json=_row(5)).json()
    assert hi["probability"] > lo["probability"]
    assert hi["risk_tier"] == "high" and lo["risk_tier"] == "low"


def test_out_of_range_422(client):
    bad = _row(0) | {"debtor": 7}
    assert client.post("/predict", json=bad).status_code == 422


def test_missing_field_422(client):
    row = _row(0)
    row.pop("gdp")
    assert client.post("/predict", json=row).status_code == 422


def test_batch(client):
    r = client.post("/predict/batch", json={"rows": [_row(i) for i in range(5)]})
    assert r.status_code == 200 and len(r.json()["predictions"]) == 5


def test_batch_limits(client):
    assert client.post("/predict/batch", json={"rows": [_row()] * 100}).status_code == 200
    assert client.post("/predict/batch", json={"rows": [_row()] * 101}).status_code == 422
    assert client.post("/predict/batch", json={"rows": []}).status_code == 422


def test_openapi_has_example_and_summaries(client):
    spec_ = client.get("/openapi.json").json()
    assert spec_["info"]["description"]
    post = spec_["paths"]["/predict"]["post"]
    assert post["summary"]
    schema = spec_["components"]["schemas"]["StudentFeatures"]
    example = schema["examples"][0]
    assert client.post("/predict", json=example).status_code == 200


@pytest.mark.parametrize("name", ["low", "medium", "high"])
def test_sample_payloads_accepted(client, name):
    assert client.post("/predict", json=SAMPLES[name]).status_code == 200
