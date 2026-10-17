"""In-process Lambda handler tests: sample API Gateway HTTP v2 event, no emulator, no Docker."""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest

from dropout_risk import features
from dropout_risk.lambda_handler import build_handler

EVENT = Path(__file__).parent / "events" / "predict.json"


@pytest.fixture(scope="module")
def handler(tmp_path_factory):
    d = tmp_path_factory.mktemp("model")
    rng = np.random.default_rng(42)
    x = rng.random((300, len(features.FEATURES)))
    y = (x[:, 0] > 0.5).astype(int)
    b = lgb.train({"objective": "binary", "verbose": -1, "seed": 42, "min_data_in_leaf": 5},
                  lgb.Dataset(x, y), num_boost_round=10)
    b.save_model(str(d / "model.txt"))
    (d / "preprocess.json").write_text(json.dumps(features.build_preprocess(
        {"value": 0.5, "fallback": False}, "dropout-risk-vtest")))
    return build_handler(d)


def test_predict_event(handler):
    res = handler(json.loads(EVENT.read_text()), None)
    assert res["statusCode"] == 200
    body = json.loads(res["body"])
    assert 0 <= body["probability"] <= 1
    assert body["risk_tier"] in {"low", "medium", "high"}
    assert len(body["top_features"]) == 3 and body["model_version"] == "dropout-risk-vtest"


def test_health_event(handler):
    ev = json.loads(EVENT.read_text()) | {"rawPath": "/health", "routeKey": "GET /health", "body": None}
    ev["requestContext"]["http"] |= {"method": "GET", "path": "/health"}
    res = handler(ev, None)
    assert res["statusCode"] == 200 and json.loads(res["body"])["status"] == "ok"


def test_out_of_range_is_422(handler):
    ev = json.loads(EVENT.read_text())
    body = json.loads(ev["body"]) | {features.FEATURES[0]: 1e9}
    ev["body"] = json.dumps(body)
    assert handler(ev, None)["statusCode"] == 422
