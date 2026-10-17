"""FastAPI serving: /predict (logged), /feedback (delayed labels), /health, /reload."""
from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from . import config, examples, model, store

log = logging.getLogger("fraud_monitor.api")
STATE: dict = {}


def _load() -> None:
    b, v = model.load_booster()
    STATE.update(booster=b, version=v, feats=list(b.feature_name()))
    log.info("loaded champion version %s", v)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load()
    yield


DESCRIPTION = """
Scores card transactions with the **champion** LightGBM model and logs every prediction to SQLite so the
monitoring job can compare windows later.

**Inputs:** `/predict` takes one transaction as a `features` map (V1..V28 + Amount, 29 numbers). The examples in
this page are real rows from the reference data (one legit, one fraud) - pick one in the *Try it out* dropdown.
**Outputs:** `score` (fraud probability 0-1), `is_fraud` (score >= threshold in `configs/config.yaml`) and the
model version that answered.

**Demo order:** `GET /health` -> `POST /predict` (copy the `request_id`) -> `POST /feedback` with the true label
(fraud labels arrive late in real life) -> look at the dashboard (`python run.py ui`).

curl / PowerShell: `Invoke-RestMethod http://127.0.0.1:8004/health` or `curl http://127.0.0.1:8004/health`.
"""

app = FastAPI(title="Fraud scoring", description=DESCRIPTION, lifespan=lifespan)


def _predict_in_extra(schema: dict, _cls: type) -> None:
    schema["examples"] = examples.predict_examples()  # evaluated lazily, when /openapi.json is built


def _feedback_extra(schema: dict, _cls: type) -> None:
    schema["examples"] = [{"request_id": e["request_id"], "label": 1 if "fraud" in e["request_id"] else 0}
                          for e in examples.predict_examples()]


class PredictIn(BaseModel):
    model_config = ConfigDict(json_schema_extra=_predict_in_extra)
    request_id: str | None = None
    features: dict[str, float]
    ts: str | None = Field(None, description="event time (ISO); defaults to now. The simulator sends simulated time")
    day: int = Field(0, ge=0, description="simulated day (monitoring window); 0 = real time")


class PredictOut(BaseModel):
    request_id: str = Field(description="echo of the request id (generated when omitted); use it for /feedback")
    score: float = Field(description="fraud probability, 0-1")
    is_fraud: bool = Field(description="score >= model.threshold from configs/config.yaml")
    model_version: str = Field(description="MLflow registry version of the champion that scored this row")


class BatchIn(BaseModel):
    items: list[PredictIn] = Field(max_length=2000)


class FeedbackIn(BaseModel):
    model_config = ConfigDict(json_schema_extra=_feedback_extra)
    request_id: str = Field(description="request_id returned by /predict")
    label: int = Field(ge=0, le=1, description="true outcome: 1 = fraud, 0 = legit")


class HealthOut(BaseModel):
    status: str
    model_version: str | None = None


class FeedbackOut(BaseModel):
    matched: int = Field(description="number of logged predictions the label was joined to")


class FeedbackBatchOut(FeedbackOut):
    received: int


class ReloadOut(BaseModel):
    model_version: str


class ClearOut(BaseModel):
    status: str


class FeedbackBatch(BaseModel):
    items: list[FeedbackIn] = Field(max_length=5000)


def _score(items: list[PredictIn]) -> list[PredictOut]:
    feats = STATE["feats"]
    for it in items:
        missing = [f for f in feats if f not in it.features]
        if missing:
            raise HTTPException(422, f"missing features: {missing[:5]}")
    X = pd.DataFrame([[it.features[f] for f in feats] for it in items], columns=feats)
    scores = STATE["booster"].predict(X)
    now = datetime.now(timezone.utc).isoformat()
    outs, rows = [], []
    thr = config.load()["model"]["threshold"]
    for it, s in zip(items, scores):
        rid = it.request_id or uuid.uuid4().hex
        outs.append(PredictOut(request_id=rid, score=float(s), is_fraud=bool(s >= thr), model_version=STATE["version"]))
        rows.append((rid, it.ts or now, it.day, STATE["version"], float(s), {f: it.features[f] for f in feats}))
    store.log_predictions(rows)
    return outs


@app.get("/health", response_model=HealthOut, summary="Liveness + which model version is serving")
def health():
    return {"status": "ok", "model_version": STATE.get("version")}


@app.post("/predict", response_model=PredictOut, summary="Score one transaction (logged)")
def predict(body: PredictIn):
    return _score([body])[0]


@app.post("/predict/batch", response_model=list[PredictOut], summary="Score up to 2000 transactions (used by the simulator)")
def predict_batch(body: BatchIn):
    return _score(body.items)


@app.post("/feedback", response_model=FeedbackOut, summary="Attach the true label to a logged prediction")
def feedback(body: FeedbackIn):
    n = store.log_labels([(body.request_id, body.label, datetime.now(timezone.utc).isoformat())])
    if not n:
        raise HTTPException(404, "unknown request_id")
    return {"matched": n}


@app.post("/feedback/batch", response_model=FeedbackBatchOut, summary="Attach many labels at once")
def feedback_batch(body: FeedbackBatch):
    ts = datetime.now(timezone.utc).isoformat()
    return {"matched": store.log_labels([(i.request_id, i.label, ts) for i in body.items]), "received": len(body.items)}


@app.post("/reload", response_model=ReloadOut, summary="Reload the champion alias without restarting")
def reload():
    """Pick up a newly promoted champion without restarting."""
    _load()
    return {"model_version": STATE["version"]}


@app.delete("/predictions", response_model=ClearOut, summary="Demo only: wipe the prediction log")
def reset_log():
    """Demo-only: wipe the prediction log before a fresh simulation."""
    store.reset()
    return {"status": "cleared"}
