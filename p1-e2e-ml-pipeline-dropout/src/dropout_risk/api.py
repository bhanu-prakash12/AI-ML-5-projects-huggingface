"""FastAPI app: /health /predict /predict/batch."""
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field, create_model

from dropout_risk.config import load_config, project_path
from dropout_risk.predict import Predictor
from dropout_risk.samples import SAMPLES
from dropout_risk.schemas import BatchResponse, Health, Prediction, student_model


DESCRIPTION = """
Predicts a student's **dropout probability** and a **risk tier** (low / medium / high), plus the
**top 3 SHAP contributions** (log-odds) explaining the score.

**Input:** `/predict` takes one student (all model features, validated ranges; 422 if out of
range). `/predict/batch` takes 1..N students. **Output:** probability, tier, top features.

**Demo:** open *POST /predict* -> *Try it out* (the example body is a high-risk student) ->
*Execute*. Late-stage model (end of year 2), not an early-warning system.
"""


def create_app(model_dir: Path | None = None) -> FastAPI:
    """App factory; the model is loaded once in the lifespan hook."""
    cfg = load_config()
    max_batch = cfg["api"]["max_batch"]

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predictor = Predictor.load(model_dir or project_path("artifacts") / "model")
        yield

    app = FastAPI(title="Student Dropout Risk", version="1", description=DESCRIPTION,
                  lifespan=lifespan)
    # Request models need the model's ranges, so read preprocess.json up front.
    d = model_dir or project_path("artifacts") / "model"
    meta: dict[str, Any] = json.loads((d / "preprocess.json").read_text())
    example = {c: SAMPLES["high"][c] for c in meta["columns"] if c in SAMPLES["high"]}
    Student = student_model(meta, example if len(example) == len(meta["columns"]) else None)
    Batch: type[BaseModel] = create_model(
        "BatchRequest",
        rows=(list[Student], Field(min_length=1, max_length=max_batch,  # type: ignore[valid-type]
                                   description=f"1..{max_batch} students")))

    @app.get("/health", response_model=Health, summary="Liveness + model version")
    def health(request: Request) -> Health:
        return Health(status="ok", model_version=request.app.state.predictor.version)

    @app.post("/predict", response_model=Prediction,
              summary="Score one student (probability, tier, top-3 SHAP)")
    def predict(student: Student, request: Request) -> dict:  # type: ignore[valid-type]
        return request.app.state.predictor.predict([student.model_dump()])[0]

    @app.post("/predict/batch", response_model=BatchResponse,
              summary="Score 1..max_batch students")
    def predict_batch(body: Batch, request: Request) -> dict:  # type: ignore[valid-type]
        rows = [r.model_dump() for r in body.rows]  # type: ignore[attr-defined]
        return {"predictions": request.app.state.predictor.predict(rows)}

    return app


def _default_app() -> FastAPI:
    return create_app()


app = _default_app() if (project_path("artifacts") / "model" / "preprocess.json").exists() \
    else FastAPI(title="Student Dropout Risk (model not trained: run `make train`)")
