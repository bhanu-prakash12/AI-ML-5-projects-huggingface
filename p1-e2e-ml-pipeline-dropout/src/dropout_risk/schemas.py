"""Pydantic request/response models (request schema derived from preprocess.json ranges)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model


class Contribution(BaseModel):
    feature: str = Field(description="Model feature name")
    contribution: float = Field(description="TreeSHAP contribution in log-odds units")
    direction: Literal["increases_risk", "decreases_risk"]


class Prediction(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{
        "probability": 0.87, "risk_tier": "high",
        "top_features": [{"feature": "curricular_units_2nd_sem_approved", "contribution": 1.1,
                          "direction": "increases_risk"}],
        "model_version": "dropout-risk-v2"}]})

    probability: float = Field(ge=0, le=1, description="P(dropout)")
    risk_tier: Literal["low", "medium", "high"] = Field(
        description="high: p >= tuned threshold; medium: p >= threshold/2; else low")
    top_features: list[Contribution] = Field(description="Top 3 features by |contribution|")
    model_version: str


class BatchResponse(BaseModel):
    predictions: list[Prediction]


class Health(BaseModel):
    status: Literal["ok"]
    model_version: str


def student_model(meta: dict[str, Any],
                  example: dict[str, float] | None = None) -> type[BaseModel]:
    """Build the request model from the validated column ranges (422 on out-of-range)."""
    fields: dict[str, Any] = {
        c: (float, Field(ge=lo, le=hi, description=f"{c.replace('_', ' ')} (valid {lo}..{hi})"))
        for c, (lo, hi) in meta["ranges"].items()
    }
    cfg = ConfigDict(json_schema_extra={"examples": [example]}) if example else None
    return create_model("StudentFeatures", __config__=cfg, **fields)
