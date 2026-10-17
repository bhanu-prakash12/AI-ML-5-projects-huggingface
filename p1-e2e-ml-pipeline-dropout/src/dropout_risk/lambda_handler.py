"""AWS Lambda entry point: Mangum adapts API Gateway / Function URL (HTTP v2) events to FastAPI."""
import os
from pathlib import Path

from mangum import Mangum

from dropout_risk.api import create_app
from dropout_risk.config import ROOT


def build_handler(model_dir: Path | None = None) -> Mangum:
    """Build the handler; the model loads once per cold start (lifespan startup)."""
    d = model_dir or Path(os.environ.get("P1_MODEL_DIR", ROOT / "artifacts" / "model"))
    return Mangum(create_app(d), lifespan="auto")


def handler(event, context):  # pragma: no cover - thin wrapper, exercised via build_handler
    global _h
    if _h is None:
        _h = build_handler()
    return _h(event, context)


_h: Mangum | None = None
