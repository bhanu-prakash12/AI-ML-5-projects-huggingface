"""FastAPI: /health, /analyze (multipart image <= 5 MB, resized <= 768 px, per-stage timings)."""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from .config import load_config
from .console import setup_logging
from .pipeline import Engine
from .schema import AnalyzeResponse

setup_logging()
log = logging.getLogger("food_lens.api")
CFG = load_config()
STATE: dict = {}

DESCRIPTION = """
**Food Lens**: photo in, dish + USDA nutrition per 100 g + two recipe ideas out. Fully local (CLIP + SmolVLM-500M).

**How to demo (this page):** open `POST /analyze` -> *Try it out* -> choose a JPEG/PNG (<= 5 MB) -> *Execute*.
Expect `top3` (CLIP guesses with probabilities), `nutrition` (from the USDA CSV, never from the VLM),
`vlm` (why / ingredients / portion estimate / recipes) and `timings_ms` per stage. First call may be slower.
Sample photos live in `assets/`.

**curl (macOS/Linux):** `curl -F "file=@assets/sample_1_cheese_plate.jpg" http://127.0.0.1:8005/analyze`

**PowerShell (Windows 10+):** `curl.exe -F "file=@assets/sample_1_cheese_plate.jpg" http://127.0.0.1:8005/analyze`

Errors: 400 empty upload, 413 larger than 5 MB, 415 not a readable image.
"""


class Health(BaseModel):
    status: str
    models_loaded: bool
    vlm: bool | None = None

    model_config = {"json_schema_extra": {"examples": [{"status": "ok", "models_loaded": True, "vlm": True}]}}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["engine"] = Engine(CFG, use_vlm=os.environ.get("FOOD_LENS_NO_VLM") != "1")
    STATE["sem"] = asyncio.Semaphore(1)  # one inference at a time on a laptop
    log.info("models loaded")
    yield
    STATE.clear()


app = FastAPI(title="Food Lens", description=DESCRIPTION, version="1.0", lifespan=lifespan)


@app.get("/health", response_model=Health, summary="Liveness + whether the models are loaded")
def health():
    return {"status": "ok", "models_loaded": "engine" in STATE, "vlm": STATE["engine"].use_vlm if STATE else None}


@app.post("/analyze", response_model=AnalyzeResponse, response_model_exclude_none=False,
          summary="Analyze one food photo (dish, nutrition, recipes, timings)")
async def analyze(file: UploadFile = File(..., description="JPEG/PNG food photo, at most 5 MB")):
    limit = CFG["api"]["max_upload_bytes"]
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"image larger than {limit // 1024 // 1024} MB")
    if not data:
        raise HTTPException(400, "empty upload")
    eng: Engine = STATE["engine"]
    async with STATE["sem"]:
        try:
            return await run_in_threadpool(eng.analyze, data)
        except OSError as e:  # PIL.UnidentifiedImageError subclasses OSError
            raise HTTPException(415, f"not a readable image: {e}") from e
