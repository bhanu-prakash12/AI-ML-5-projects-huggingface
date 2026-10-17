"""FastAPI: ONE model in memory (base + LoRA adapter); mode=base uses disable_adapter()."""
from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field

from med_assistant.config import load_config, resolve
from med_assistant.data import build_messages
from med_assistant.llm import base_mode, generate_batch, load
from med_assistant.safety import REFUSAL, is_emergency

_S: dict = {}
_LOCK = threading.Lock()  # adapter toggling is global model state: one generation at a time


class GenReq(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [
        {"question": "What are the symptoms of type 2 diabetes?", "mode": "tuned", "max_new_tokens": 120},
        {"question": "What are the symptoms of type 2 diabetes?", "mode": "base", "max_new_tokens": 120},
        {"question": "I have severe chest pain and cannot breathe", "mode": "tuned"}]})

    question: str = Field(min_length=3, max_length=1000)
    mode: Literal["base", "tuned"] = "tuned"
    max_new_tokens: int = Field(default=200, ge=1, le=300)


class HealthResp(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"status": "ok", "adapter_loaded": True}]})
    status: str
    adapter_loaded: bool


class GenResp(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [
        {"mode": "tuned", "answer": "Type 2 diabetes is characterized by ...", "disclaimer":
         "Educational information only, not medical advice. Consult a qualified clinician.",
         "refused": False, "seconds": 4.2},
        {"mode": "tuned", "answer": "This sounds like it could be an emergency. ...", "disclaimer":
         "Educational information only, not medical advice. Consult a qualified clinician.",
         "refused": True, "seconds": 0.0}]})

    mode: str
    answer: str
    disclaimer: str
    refused: bool
    seconds: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    adapter = resolve(cfg, "artifacts") / "adapter"
    has = adapter.exists()
    tok, model, dev = load(cfg, adapter if has else None)
    _S.update(cfg=cfg, tok=tok, model=model, dev=dev, adapter=has)
    yield
    _S.clear()


DESCRIPTION = """
**What:** one Qwen2.5-0.5B-Instruct model in memory, with a LoRA adapter. `mode=tuned` uses the adapter,
`mode=base` switches it off for the same request, so you can compare them side by side.

**Input:** `POST /generate` with `question`, `mode` (`base` or `tuned`) and optional `max_new_tokens`.
**Output:** `answer`, a `disclaimer`, `refused` (true when the safety filter fired) and `seconds`.

**Demo:** click *Try it out* on `/generate`, pick an example from the dropdown:
1. tuned question -> a short MedQuAD-style answer; 2. the same question with `mode=base` -> a longer markdown-style answer;
3. the chest-pain example -> `refused: true` and a fixed emergency message (no model call, ~0 s).

Educational demo, not medical advice. On a CPU-only machine each answer can take tens of seconds.
"""

app = FastAPI(title="med-assistant (LoRA demo)", description=DESCRIPTION, lifespan=lifespan)


@app.get("/health", response_model=HealthResp, summary="Is the model loaded?")
def health() -> HealthResp:
    return HealthResp(status="ok" if "model" in _S else "loading", adapter_loaded=_S.get("adapter", False))


def _run(req: GenReq) -> str:
    cfg = _S["cfg"]
    msgs = [build_messages(cfg["system_prompt"], req.question)]
    with _LOCK, base_mode(_S["model"], req.mode == "base"):
        return generate_batch(_S["tok"], _S["model"], _S["dev"], msgs, req.max_new_tokens, 1)[0]


@app.post("/generate", response_model=GenResp, summary="Answer a medical question (base or LoRA-tuned)")
async def generate(req: GenReq) -> GenResp:
    if "model" not in _S:
        raise HTTPException(503, "model loading")
    cfg = _S["cfg"]
    if is_emergency(req.question):
        return GenResp(mode=req.mode, answer=REFUSAL, disclaimer=cfg["disclaimer"], refused=True, seconds=0.0)
    if req.mode == "tuned" and not _S["adapter"]:
        raise HTTPException(409, "adapter not trained; run make train")
    t0 = time.time()
    ans = await run_in_threadpool(_run, req)
    return GenResp(mode=req.mode, answer=ans, disclaimer=cfg["disclaimer"], refused=False,
                   seconds=round(time.time() - t0, 2))
