"""FastAPI /ingest /ask (SSE) /ask/simple (JSON) /health. One process owns both models (lifespan, workers=1)."""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from study_assistant.config import load_config, resolve

log = logging.getLogger(__name__)
INGEST_LOCK = threading.Lock()


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    history: list[tuple[str, str]] = Field(default_factory=list, max_length=10)
    doc_type: Literal["notes", "exam"] | None = None

    model_config = {"json_schema_extra": {"examples": [
        {"question": "What is the bias-variance tradeoff?", "history": [], "doc_type": None},
        {"question": "Who won the 1998 football world cup?", "history": [], "doc_type": "notes"},
    ]}}


class Source(BaseModel):
    label: str
    source: str | None = None
    page: int | str | None = None
    doc_type: str | None = None
    score: float
    snippet: str


class AskResponse(BaseModel):
    answer: str
    refused: bool
    top_score: float | None = None
    citations: list[str] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)

    model_config = {"json_schema_extra": {"examples": [{
        "answer": "The bias-variance tradeoff ... [wiki_bias_variance_tradeoff.md:1]", "refused": False,
        "top_score": 0.86, "citations": ["wiki_bias_variance_tradeoff.md:1"],
        "sources": [{"label": "wiki_bias_variance_tradeoff.md:1", "source": "wiki_bias_variance_tradeoff.md", "page": 1,
                     "doc_type": "notes", "score": 0.86, "snippet": "..."}]}]}}


class HealthResponse(BaseModel):
    status: str
    profile: str | None = None


class IngestResponse(BaseModel):
    chunks: int
    added: int


DESCRIPTION = """
Local RAG study assistant: answers ONLY from your ML notes and past exams, cites sources, and refuses when the
answer is not in the corpus.

**Demo in Swagger**
1. Wait until `GET /health` returns `{"status": "ok"}` (models load once at startup).
2. Open `POST /ask/simple`, click *Try it out*, run the pre-filled example (in-corpus: answered with sources).
3. Change the question to `Who won the 1998 football world cup?` -> `refused: true` (score below the gate).

`POST /ask` is the same pipeline as a Server-Sent Events stream (events `sources`, `token`, `done`, `error`);
the Gradio chat uses it. Swagger cannot render SSE, so use `/ask/simple` here.
"""


def load_resources(cfg: dict[str, Any]) -> dict[str, Any]:
    """Load embeddings, Chroma store and the LLM once. Monkeypatched in tests."""
    from study_assistant import index, llm

    emb = index.get_embeddings(cfg)
    store = index.get_store(cfg, emb)
    llm.load(cfg)
    return {"store": store}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    cfg = load_config()
    app.state.cfg = cfg
    app.state.res = await run_in_threadpool(load_resources, cfg)
    app.state.sem = asyncio.Semaphore(cfg["profiles"][cfg["profile"]]["max_concurrent"])
    log.info("models loaded")
    yield


app = FastAPI(title="Study Assistant", description=DESCRIPTION, version="0.1", lifespan=lifespan)


def stream_events(store: Any, req: AskRequest, cfg: dict[str, Any]) -> Iterator[dict[str, Any]]:
    from study_assistant import rag

    return rag.ask_stream(store, req.question, req.history, req.doc_type, cfg)


def sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/health", response_model=HealthResponse, summary="Is the API ready (models loaded)?")
def health() -> dict[str, Any]:
    ok = hasattr(app.state, "res")
    return {"status": "ok" if ok else "loading", "profile": app.state.cfg["profile"] if ok else None}


@app.post("/ask", summary="Ask a question (SSE stream: sources, token, done)", response_class=StreamingResponse)
async def ask(req: AskRequest) -> StreamingResponse:
    if not hasattr(app.state, "res"):
        raise HTTPException(503, "models not loaded")
    store, cfg = app.state.res["store"], app.state.cfg

    async def gen() -> AsyncIterator[str]:
        async with app.state.sem:
            it = stream_events(store, req, cfg)
            sentinel = object()
            try:
                while (ev := await run_in_threadpool(next, it, sentinel)) is not sentinel:
                    if "sources" in ev:
                        yield sse("sources", ev["sources"])
                    elif "token" in ev:
                        yield sse("token", ev["token"])
                    else:
                        yield sse("done", ev["done"])
            except Exception as e:  # surface as an SSE error event instead of a broken stream
                log.exception("ask failed")
                yield sse("error", str(e))

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/ask/simple", response_model=AskResponse, summary="Ask a question (single JSON reply, for Swagger demos)")
async def ask_simple(req: AskRequest) -> AskResponse:
    if not hasattr(app.state, "res"):
        raise HTTPException(503, "models not loaded")
    store, cfg = app.state.res["store"], app.state.cfg

    def collect() -> AskResponse:
        answer: list[str] = []
        sources: list[dict[str, Any]] = []
        done: dict[str, Any] = {}
        for ev in stream_events(store, req, cfg):
            if "sources" in ev:
                sources = ev["sources"]
            elif "token" in ev:
                answer.append(ev["token"])
            else:
                done = ev["done"]
        return AskResponse(answer="".join(answer), refused=bool(done.get("refused")), top_score=done.get("top_score"),
                           citations=done.get("citations", []), sources=sources)

    async with app.state.sem:
        try:
            return await run_in_threadpool(collect)
        except Exception as e:
            log.exception("ask/simple failed")
            raise HTTPException(500, str(e)) from e


def run_ingest(cfg: dict[str, Any], store: Any) -> dict[str, int]:
    from study_assistant import index
    from study_assistant.ingest import ingest_corpus

    ch = cfg["chunking"]["notes"]
    chunks, stats = ingest_corpus(resolve(cfg, "corpus"), ch["size"], ch["overlap"])
    added = index.add_chunks(store, chunks)
    return {"chunks": stats["total_chunks"], "added": added}


@app.post("/ingest", response_model=IngestResponse, summary="Re-chunk the corpus and add unseen chunks to Chroma")
async def ingest() -> dict[str, int]:
    if not hasattr(app.state, "res"):
        raise HTTPException(503, "models not loaded")
    if not INGEST_LOCK.acquire(blocking=False):
        raise HTTPException(409, "ingest already running")
    try:
        return await run_in_threadpool(run_ingest, app.state.cfg, app.state.res["store"])
    finally:
        INGEST_LOCK.release()
