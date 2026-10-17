"""Chroma collection + bge-small embeddings (cosine)."""
from __future__ import annotations

import logging
import os
from typing import Any

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from study_assistant.config import load_config, resolve  # noqa: E402
from study_assistant.ingest import Chunk  # noqa: E402

log = logging.getLogger(__name__)
COLLECTION = "study_notes"
BATCH = 64


def get_embeddings(cfg: dict[str, Any] | None = None):
    from langchain_huggingface import HuggingFaceEmbeddings

    from study_assistant.device import get_device

    cfg = cfg or load_config()
    m = cfg["models"]["embedding"]
    return HuggingFaceEmbeddings(
        model_name=m["name"],
        model_kwargs={"device": get_device(), "revision": m["revision"]},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_store(cfg: dict[str, Any] | None = None, embeddings=None):
    from langchain_chroma import Chroma

    cfg = cfg or load_config()
    path = resolve(cfg, "chroma")
    path.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings or get_embeddings(cfg),
        persist_directory=str(path),
        collection_metadata={"hnsw:space": "cosine"},
    )


def clean_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """Chroma rejects None values."""
    return {k: v for k, v in meta.items() if v is not None}


def add_chunks(store, chunks: list[Chunk]) -> int:
    """Embed and add only chunks whose IDs are not yet stored. Returns number added."""
    ids = [c.id for c in chunks]
    existing: set[str] = set()
    for i in range(0, len(ids), 500):
        existing.update(store.get(ids=ids[i : i + 500])["ids"])
    new = [c for c in chunks if c.id not in existing]
    for i in range(0, len(new), BATCH):
        b = new[i : i + BATCH]
        store.add_texts([c.text for c in b], [clean_meta(c.metadata) for c in b], [c.id for c in b])
    log.info("index: %d new, %d already present", len(new), len(existing))
    return len(new)


def search(store, query: str, k: int = 4, doc_type: str | None = None,
           instruction: str = "") -> list[tuple[Any, float]]:
    """Similarity search with bge query instruction; returns (doc, cosine_similarity)."""
    flt = {"doc_type": doc_type} if doc_type else None
    res = store.similarity_search_with_score(instruction + query, k=k, filter=flt)
    return [(d, 1.0 - dist) for d, dist in res]
