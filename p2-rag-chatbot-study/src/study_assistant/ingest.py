"""Loaders, chunking, content-hash IDs."""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

log = logging.getLogger(__name__)

# A question starts a line with "Question 3", "Q3.", "3." or "3)" (optionally "(a)" sub-parts stay inside).
QUESTION_RE = re.compile(
    r"(?m)^[ \t]*(?:(?:Question|Problem|Q)[ \t]*\.?[ \t]*\d{1,2}\b[.):]?|\d{1,2}[.)])(?=[ \t]|$)"
)
FRONT_RE = re.compile(r"\A---\n(.*?)\n---\n", re.S)


@dataclass
class Chunk:
    text: str
    metadata: dict[str, Any]
    id: str = field(init=False)

    def __post_init__(self) -> None:
        self.id = chunk_id(self.text, self.metadata)


def chunk_id(text: str, meta: dict[str, Any]) -> str:
    """Stable content hash: same text + source + page -> same ID (idempotent ingest)."""
    key = f"{meta.get('source')}|{meta.get('page')}|{text}"
    return hashlib.sha256(key.encode()).hexdigest()[:32]


def parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    m = FRONT_RE.match(raw)
    if not m:
        return {}, raw
    meta = {}
    for line in m.group(1).splitlines():
        k, _, v = line.partition(":")
        meta[k.strip()] = v.strip()
    return meta, raw[m.end():]


def chunk_notes(raw: str, source: str, size: int = 800, overlap: int = 120) -> list[Chunk]:
    front, body = parse_front_matter(raw)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap, separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]
    )
    base = {"source": source, "page": 1, "doc_type": "notes", "year": None,
            "title": front.get("title"), "url": front.get("url"), "revision": front.get("revision")}
    return [Chunk(t, dict(base)) for t in splitter.split_text(body) if t.strip()]


def split_questions(text: str) -> list[tuple[str, str]]:
    """Return [(label, text)] split at question markers; [] if no marker found."""
    marks = list(QUESTION_RE.finditer(text))
    if len(marks) < 2:
        return []
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        body = text[m.start():end].strip()
        label = re.sub(r"\D", "", m.group(0)) or str(i + 1)
        out.append((label, body))
    return out


def chunk_exam_pages(pages: list[str], source: str, year: int | None = None,
                     size: int = 800, overlap: int = 120) -> tuple[list[Chunk], dict[str, int]]:
    """Per-question chunking; pages without question markers fall back to recursive splitting.

    Returns (chunks, stats) where stats counts question vs fallback chunks.
    """
    fallback = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    chunks: list[Chunk] = []
    stats = {"question_chunks": 0, "fallback_chunks": 0}
    for page_no, text in enumerate(pages, start=1):
        base = {"source": source, "page": page_no, "doc_type": "exam", "year": year}
        qs = split_questions(text)
        if qs:
            for label, body in qs:
                parts = fallback.split_text(body) if len(body) > size * 2 else [body]
                for part in parts:
                    chunks.append(Chunk(part, {**base, "question": label}))
                    stats["question_chunks"] += 1
        else:
            for part in fallback.split_text(text):
                if part.strip():
                    chunks.append(Chunk(part, dict(base)))
                    stats["fallback_chunks"] += 1
    return chunks, stats


def load_pdf_pages(path: Path) -> list[str]:
    from pypdf import PdfReader

    return [(p.extract_text() or "") for p in PdfReader(str(path)).pages]


def year_from_name(name: str) -> int | None:
    m = re.search(r"(19|20)\d{2}", name)
    return int(m.group(0)) if m else None


def ingest_corpus(corpus: Path, size: int = 800, overlap: int = 120) -> tuple[list[Chunk], dict[str, Any]]:
    chunks: list[Chunk] = []
    stats: dict[str, Any] = {"notes_docs": 0, "exam_docs": 0, "question_chunks": 0, "fallback_chunks": 0}
    for f in sorted((corpus / "notes").glob("*.md")) if (corpus / "notes").exists() else []:
        chunks += chunk_notes(f.read_text(), f.name, size, overlap)
        stats["notes_docs"] += 1
    exams = corpus / "exams"
    for f in sorted(exams.glob("*.pdf")) if exams.exists() else []:
        c, s = chunk_exam_pages(load_pdf_pages(f), f.name, year_from_name(f.name), size, overlap)
        chunks += c
        stats["exam_docs"] += 1
        stats["question_chunks"] += s["question_chunks"]
        stats["fallback_chunks"] += s["fallback_chunks"]
    # dedupe by ID (idempotency within a run too)
    uniq = {c.id: c for c in chunks}
    stats["total_chunks"] = len(uniq)
    stats["duplicates_dropped"] = len(chunks) - len(uniq)
    exam_total = stats["question_chunks"] + stats["fallback_chunks"]
    stats["fallback_rate"] = round(stats["fallback_chunks"] / exam_total, 3) if exam_total else None
    return list(uniq.values()), stats
