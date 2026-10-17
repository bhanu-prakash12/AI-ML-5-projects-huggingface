"""Load -> clean -> dedupe -> filter by token length -> chat template -> deterministic splits."""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

_WS = re.compile(r"\s+")


def clean_text(s: Any) -> str:
    return _WS.sub(" ", str(s)).strip() if isinstance(s, str) else ""


def clean_and_dedupe(df: pd.DataFrame, min_answer_chars: int = 80) -> pd.DataFrame:
    d = df.copy()
    d["question"] = d["Question"].map(clean_text)
    d["answer"] = d["Answer"].map(clean_text)
    d = d[(d.question != "") & (d.answer.str.len() >= min_answer_chars)]
    d = d.assign(_q=d.question.str.lower(), _a=d.answer.str.lower())
    d = d.drop_duplicates("_q").drop_duplicates("_a")
    return d[["qtype", "question", "answer"]].reset_index(drop=True)


def build_messages(system: str, q: str, a: str | None = None) -> list[dict[str, str]]:
    m = [{"role": "system", "content": system}, {"role": "user", "content": q}]
    if a is not None:
        m.append({"role": "assistant", "content": a})
    return m


def select_and_split(
    df: pd.DataFrame, system: str, template: Callable[[list[dict[str, str]]], str],
    count_tokens: Callable[[str], int], max_tokens: int, sizes: dict[str, int], seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Shuffle deterministically, keep rows whose full chat text fits max_tokens, split disjointly."""
    need = sizes["train"] + sizes["val"] + sizes["test"]
    rows: list[dict[str, Any]] = []
    for r in df.sample(frac=1.0, random_state=seed).itertuples():
        text = template(build_messages(system, r.question, r.answer))
        n = count_tokens(text)
        if n <= max_tokens:
            rows.append({"question": r.question, "answer": r.answer, "qtype": r.qtype, "text": text, "n_tokens": n})
            if len(rows) == need:
                break
    if len(rows) < need:
        raise ValueError(f"only {len(rows)} rows fit, need {need}")
    a, b = sizes["train"], sizes["train"] + sizes["val"]
    return {"train": rows[:a], "val": rows[a:b], "test": rows[b:]}


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]
