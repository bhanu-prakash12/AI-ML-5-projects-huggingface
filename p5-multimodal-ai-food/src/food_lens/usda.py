"""Match Food-101 classes to USDA FoodData Central rows (per 100 g). Never invents values."""
from __future__ import annotations

import re

NUTRIENTS = {1008: "kcal", 1003: "protein_g", 1004: "fat_g", 1005: "carbs_g", 1079: "fiber_g"}
STOP = {"with", "and", "of", "in", "the", "a", "raw", "cooked"}


def tokens(text: str) -> set[str]:
    out = set()
    for w in re.findall(r"[a-z]+", text.lower()):
        if w in STOP:
            continue
        out.add(w[:-1] if len(w) > 3 and w.endswith("s") else w)
    return out


def score(query: str, desc: str) -> float:
    """Fraction of query tokens in description, minus a small penalty for extra description words."""
    q, d = tokens(query), tokens(desc)
    if not q or not d:
        return 0.0
    return len(q & d) / len(q) - 0.02 * len(d - q)


def best_match(query: str, foods: list[tuple[int, str, str]]) -> tuple[int, str, str, float] | None:
    """foods: (fdc_id, description, data_type). Returns best with full token coverage, else None."""
    best = None
    for fid, desc, dtype in foods:
        s = score(query, desc)
        if s >= 0.9 and (best is None or s > best[3]):
            best = (fid, desc, dtype, s)
    return best
