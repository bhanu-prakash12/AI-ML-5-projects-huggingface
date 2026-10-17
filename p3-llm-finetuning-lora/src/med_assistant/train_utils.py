"""Small pure helpers for training reports."""
from __future__ import annotations

import statistics as st


def loss_trend(losses: list[float], k: int = 5) -> dict[str, float | bool]:
    """Compare mean of first k vs last k logged losses."""
    k = min(k, max(1, len(losses) // 2))
    first, last = st.mean(losses[:k]), st.mean(losses[-k:])
    return {"first": first, "last": last, "falling": last < first}


def eta_minutes(sec_per_step: float, total_steps: int, done: int = 0) -> float:
    return sec_per_step * (total_steps - done) / 60.0
