"""ROUGE-L (rouge_score, stemmed) and length stats."""
from __future__ import annotations

import statistics as st


def rouge_l(preds: list[str], refs: list[str]) -> float:
    from rouge_score import rouge_scorer

    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return st.mean(sc.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs, strict=True))


def avg_words(texts: list[str]) -> float:
    return st.mean(len(t.split()) for t in texts)


def rouge_l_each(preds: list[str], refs: list[str]) -> list[float]:
    from rouge_score import rouge_scorer

    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return [sc.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs, strict=True)]


def paired_bootstrap(a: list[float], b: list[float], n: int = 2000, seed: int = 42) -> dict[str, float]:
    """Mean(b - a) with 95% bootstrap CI and win/tie/loss counts of b vs a."""
    import numpy as np

    d = np.array(b) - np.array(a)
    rng = np.random.default_rng(seed)
    means = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return {"mean_diff": float(d.mean()), "ci_lo": float(np.percentile(means, 2.5)),
            "ci_hi": float(np.percentile(means, 97.5)), "wins": int((d > 1e-9).sum()),
            "ties": int((abs(d) <= 1e-9).sum()), "losses": int((d < -1e-9).sum())}
