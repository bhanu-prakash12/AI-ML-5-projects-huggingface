import pandas as pd
import pytest

from med_assistant.data import build_messages, clean_and_dedupe, select_and_split

A = "x" * 100


def df():
    rows = [{"qtype": "t", "Question": f"Q {i}?", "Answer": f"{A} {i}"} for i in range(30)]
    rows += [{"qtype": "t", "Question": "q 1?", "Answer": "other " + A}]  # dup question (case-insens.)
    rows += [{"qtype": "t", "Question": "short", "Answer": "tiny"}]
    rows += [{"qtype": "t", "Question": "  Q   2? ", "Answer": float("nan")}]
    return pd.DataFrame(rows)


tpl = lambda m: "|".join(x["content"] for x in m)  # noqa: E731
sizes = {"train": 10, "val": 5, "test": 5}


def test_clean_dedupe():
    c = clean_and_dedupe(df())
    assert len(c) == 30 and c.question.str.lower().is_unique


def test_split_disjoint_deterministic_and_length_filter():
    c = clean_and_dedupe(df())
    a = select_and_split(c, "sys", tpl, len, 10_000, sizes, 42)
    b = select_and_split(c, "sys", tpl, len, 10_000, sizes, 42)
    assert a == b
    qs = [r["question"] for k in a for r in a[k]]
    assert len(qs) == 20 and len(set(qs)) == 20
    assert [len(a[k]) for k in a] == [10, 5, 5]
    assert all(r["n_tokens"] <= 10_000 for k in a for r in a[k])
    with pytest.raises(ValueError):
        select_and_split(c, "sys", tpl, len, 50, sizes, 42)


def test_messages():
    assert build_messages("s", "q")[-1]["role"] == "user"
    assert build_messages("s", "q", "a")[-1]["role"] == "assistant"
