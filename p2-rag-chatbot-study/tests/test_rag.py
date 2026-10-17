import sys
import types

from langchain_core.documents import Document

from study_assistant import rag
from study_assistant.config import load_config


def doc(text, src="a.md", page=1):
    return Document(page_content=text, metadata={"source": src, "page": page, "doc_type": "notes"})


def test_condense_skipped_without_history():
    def boom(*a, **k):
        raise AssertionError("LLM must not be called")

    assert rag.condense("What is SVM?", [], boom) == "What is SVM?"


def test_condense_uses_first_line():
    out = rag.condense("and its cost?", [("What is SVM?", "A classifier.")],
                       lambda m, max_new_tokens: "What is the cost of an SVM?\nextra")
    assert out == "What is the cost of an SVM?"


def test_citations_and_refusal():
    assert rag.cited_labels("x [a.md:1] y [b.pdf:3] [a.md:1]") == ["a.md:1", "b.pdf:3"]
    assert rag.is_refusal("This is not in your notes.")
    assert not rag.is_refusal("Gradient descent minimizes loss.")


def test_build_messages_truncates_to_budget():
    cfg = load_config()
    cfg["generation"] = {**cfg["generation"], "context_budget_tokens": 50}
    docs = [doc("word " * 30, f"s{i}.md") for i in range(4)]
    hist = [("q1", "a1"), ("q2", "a2")]
    msgs, used = rag.build_messages("why?", docs, hist, cfg, lambda t: len(t.split()))
    assert len(used) < 4
    assert msgs[-1]["role"] == "user" and "why?" in msgs[-1]["content"]
    assert all(m["content"] != "q1" for m in msgs)  # history dropped first


def test_ask_stream_gate_refuses_without_llm(monkeypatch):
    cfg = load_config()
    cfg["retrieval"] = {**cfg["retrieval"], "score_threshold": 0.65}
    monkeypatch.setattr(rag, "retrieve", lambda *a, **k: rag.Retrieved([doc("x")], [0.3]))
    fake = types.SimpleNamespace(generate=lambda *a, **k: "", count_tokens=len)
    monkeypatch.setitem(sys.modules, "study_assistant.llm", fake)
    monkeypatch.setattr("study_assistant.llm", fake, raising=False)
    r = rag.ask(None, "who won the world cup?", cfg=cfg)
    assert r["refused"] and r["gate"] and r["answer"] == rag.REFUSAL and r["sources"] == []


def test_ask_stream_appends_sources_when_model_does_not_cite(monkeypatch):
    cfg = load_config()
    monkeypatch.setattr(rag, "retrieve", lambda *a, **k: rag.Retrieved([doc("svm text", "svm.md")], [0.9]))
    fake = types.SimpleNamespace(generate=lambda *a, **k: "", count_tokens=len,
                                 stream=lambda *a, **k: iter(["It maximizes ", "the margin."]))
    monkeypatch.setitem(sys.modules, "study_assistant.llm", fake)
    monkeypatch.setattr("study_assistant.llm", fake, raising=False)
    r = rag.ask(None, "What does SVM maximize?", cfg=cfg)
    assert not r["refused"] and r["citations"] == ["svm.md:1"] and r["model_cited"] == []
    assert r["answer"].endswith("[svm.md:1]")
