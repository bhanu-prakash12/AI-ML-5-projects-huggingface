"""p2 Study assistant: RAG chat over ML notes/exams, in-process, shares the single Qwen 0.5B with p3."""
from __future__ import annotations

import gradio as gr

from hub import paths  # noqa: F401
from hub.gpu import gpu
from hub.models import LOCK, p2_store, shared_llm
from hub.tabs.common import header

HOWTO = """**How to use:** type a question or click an example. Answers come only from the indexed ML notes (20 Wikipedia
articles) and two MIT OCW exams, and end with their sources. *Document type* narrows the search.

**What to expect:** in-corpus questions get an answer plus a *Sources* section. A question outside the corpus gets
`This is not in your notes.`: a similarity gate refuses **before** the model runs. This hub runs the small 0.5B model
(shared with p3), so wording is weaker than the 1.5B in the standalone project."""

EXAMPLES = [
    ["What is the bias-variance tradeoff?", "all"],
    ["How does gradient descent update model parameters?", "notes"],
    ["Who won the 1998 football world cup?", "all"],
]


def fmt_sources(src: list[dict]) -> str:
    if not src:
        return ""
    return "\n\n<details><summary>Sources</summary>\n\n" + "\n".join(
        f"- `{s['label']}` (sim {s['score']}): {s['snippet'][:120].strip()}..." for s in src) + "\n\n</details>"


@gpu(60)
def answer(message: str, pairs: list, doc_type: str) -> dict:
    from med_assistant.llm import base_mode
    from study_assistant import rag

    _, _, model, _, _ = shared_llm()
    cfg, store = p2_store()
    with LOCK, base_mode(model, True):  # adapter off = plain base model
        return rag.ask(store, message, pairs, None if doc_type == "all" else doc_type, cfg)


def chat(message, history, doc_type):
    pairs, last_user = [], None
    for h in history:
        if h["role"] == "user":
            last_user = h["content"]
        elif last_user is not None:
            pairs.append((last_user, h["content"]))
            last_user = None
    r = answer(message, pairs[-3:], doc_type)
    return r["answer"] + fmt_sources(r["sources"])


def render() -> None:
    header("p2", "Study assistant (RAG)", "Retrieval-augmented chat over ML notes: cites sources, refuses when unsure.",
           "p2-rag-chatbot-study")
    gr.Markdown(HOWTO)
    gr.ChatInterface(chat, type="messages", examples=EXAMPLES, cache_examples=False,
                     additional_inputs=[gr.Dropdown(["all", "notes", "exam"], value="all", label="Document type")])
    with gr.Accordion("Pros / cons / at scale", open=False):
        gr.Markdown("- **Pros:** grounded answers with sources, refuses instead of guessing.\n"
                    "- **Cons:** a 0.5B model paraphrases badly; one request at a time.\n"
                    "- **At scale:** client/server vector DB, GPU serving (vLLM), semantic cache, auth, rate limits.")
    with gr.Accordion("Measured results (reports/eval.md, indicative: 31 auto-drafted questions)", open=False):
        gr.Markdown("recall@4 **0.792**, MRR **0.719**, refusal accuracy **1.0** (7/7 unanswerable refused, 24/24 answered). "
                    "Measured with the 1.5B model; retrieval and the refusal gate do not depend on the LLM.")
