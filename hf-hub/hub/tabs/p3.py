"""p3 Medical Q&A: base vs LoRA-tuned. One model (shared with p2); real recorded results from reports/ as replay."""
from __future__ import annotations

import json
import time

import gradio as gr

from hub import paths  # noqa: F401
from hub.gpu import gpu
from hub.loader import project_dir
from hub.models import LOCK, shared_llm
from hub.tabs.common import header

HOWTO = """**Educational demo, not medical advice.** For emergencies call your local emergency number.

**How to use:** type a question (or click an example), press **Ask both**. The same model answers twice: adapter
**off** (Base) and **on** (Tuned). Example 3 mentions chest pain: a safety filter refuses before the model runs.

**Honest result** (`reports/compare.md`): ROUGE-L base 0.157 vs tuned 0.154 on 50 test questions, no measurable gain:
the adapter changed *style* (answers 140 -> 87 words), not accuracy."""

EXAMPLES = [["What are the symptoms of type 2 diabetes?"], ["How is high blood pressure diagnosed?"],
            ["I have severe chest pain and cannot breathe"]]


@gpu(60)
def generate(question: str, mode: str) -> dict:
    from med_assistant.data import build_messages
    from med_assistant.llm import base_mode, generate_batch
    from med_assistant.safety import REFUSAL, is_emergency

    cfg, tok, model, dev, has = shared_llm()
    if is_emergency(question):
        return {"answer": REFUSAL, "refused": True, "seconds": 0.0, "disclaimer": cfg["disclaimer"]}
    t0 = time.time()
    with LOCK, base_mode(model, mode == "base" or not has):
        ans = generate_batch(tok, model, dev, [build_messages(cfg["system_prompt"], question)], 160, 1)[0]
    return {"answer": ans, "refused": False, "seconds": round(time.time() - t0, 1), "disclaimer": cfg["disclaimer"]}


def _fmt(r: dict, mode: str) -> str:
    head = "**REFUSED (safety filter)**\n\n" if r["refused"] else ""
    return f"{head}{r['answer']}\n\n_{r['disclaimer']}_\n\n`{mode}, {r['seconds']} s`"


def ask(q: str):
    if not q or len(q.strip()) < 3:
        yield "", "", "Type a question of at least 3 characters."
        return
    yield "_waiting..._", "_waiting..._", "Asking the BASE model (adapter off)..."
    base = _fmt(generate(q, "base"), "base")
    yield base, "_waiting..._", "Asking the TUNED model (adapter on)..."
    yield base, _fmt(generate(q, "tuned"), "tuned"), "Done."


def _recorded() -> list[list[str]]:
    p = project_dir("p3") / "reports" / "compare.json"
    if not p.exists():
        return []
    out = json.loads(p.read_text())["outputs"]
    b, t = out["base"]["test"], out["tuned"]["test"]
    return [[f"test #{i + 1}", b[i][:400], t[i][:400]] for i in range(min(5, len(b), len(t)))]


def render() -> None:
    header("p3", "Medical Q&A: base vs LoRA", "Fine-tuning with LoRA (0.22% of weights) and an honest evaluation.",
           "p3-llm-finetuning-lora")
    gr.Markdown(HOWTO)
    q = gr.Textbox(label="Your medical question", placeholder="What are the symptoms of type 2 diabetes?")
    btn = gr.Button("Ask both", variant="primary")
    status = gr.Markdown("Status: idle")
    with gr.Row():
        a = gr.Markdown(label="Base")
        b = gr.Markdown(label="Tuned")
    gr.Examples(EXAMPLES, inputs=q, label="Click an example, then press Ask both")
    btn.click(ask, q, [a, b, status])
    q.submit(ask, q, [a, b, status])
    with gr.Accordion("Recorded results from the real run (no model call)", open=False):
        gr.Markdown("5 of the 50 test questions answered by the base and the tuned model during `make eval` "
                    "(greedy, max 200 new tokens). Full table: `reports/compare.md`.")
        gr.Dataframe(_recorded(), headers=["question", "base answer", "tuned answer"], wrap=True)
        gr.Markdown("Training: LoRA r8/alpha16, 1.08M trainable params (0.22%), 37 steps, 5.2 min on MPS; "
                    "val loss 2.134 -> 1.917. Forgetting check: 4/4 keyword-correct for both models.")
