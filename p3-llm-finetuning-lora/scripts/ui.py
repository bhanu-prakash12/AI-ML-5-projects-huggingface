"""Gradio side-by-side UI (pure HTTP client of the API on :8003).

usage: python run.py ui   (start the API first: python run.py serve)
"""
import httpx
import gradio as gr

from med_assistant.config import load_config

CFG = load_config()
API = f"http://127.0.0.1:{CFG['api']['port']}"

HOW_TO = """
# Medical Q&A: base vs LoRA-tuned (Qwen2.5-0.5B)
**Educational demo, not medical advice.** For emergencies call your local emergency number.

**How to use:** type a question (or click an example below), press **Ask both**. The same model answers twice:
once with the LoRA adapter switched off (**Base**) and once on (**Tuned**).

**What to expect**
- Base: longer, markdown-style lists. Tuned: shorter, MedQuAD-style prose. Honest result from `reports/compare.md`:
  **no measurable ROUGE-L gain** (the adapter changed style, not accuracy).
- Example 3 mentions chest pain: a safety filter refuses *before* the model runs (both columns show the emergency message).
- On a CPU-only machine each answer can take tens of seconds; the status line shows progress.
"""

NOTE = """
**Pros:** adapter is ~19 MB, one model in memory serves both modes, safety check runs before generation.
**Cons:** 0.5B model and 300 training examples; keyword safety filter is crude; answers can be wrong.
**At scale:** vLLM multi-LoRA behind a gateway with safety filter, eval in CI, clinician review.
"""

EXAMPLES = [
    ["What are the symptoms of type 2 diabetes?"],       # expect: base long list, tuned shorter prose
    ["How is high blood pressure diagnosed?"],           # expect: both answer; compare style and length
    ["I have severe chest pain and cannot breathe"],    # expect: refusal in both columns, no model call
]


def _one(q: str, mode: str) -> str:
    try:
        r = httpx.post(f"{API}/generate", json={"question": q, "mode": mode}, timeout=300)
        r.raise_for_status()
        j = r.json()
        head = "**REFUSED (safety filter)**\n\n" if j["refused"] else ""
        return f"{head}{j['answer']}\n\n_{j['disclaimer']}_\n\n`{mode}, {j['seconds']} s`"
    except Exception as e:  # noqa: BLE001
        return f"error: {e}\n\nIs the API running? Start it with `python run.py serve` (port {CFG['api']['port']})."


def ask(q: str):
    if not q or len(q.strip()) < 3:
        yield "", "", "Type a question of at least 3 characters."
        return
    yield "_waiting..._", "_waiting..._", "Asking the BASE model (adapter off)..."
    base = _one(q, "base")
    yield base, "_waiting..._", "Asking the TUNED model (adapter on)..."
    tuned = _one(q, "tuned")
    yield base, tuned, "Done."


with gr.Blocks(title="Medical Q&A: base vs LoRA-tuned") as demo:
    gr.Markdown(HOW_TO)
    q = gr.Textbox(label="Your medical question", placeholder="What are the symptoms of type 2 diabetes?")
    btn = gr.Button("Ask both", variant="primary")
    status = gr.Markdown("Status: idle", label="Status")
    with gr.Row():
        a = gr.Markdown(label="Base model (adapter off)")
        b = gr.Markdown(label="Tuned model (LoRA adapter on)")
    gr.Examples(EXAMPLES, inputs=q, label="Click an example, then press Ask both")
    with gr.Accordion("Pros / cons / at scale", open=False):
        gr.Markdown(NOTE)
    btn.click(ask, q, [a, b, status])
    q.submit(ask, q, [a, b, status])

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=CFG["ui"]["port"])
