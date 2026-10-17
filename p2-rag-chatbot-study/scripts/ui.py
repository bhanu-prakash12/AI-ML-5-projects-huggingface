"""Gradio chat that consumes the FastAPI /ask SSE stream (no models loaded here).

Inputs: a question + document filter. Outputs: streamed answer with sources. Needs the API running (`python run.py serve`).
Sample: python scripts/ui.py   (port from config ui.port or P2_UI_PORT)
"""
import json
import os
import sys
from pathlib import Path

import gradio as gr
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from study_assistant import console  # noqa: E402
from study_assistant.config import api_port, load_config, ui_port  # noqa: E402

CFG = load_config()
API = os.environ.get("API_URL", f"http://127.0.0.1:{api_port(CFG)}")


def sse_events(resp):
    ev = None
    for line in resp.iter_lines():
        if line.startswith("event: "):
            ev = line[7:]
        elif line.startswith("data: ") and ev:
            yield ev, json.loads(line[6:])


def fmt_sources(src: list[dict]) -> str:
    if not src:
        return ""
    return "\n\n<details><summary>Sources</summary>\n\n" + "\n".join(
        f"- `{s['label']}` (sim {s['score']}): {s['snippet'][:120].strip()}..." for s in src) + "\n\n</details>"


def chat(message, history, doc_type):
    pairs, last_user = [], None
    for h in history:
        if h["role"] == "user":
            last_user = h["content"]
        elif last_user is not None:
            pairs.append((last_user, h["content"]))
            last_user = None
    body = {"question": message, "history": pairs[-3:], "doc_type": None if doc_type == "all" else doc_type}
    text, src = "", []
    try:
        with httpx.stream("POST", f"{API}/ask", json=body, timeout=300) as r:
            r.raise_for_status()
            for ev, data in sse_events(r):
                if ev == "sources":
                    src = data
                elif ev == "token":
                    text += data
                    yield text
                elif ev == "error":
                    yield f"Error: {data}"
                    return
    except httpx.HTTPError as e:
        yield f"API unavailable ({e}). Start it with `python run.py serve` (or `make serve`)."
        return
    yield text + fmt_sources(src)


HOWTO = """## Study Assistant: ask your ML notes and past exams
**How to use:** type a question or click an example below. Answers come only from the indexed notes/exams and end with
their sources. Use *Document type* to search only notes or only exams.
**What to expect:** in-corpus questions stream an answer plus a *Sources* section; a question outside the corpus gets
`This is not in your notes.` (the refusal gate fires before the model runs). First answer may take a few seconds."""

TRADEOFFS = """- **Pros:** grounded answers with sources, refuses instead of guessing, fully local.
- **Cons:** a 1.5B model can paraphrase badly or miss multi-chunk synthesis; one request at a time.
- **At scale:** client/server vector DB, GPU serving (vLLM), semantic cache, auth and rate limits."""

EXAMPLES = [
    ["What is the bias-variance tradeoff?", "all"],  # in-corpus -> answer + sources
    ["How does gradient descent update model parameters?", "notes"],  # in-corpus -> answer + sources
    ["Who won the 1998 football world cup?", "all"],  # out-of-corpus -> refusal
]

with gr.Blocks(title="Study Assistant") as demo:
    gr.Markdown(HOWTO)
    gr.ChatInterface(chat, type="messages", examples=EXAMPLES, cache_examples=False,
                     additional_inputs=[gr.Dropdown(["all", "notes", "exam"], value="all", label="Document type")])
    with gr.Accordion("Pros / cons / at scale", open=False):
        gr.Markdown(TRADEOFFS)

if __name__ == "__main__":
    console.banner("Study Assistant UI", f"Gradio chat talking to the API at {API}.",
                   f"open http://127.0.0.1:{ui_port(CFG)} (API must be running: python run.py serve)")
    demo.launch(server_name="127.0.0.1", server_port=ui_port(CFG))
