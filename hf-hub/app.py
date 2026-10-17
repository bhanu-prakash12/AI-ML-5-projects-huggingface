"""Hugging Face Space entry: one Gradio app hosting p1..p5 (+ p5's REST routes /health /analyze) on port 7860.

`spaces` is imported first (ZeroGPU requirement). Launched via Blocks.launch() so ZeroGPU can detect @spaces.GPU functions.
"""
import asyncio
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
PORT = int(os.environ.get("PORT", "7860"))
os.environ["FOOD_LENS_API"] = f"http://127.0.0.1:{PORT}"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from hub.gpu import gpu, spaces  # noqa: E402,F401  (imports `spaces` before torch)

import gradio as gr  # noqa: E402

from hub import home, paths  # noqa: E402,F401
from hub.theme import CSS, THEME  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
TABS = [("p1", "P1 Dropout risk"), ("p2", "P2 Study assistant"), ("p3", "P3 Medical LoRA"),
        ("p4", "P4 Drift monitor"), ("p5", "P5 Food Lens")]

# ---- p5: engine behind its own REST routes, inference inside a ZeroGPU function ----
from food_lens import api as food_api  # noqa: E402
from food_lens.pipeline import Engine  # noqa: E402

ENGINE = Engine(food_api.CFG, use_vlm=os.environ.get("FOOD_LENS_NO_VLM") != "1")


@gpu(60)
def _food_analyze(data: bytes) -> dict:
    """Module-level, bytes-only: ZeroGPU pickles the call arguments."""
    return ENGINE.analyze(data)


class GpuEngine:
    use_vlm = ENGINE.use_vlm

    def analyze(self, data: bytes) -> dict:
        return _food_analyze(data)


food_api.STATE["engine"] = GpuEngine()
food_api.STATE["sem"] = asyncio.Semaphore(1)

if os.environ.get("HUB_PRELOAD", "1") == "1":  # load shared models once at startup (ZeroGPU best practice)
    from hub.models import p2_store, shared_llm

    shared_llm()
    p2_store()

from hub.tabs import p1, p2, p3, p4, p5  # noqa: E402

with gr.Blocks(title="AI Engineering Projects", theme=THEME, css=CSS) as demo:
    with gr.Tabs() as tabs:
        with gr.Tab("Home", id="home"):
            home.render(tabs)
        for (pid, label), mod in zip(TABS, (p1, p2, p3, p4, p5)):
            with gr.Tab(label, id=pid):
                mod.render()

if __name__ == "__main__":
    allowed = [str(ROOT / "projects" / "p4" / "reports")]
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0", server_port=PORT, prevent_thread_lock=True, allowed_paths=allowed)
    for route in [r for r in food_api.app.routes if getattr(r, "path", "") in ("/health", "/analyze")][::-1]:
        demo.app.router.routes.insert(0, route)
    demo.block_thread()
