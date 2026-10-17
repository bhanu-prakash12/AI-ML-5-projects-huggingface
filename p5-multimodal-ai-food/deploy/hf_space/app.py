"""Hugging Face Space entry point: Gradio UI + the REST routes (/health, /analyze) in one process on port 7860.

Launched through gr.Blocks.launch() on purpose: ZeroGPU hooks that call to detect the @spaces.GPU function.
"""
import asyncio
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
PORT = int(os.environ.get("PORT", "7860"))
os.environ["FOOD_LENS_API"] = f"http://127.0.0.1:{PORT}"  # the UI calls the API in this same process
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:  # ZeroGPU: must be imported before torch; harmless (no-op) on CPU/local runs
    import spaces
except ImportError:
    spaces = None

import gradio as gr  # noqa: E402

from food_lens import api  # noqa: E402
from food_lens.pipeline import Engine  # noqa: E402

ENGINE = Engine(api.CFG, use_vlm=os.environ.get("FOOD_LENS_NO_VLM") != "1")  # models load at startup


def _analyze(data: bytes) -> dict:
    """Module-level and bytes-only on purpose: ZeroGPU pickles the call arguments to its GPU worker."""
    return ENGINE.analyze(data)


gpu_analyze = spaces.GPU(duration=60)(_analyze) if spaces is not None else _analyze


class GpuEngine:
    """What the API routes call: same interface as Engine, but inference runs inside the GPU function."""

    use_vlm = ENGINE.use_vlm

    def analyze(self, data: bytes) -> dict:
        return gpu_analyze(data)


api.STATE["engine"] = GpuEngine()
api.STATE["sem"] = asyncio.Semaphore(1)

spec = importlib.util.spec_from_file_location("food_ui", ROOT / "scripts" / "ui.py")
ui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ui)
demo = ui.build()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=PORT, prevent_thread_lock=True)
    # add the REST routes ahead of Gradio's catch-all routes
    for route in [r for r in api.app.routes if getattr(r, "path", "") in ("/health", "/analyze")][::-1]:
        demo.app.router.routes.insert(0, route)
    demo.block_thread()
