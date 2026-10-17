"""p5 Food Lens: the project's own Gradio UI. Its REST routes are grafted onto the hub app in app.py."""
from __future__ import annotations

from hub import paths  # noqa: F401
from hub.loader import load_script
from hub.tabs.common import header


def render() -> None:
    header("p5", "Food Lens", "CLIP zero-shot food recognition -> USDA nutrition -> SmolVLM recipes. Upload or webcam.",
           "p5-multimodal-ai-food")
    load_script("p5", "ui").build().render()
