"""Gradio UI (upload + webcam + 3 sample photos) that calls the FastAPI /analyze endpoint.

Inputs:  a food photo (upload, webcam or a bundled example from assets/); the API must be running on :8005.
Outputs: CLIP top-3 bar plot, USDA nutrition table, recipe ideas, per-stage timings (page on :7865).
Example: python run.py serve   (terminal 1)   then   python run.py ui   (terminal 2)
"""
import argparse
import os
import tempfile

import gradio as gr
import pandas as pd
import requests
from PIL import Image, ImageOps

from food_lens import console
from food_lens.config import ROOT, load_config

cfg = load_config()
API = os.environ.get("FOOD_LENS_API", f"http://127.0.0.1:{cfg['ports']['api']}")
EMPTY = ({}, pd.DataFrame(), "", "")


def shrink(path: str, max_side: int = 1280) -> str:
    """Downscale + re-encode as JPEG so big phone photos stay under the API's upload limit (and rotate by EXIF)."""
    try:
        im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    except Exception:  # unreadable or unsupported (e.g. HEIC): let the API report it
        return path
    im.thumbnail((max_side, max_side))
    out = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    im.save(out, "JPEG", quality=90)
    out.close()
    return out.name


def analyze(path: str | None):
    """Returns (top3 dict, nutrition df, markdown details, status)."""
    if not path:
        return (*EMPTY[:3], "Upload or capture a food photo first.")
    try:
        with open(shrink(path), "rb") as f:
            r = requests.post(f"{API}/analyze", files={"file": f}, timeout=300)
    except requests.RequestException as e:
        return (*EMPTY[:3], f"API unreachable: {e} (is `python run.py serve` running?)")
    if not r.ok:
        detail = r.json().get("detail", r.text) if r.headers.get("content-type", "").startswith("application/json") else r.text
        return (*EMPTY[:3], f"API error {r.status_code}: {detail}")
    d = r.json()
    top = {t["name"]: t["prob"] for t in d["top3"]}
    n = d["nutrition"]
    if n:
        nut = pd.DataFrame([{"nutrient (per 100 g)": k, "value": ("n/a" if v is None else v)} for k, v in n["per_100g"].items()])
        src = (f"[USDA FoodData Central: {n['matched_description']}]({n['source_url']})" if n["source_url"]
               else "No USDA match for this class (values left empty, not invented).")
    else:
        nut, src = pd.DataFrame(), "No nutrition row."
    v = d["vlm"]
    md = [f"### {d['dish_name']}", src]
    if v:
        md += [f"**What the model sees:** {v['why']}", f"**Visible ingredients:** {', '.join(v['visible_ingredients'])}",
               f"**Portion (estimate, unreliable):** {v['portion_estimate']['description']}, ~{v['portion_estimate']['approx_grams']} g",
               "**Recipe ideas:**", *[f"- **{x['title']}**: {x['summary']}" for x in v["recipes"]]]
    else:
        md.append(f"_VLM unavailable: {d['vlm_error']}_")
    md.append(f"> {d['disclaimer']}")
    t = d["timings_ms"]
    return top, nut, "\n\n".join(md), f"Timings ms: {t}"


HOWTO = """# Food Lens
Photo -> dish (CLIP) -> USDA nutrition -> recipe ideas (SmolVLM). Local, no cloud.

**How to use**
1. Click one of the **sample photos** below (or upload / use the webcam), then press **Analyze**.
2. Wait about 7 s on an Apple Silicon laptop (CPU-only machines are slower): CLIP answers in milliseconds, the VLM takes the rest.
3. Read the result: top-3 box = CLIP probabilities; table = USDA values per 100 g (empty = no USDA match, never invented); text = why, ingredients, portion guess, 2 recipes.

*Needs the API running (`python run.py serve`).*
"""
PROSCONS = """**Pros:** fully local, no training, nutrition grounded in USDA data, per-stage timings shown.

**Cons:** only the 101 Food-101 dishes; portion size is a rough guess; the 500M VLM writes generic text; one photo at a time.

**At scale:** batch CLIP on a GPU, queue the VLM, cache by image hash (see README section 13)."""
EXAMPLES = [[str(p)] for p in sorted((ROOT / "assets").glob("*.jpg"))]


def build() -> gr.Blocks:
    with gr.Blocks(title="Food Lens") as ui:
        gr.Markdown(HOWTO)
        with gr.Row():
            img = gr.Image(sources=["upload", "webcam"], type="filepath", label="Food photo")
            with gr.Column():
                plot = gr.Label(num_top_classes=3, label="CLIP top-3 (probability)")
                table = gr.Dataframe(label="Nutrition per 100 g (USDA)")
        btn = gr.Button("Analyze", variant="primary")
        if EXAMPLES:
            gr.Examples(EXAMPLES, inputs=img, label="Sample photos (click, then Analyze): expect a plausible dish, a nutrition table and 2 recipes",
                        cache_examples=False)
        status = gr.Textbox(label="Status / per-stage timings (ms)", interactive=False, placeholder="Idle. Pick a photo and press Analyze.")
        details = gr.Markdown()
        with gr.Accordion("Pros / cons / at scale", open=False):
            gr.Markdown(PROSCONS)
        btn.click(analyze, img, [plot, table, details, status], show_progress="full")
    return ui


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=cfg["ports"]["ui"], help="UI port (default from config: 7865)")
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    console.setup_logging()
    console.banner("Food Lens UI", f"Gradio page that calls the API at {API}.", f"open http://{a.host}:{a.port} and click a sample photo")
    build().launch(server_name=a.host, server_port=a.port)
