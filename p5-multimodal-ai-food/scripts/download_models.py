"""Download CLIP + SmolVLM at pinned revisions into the HF cache (one-time, needs network).

Inputs:  configs/config.yaml (repo + pinned revision per model).
Outputs: files in the Hugging Face cache (~/.cache/huggingface or %USERPROFILE%\\.cache\\huggingface); prints the paths.
Example: python run.py models
"""
import argparse

from huggingface_hub import snapshot_download

from food_lens import console
from food_lens.config import load_config

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--only", choices=["clip", "vlm"], help="download just one model")
args = ap.parse_args()
console.setup_logging()
console.banner("Download models", "Fetches CLIP (open_clip safetensors) and SmolVLM-500M at pinned revisions.",
               "about 1.5 GB total on first run, instant when cached; afterwards set HF_HUB_OFFLINE=1")
cfg = load_config()["models"]
jobs = [("clip", {"allow_patterns": ["open_clip_model.safetensors", "open_clip_config.json"]}),
        ("vlm", {"ignore_patterns": ["onnx/*", "*.onnx"]})]
jobs = [j for j in jobs if not args.only or j[0] == args.only]
for i, (key, extra) in enumerate(jobs, 1):
    m = cfg[key]
    console.step(i, len(jobs), f"{key}: {m['repo']} @ {m['revision'][:8]}")
    p = snapshot_download(m["repo"], revision=m["revision"], **extra)
    console.ok(f"{key} -> {p}")
console.note("Pinned revisions make the demo reproducible and fully offline afterwards.",
             "First run needs network and disk space.", "Mirror the cache (or bake it into an image) for many machines.")
