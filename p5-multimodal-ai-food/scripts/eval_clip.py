"""Phase 1: CLIP zero-shot top-1/top-5 on a streamed Food-101 validation sample -> reports/clip_eval.json.
Sample images are cached (resized, JPEG) in data/eval_sample/ so reruns need no network.

Inputs:  data/eval_sample/ (cached) or a streamed ethz/food101 validation sample (network, first run only); cached CLIP.
Outputs: reports/clip_eval.json (top-1, top-5, seconds/image, device); data/labels.json.
Example: python run.py eval --n 300
"""
import argparse
import json
import sys
import time

from datasets import load_dataset
from PIL import Image

from food_lens import console
from food_lens.clip_zs import ClipClassifier
from food_lens.config import ROOT, load_config

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--n", type=int, help="images to sample if the cache is empty (default: config eval.n_images)")
ap.add_argument("--no-write", action="store_true", help="print results but keep reports/clip_eval.json unchanged")
args = ap.parse_args()
console.setup_logging()
console.banner("CLIP zero-shot evaluation", "Classifies a balanced Food-101 validation sample with 101 text prompts (no training).",
               "top-1 / top-5 accuracy, seconds per image and the device used")
cfg = load_config()
N = args.n or cfg["eval"]["n_images"]
sample_dir = ROOT / cfg["paths"]["data"] / "eval_sample"
meta = sample_dir / "meta.json"
labels_file = ROOT / cfg["paths"]["data"] / "labels.json"


def peak_rss_mb() -> float | None:
    try:
        import resource  # POSIX only
    except ImportError:
        return None  # Windows: not measured
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / (1e6 if sys.platform == "darwin" else 1e3), 0)  # macOS bytes, Linux KB


def fetch() -> tuple[list[str], list[dict]]:
    sample_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("ethz/food101", split=cfg["eval"]["split"], streaming=True)
    names = ds.features["label"].names
    labels_file.write_text(json.dumps(names))
    per_class_cap = -(-N // len(names))  # ceil -> balanced sample
    counts, items = {}, []
    for ex in ds.shuffle(seed=cfg["seed"], buffer_size=2000):
        lab = ex["label"]
        if counts.get(lab, 0) >= per_class_cap:
            continue
        counts[lab] = counts.get(lab, 0) + 1
        im = ex["image"].convert("RGB")
        im.thumbnail((512, 512))
        fn = f"{len(items):04d}.jpg"
        im.save(sample_dir / fn, quality=88)
        items.append({"file": fn, "label": names[lab]})
        if len(items) >= N:
            break
    meta.write_text(json.dumps({"names": names, "items": items}))
    return names, items


console.step(1, 3, "loading the cached sample (or streaming it once from Hugging Face)")
if meta.exists():
    d = json.loads(meta.read_text())
    names, items = d["names"], d["items"]
else:
    names, items = fetch()
labels_file.write_text(json.dumps(names))

console.step(2, 3, f"loading CLIP and scoring {len(items)} images")
clf = ClipClassifier(names, cfg)
t0 = time.time()
top1 = top5 = 0
for i in range(0, len(items), 16):
    batch = items[i : i + 16]
    ims = [Image.open(sample_dir / b["file"]) for b in batch]
    for b, pred in zip(batch, clf.topk(ims, 5)):
        p = [x[0] for x in pred]
        top1 += p[0] == b["label"]
        top5 += b["label"] in p
dt = time.time() - t0
n = len(items)
out = {
    "n_images": n, "n_classes": len(names), "top1": round(top1 / n, 4), "top5": round(top5 / n, 4),
    "templates": len(__import__("food_lens.clip_zs", fromlist=["TEMPLATES"]).TEMPLATES),
    "device": clf.device, "seconds_per_image": round(dt / n, 4),
    "peak_rss_mb": peak_rss_mb(),
    "note": "balanced sample from streamed ethz/food101 validation, shuffled seed 42",
}
console.step(3, 3, "results")
console.kv_table(out, "CLIP zero-shot")
if not args.no_write:
    (ROOT / cfg["paths"]["reports"] / "clip_eval.json").write_text(json.dumps(out, indent=1))
    console.ok("wrote reports/clip_eval.json")
console.note("No training and no labels beyond class names; top-5 shows the right dish is usually in the shortlist the VLM sees.",
             "Food-101 has 101 fixed classes; mixed plates and other cuisines are forced into one of them.",
             "Fine-tune or linear-probe CLIP on your own classes once zero-shot accuracy stops being enough.")
sys.exit(0)
