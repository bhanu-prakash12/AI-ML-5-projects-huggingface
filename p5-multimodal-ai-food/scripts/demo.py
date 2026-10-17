"""Narrated demo: CLIP -> USDA rows -> SmolVLM JSON on cached sample photos, with per-stage timings.

Inputs:  data/eval_sample/*.jpg (cached by `eval`), else assets/*.jpg; cached models (HF_HUB_OFFLINE=1).
Outputs: console tables; reports/vlm_demo.json + reports/vlm_demo.md (real measurements only).
Example: python run.py demo --n 3
"""
import argparse
import json
import logging
import time

from food_lens import console
from food_lens.config import ROOT, load_config
from food_lens.pipeline import Engine

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--n", type=int, default=6, help="number of images to analyze (default 6)")
ap.add_argument("--no-write", action="store_true", help="do not overwrite reports/vlm_demo.*")
args = ap.parse_args()
console.setup_logging(logging.WARNING)

console.banner("Food Lens demo", "photo -> CLIP top-3 -> USDA nutrition row -> SmolVLM (dish, why, ingredients, recipes)",
               "one table row per image; the VLM stage dominates latency (seconds), CLIP is milliseconds")
cfg = load_config()
files = sorted((ROOT / cfg["paths"]["data"] / "eval_sample").glob("*.jpg")) or sorted((ROOT / "assets").glob("*.jpg"))
if not files:
    console.fail("no images found: run `python run.py eval` once or keep assets/*.jpg")
    raise SystemExit(1)
pick = files[:: max(1, len(files) // args.n)][: args.n]

console.step(1, 4, "loading CLIP + SmolVLM (first load takes a few seconds; downloads never happen offline)")
t0 = time.perf_counter()
eng = Engine(cfg)
load_s = round(time.perf_counter() - t0, 1)
console.ok(f"models loaded in {load_s}s")

console.step(2, 4, f"analyzing {len(pick)} photos")
rows = []
for f in pick:
    r = eng.analyze(f.read_bytes())
    rows.append({"file": f.name, **{k: r.get(k) for k in ("dish", "top3", "vlm", "vlm_error", "vlm_attempts", "nutrition", "timings_ms")}})
    console.ok(f"{f.name}: {r['dish']} in {r['timings_ms']['total']} ms")

console.step(3, 4, "per-stage results")
console.table(
    [[r["file"], f"{r['top3'][0]['label']} ({r['top3'][0]['prob']})", r["dish"], bool(r["vlm"]), r.get("vlm_attempts"),
      ((r["nutrition"] or {}).get("per_100g", {}).get("kcal")), r["timings_ms"]["clip"], r["timings_ms"]["vlm"], r["timings_ms"]["total"]]
     for r in rows],
    ["image", "CLIP #1", "dish", "JSON", "tries", "kcal", "clip", "vlm", "total"], "Demo results (kcal per 100 g; times in ms)")

valid = sum(1 for r in rows if r["vlm"])
retried = sum(1 for r in rows if r.get("vlm_attempts") == 2)
tot = sorted(r["timings_ms"]["total"] for r in rows)
summary = {"n": len(rows), "valid_json": valid, "needed_retry": retried, "load_s": load_s,
           "total_ms_median": tot[len(tot) // 2], "total_ms_max": tot[-1],
           "vlm_ms_median": sorted(r["timings_ms"]["vlm"] for r in rows)[len(rows) // 2]}
console.kv_table(summary, "Summary")

console.step(4, 4, "writing reports" if not args.no_write else "skipping report write (--no-write)")
if not args.no_write:
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "vlm_demo.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    md = ["# VLM demo (SmolVLM-500M, real run)", "", f"- {summary}", "",
          "| image | CLIP top-1 | final dish | JSON ok | attempts | kcal/100g (USDA) | total ms | vlm ms |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        n = (r["nutrition"] or {}).get("per_100g", {})
        md.append(f"| {r['file']} | {r['top3'][0]['label']} ({r['top3'][0]['prob']}) | {r['dish']} | {bool(r['vlm'])} | "
                  f"{r.get('vlm_attempts')} | {n.get('kcal')} | {r['timings_ms']['total']} | {r['timings_ms']['vlm']} |")
    (ROOT / "reports" / "vlm_demo.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    console.ok("wrote reports/vlm_demo.json and reports/vlm_demo.md")
console.note(
    "CLIP zero-shot needs no training; nutrition comes only from the USDA CSV, so the VLM cannot invent numbers.",
    "SmolVLM-500M text is generic and its portion guess is weak; one photo at a time; the VLM may only pick among CLIP's top-3.",
    "A GPU server would batch CLIP and queue the VLM (see README section 13); on CPU-only Windows the VLM stage is slower.")
