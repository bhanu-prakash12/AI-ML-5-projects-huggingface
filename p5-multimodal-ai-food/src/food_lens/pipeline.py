"""Orchestration: image -> CLIP top-3 -> nutrition rows (CSV) -> VLM JSON -> response with timings."""
from __future__ import annotations

import csv
import io
import time

from PIL import Image, ImageOps

from . import vlm
from .clip_zs import ClipClassifier, pretty
from .config import ROOT, load_config

DISCLAIMER = (
    "Estimates only. Dish is a model guess from a photo; nutrition is USDA per 100 g of a matched generic food, "
    "not this exact plate; portion size is a rough visual estimate. Not medical or dietary advice."
)
NUM = ("kcal", "protein_g", "fat_g", "carbs_g", "fiber_g")


def load_nutrition(path=None) -> dict[str, dict]:
    path = path or ROOT / load_config()["paths"]["data"] / "nutrition.csv"
    with open(path, newline="", encoding="utf-8") as f:
        return {r["class"]: r for r in csv.DictReader(f)}


def nutrition_view(row: dict | None) -> dict | None:
    """Numbers come only from the CSV; empty -> None (never invented)."""
    if not row:
        return None
    return {
        "class": row["class"], "fdc_id": row["fdc_id"] or None,
        "matched_description": row["matched_description"] or None, "source_url": row["source_url"] or None,
        "per_100g": {k: (float(row[k]) if row[k] != "" else None) for k in NUM},
    }


def prepare_image(data: bytes, max_side: int) -> Image.Image:
    im = Image.open(io.BytesIO(data))
    im.load()
    im = ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail((max_side, max_side))
    return im


class Engine:
    def __init__(self, cfg: dict | None = None, use_vlm: bool = True):
        self.cfg = cfg or load_config()
        self.nutrition = load_nutrition()
        self.labels = list(self.nutrition)
        self.clip = ClipClassifier(self.labels, self.cfg)
        self.use_vlm = use_vlm
        if use_vlm:
            vlm.load(self.cfg)

    def analyze(self, data: bytes) -> dict:
        t0 = time.perf_counter()
        im = prepare_image(data, self.cfg["api"]["max_image_side"])
        t1 = time.perf_counter()
        top = self.clip.topk([im], self.cfg["api"]["top_k"])[0]
        t2 = time.perf_counter()
        rows = [self.nutrition[c] for c, _ in top if c in self.nutrition]
        out = {"top3": [{"label": c, "name": pretty(c), "prob": round(p, 4)} for c, p in top],
               "vlm": None, "vlm_error": None, "disclaimer": DISCLAIMER}
        dish = top[0][0]
        t3 = t2
        if self.use_vlm:
            vim = im.copy()
            vim.thumbnail((self.cfg["vlm"]["max_image_px"],) * 2)
            res = vlm.analyze(vim, top, rows)
            t3 = time.perf_counter()
            if res["ok"]:
                r = res["result"]
                r["portion_estimate"]["is_estimate"] = True
                out["vlm"] = r
                cand = {c for c, _ in top}
                norm = r["dish"].strip().lower().replace(" ", "_")
                dish = norm if norm in cand else dish  # VLM may only choose among CLIP candidates
                out["vlm_attempts"] = res["attempts"]
            else:
                out["vlm_error"] = res["error"]
        out["dish"] = dish
        out["dish_name"] = pretty(dish)
        out["nutrition"] = nutrition_view(self.nutrition.get(dish))
        out["timings_ms"] = {
            "decode_resize": round((t1 - t0) * 1000), "clip": round((t2 - t1) * 1000),
            "vlm": round((t3 - t2) * 1000), "total": round((time.perf_counter() - t0) * 1000),
        }
        return out
