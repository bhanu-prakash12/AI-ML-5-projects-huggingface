"""SmolVLM wrapper: lazy singleton, JSON-only prompt -> extract -> pydantic -> one retry -> typed error."""
from __future__ import annotations

import json
import logging
import re
from typing import Callable

from PIL import Image
from pydantic import ValidationError

from .config import load_config
from .device import empty_cache, get_device, inference_dtype
from .schema import VlmResult, extract_json

log = logging.getLogger(__name__)
_STATE: dict = {}

SCHEMA_HINT = (
    '{"dish": "<one of the candidates>", "why": "<one sentence>", "visible_ingredients": ["..."], '
    '"portion_estimate": {"description": "<e.g. one slice, about 150 g>", "approx_grams": 150}, '
    '"recipes": [{"title": "...", "summary": "<one sentence>"}, {"title": "...", "summary": "<one sentence>"}]}'
)


def load(cfg: dict | None = None) -> dict:
    """Lazy singleton: fp16 on MPS, fp32 on CPU."""
    if _STATE:
        return _STATE
    from transformers import AutoModelForVision2Seq, AutoProcessor

    cfg = cfg or load_config()
    m = cfg["models"]["vlm"]
    device = get_device()
    dtype = inference_dtype(device)
    log.info("loading SmolVLM on %s (%s); first load can take a while", device, dtype)
    proc = AutoProcessor.from_pretrained(m["repo"], revision=m["revision"], do_image_splitting=cfg["vlm"]["image_splitting"])
    model = AutoModelForVision2Seq.from_pretrained(m["repo"], revision=m["revision"], torch_dtype=dtype).to(device).eval()
    _STATE.update(proc=proc, model=model, device=device, cfg=cfg)
    return _STATE


def unload() -> None:
    import gc

    dev = _STATE.get("device", "cpu")
    _STATE.clear()
    gc.collect()
    empty_cache(dev)


PREFILL = '{"dish": "'  # small VLMs ignore "JSON only"; starting the assistant turn inside the object forces the format
def generate(image: Image.Image, prompt: str, max_new_tokens: int | None = None, prefill: str = PREFILL,
             stop: str | None = None) -> str:
    import torch

    s = load()
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = s["proc"].apply_chat_template(messages, add_generation_prompt=True) + prefill
    inputs = s["proc"](text=text, images=[image.convert("RGB")], return_tensors="pt").to(s["device"])
    with torch.inference_mode():
        out = s["model"].generate(**inputs, max_new_tokens=max_new_tokens or s["cfg"]["vlm"]["max_new_tokens"], do_sample=False,
                             **({"stop_strings": [stop], "tokenizer": s["proc"].tokenizer} if stop else {}))
    return prefill + s["proc"].batch_decode(out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0].strip()


def _ask(image: Image.Image, question: str, tokens: int) -> str:
    """One short free-text answer (no JSON syntax: the 500M VLM copies any template it is shown)."""
    return generate(image, question, tokens, prefill="").strip().splitlines()[0].strip(" \"'")


def _sentence(text: str) -> str:
    return text.rsplit(".", 1)[0] + "." if "." in text else text


def generate_json(image: Image.Image, prompt: str) -> str:
    """A 500M VLM cannot reliably emit the whole schema at once. Ask one short question per field, assemble the
    JSON in code (CLAUDE.md rule still applies: analyze() validates with pydantic and retries once with the error)."""
    base, _, tail = prompt.partition(" Reply with ONLY")
    err = tail.partition("\n\n")[2]  # retry suffix appended by analyze()
    cands = re.findall(r"([a-z0-9_]+) \(CLIP", base)
    ask = lambda q, n: _ask(image, f"{q} {err}".strip(), n)  # noqa: E731
    raw = ask(f"Candidate dishes: {', '.join(cands)}. Which one is shown in the photo? Answer with just its name.", 10)
    dish = next((c for c in cands if c in raw.lower().replace(" ", "_")), cands[0] if cands else raw)
    nice = dish.replace("_", " ")
    out: dict = {"dish": dish}
    out["why"] = _sentence(ask(f"{base.split('Pick the dish')[0]}Describe what you can see in this photo in one sentence.", 40))
    ing = ask(f"List the 3 to 5 main ingredients visible in this {nice}, comma separated, nothing else.", 30)
    out["visible_ingredients"] = [x.strip(" .") for x in ing.split(",") if x.strip(" .")][:6]
    size = ask(f"Is the portion of {nice} in the photo small, medium or large? Answer with one word.", 4).strip(" .").lower()
    desc = f"{size} portion (visual guess)" if size in ("small", "medium", "large") else "portion size unclear"
    g = re.search(r"\d+", ask(f"Estimate the weight of the {nice} in grams. Answer with a number only.", 6))
    out["portion_estimate"] = {"description": desc, "approx_grams": int(g.group()) if g else None}
    recipes: list[dict] = []
    for _ in range(2):
        avoid = f" Not {recipes[0]['title']}." if recipes else ""
        title = ask(f"Name one simple recipe that uses {nice}, in 2 to 5 words.{avoid}", 10).strip(" .*:")
        summ = ask(f"Describe the recipe '{title}' in one short sentence.", 35)
        recipes.append({"title": title or nice, "summary": _sentence(summ)})
    out["recipes"] = recipes
    return json.dumps(out)


def build_prompt(candidates: list[tuple[str, float]], nutrition: list[dict]) -> str:
    cand = "; ".join(f"{c} (CLIP {p:.2f})" for c, p in candidates)
    facts = "; ".join(
        f"{r['class']}: {r.get('kcal') or '?'} kcal, {r.get('protein_g') or '?'} g protein per 100 g" for r in nutrition
    )
    return (
        "You are a food assistant. Look at the photo. "
        f"Candidate dishes from an image classifier: {cand}. Reference facts (USDA, per 100 g): {facts}. "
        "Pick the dish that best matches the photo from the candidates. "
        "Do NOT state any calorie or nutrient numbers yourself. "
        f"Reply with ONLY one JSON object, no markdown, in exactly this shape: {SCHEMA_HINT}"
    )


def analyze(image: Image.Image, candidates: list[tuple[str, float]], nutrition: list[dict],
            gen: Callable[[Image.Image, str], str] | None = None) -> dict:
    """Returns {"ok": True, "result": VlmResult-dict, "attempts": n, "raw": str} or {"ok": False, "error":..., "raw":...}."""
    gen = gen or generate_json
    prompt = build_prompt(candidates, nutrition)
    raw, err = "", ""
    for attempt in (1, 2):
        p = prompt if attempt == 1 else (
            f"{prompt}\n\nYour previous reply was invalid: {err}. Reply again with ONLY the corrected JSON object."
        )
        raw = gen(image, p)
        try:
            res = VlmResult.model_validate(extract_json(raw))
            return {"ok": True, "result": res.model_dump(), "attempts": attempt, "raw": raw}
        except (ValueError, ValidationError, json.JSONDecodeError) as e:
            err = str(e).replace("\n", " ")[:300]
            log.warning("VLM output invalid (attempt %d): %s", attempt, err)
    return {"ok": False, "error": f"invalid VLM output after retry: {err}", "raw": raw[:500], "attempts": 2}
