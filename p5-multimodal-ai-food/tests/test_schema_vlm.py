import json

import pytest
from PIL import Image

from food_lens import vlm
from food_lens.schema import extract_json

GOOD = {
    "dish": "pizza", "why": "round crust with cheese", "visible_ingredients": ["cheese", "tomato"],
    "portion_estimate": {"description": "two slices", "approx_grams": 200},
    "recipes": [{"title": "Margherita", "summary": "Classic."}, {"title": "Pita pizza", "summary": "Quick."}],
}
IMG = Image.new("RGB", (8, 8))
CANDS = [("pizza", 0.8), ("lasagna", 0.1), ("nachos", 0.05)]


def test_extract_json_first_object_with_noise():
    assert extract_json('Sure! ```json\n{"a": "x}y", "b": {"c": 1}}\n``` trailing {"z":1}') == {"a": "x}y", "b": {"c": 1}}


def test_extract_json_errors():
    with pytest.raises(ValueError):
        extract_json("no braces")
    with pytest.raises(ValueError):
        extract_json('{"a": 1')


def test_analyze_ok_first_try():
    out = vlm.analyze(IMG, CANDS, [], gen=lambda im, p: json.dumps(GOOD))
    assert out["ok"] and out["attempts"] == 1 and out["result"]["portion_estimate"]["is_estimate"] is True


def test_analyze_retry_appends_error():
    prompts = []

    def gen(im, p):
        prompts.append(p)
        return "not json" if len(prompts) == 1 else json.dumps(GOOD)

    out = vlm.analyze(IMG, CANDS, [], gen=gen)
    assert out["ok"] and out["attempts"] == 2 and "previous reply was invalid" in prompts[1]


def test_analyze_typed_error_after_retry():
    bad = dict(GOOD, recipes=[GOOD["recipes"][0]])  # only 1 recipe -> validation error
    out = vlm.analyze(IMG, CANDS, [], gen=lambda im, p: json.dumps(bad))
    assert not out["ok"] and "invalid VLM output" in out["error"] and out["attempts"] == 2


def test_prompt_forbids_numbers_and_lists_candidates():
    p = vlm.build_prompt(CANDS, [{"class": "pizza", "kcal": "266", "protein_g": "11"}])
    assert "Do NOT state any calorie" in p and "pizza (CLIP 0.80)" in p and "266 kcal" in p
