"""Pydantic output schema + JSON extraction for the VLM."""
from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field, field_validator, model_validator


class Recipe(BaseModel):
    title: str = Field(min_length=2)
    summary: str = Field(min_length=2)


class Portion(BaseModel):
    description: str
    approx_grams: int | None = Field(default=None, ge=0, le=3000)
    is_estimate: bool = True  # always flagged: portion estimates from a photo are unreliable

    @field_validator("approx_grams", mode="before")
    @classmethod
    def _grams(cls, v):
        if isinstance(v, str):
            m = re.search(r"\d+", v)
            return int(m.group()) if m else None
        return v


class VlmResult(BaseModel):
    dish: str = Field(min_length=1)
    why: str
    visible_ingredients: list[str]
    portion_estimate: Portion
    recipes: list[Recipe] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def _no_nutrient_claims(self):
        # Nutrition numbers must come from the USDA CSV, never from the VLM.
        if re.search(r"kcal|calorie|protein", (self.why + " " + self.portion_estimate.description).lower()):
            raise ValueError("why/portion_estimate must not mention calories or nutrients")
        return self


def extract_json(text: str) -> dict:
    """Return the first balanced {...} object in text (string-aware). Raises ValueError."""
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("unterminated JSON object")


class Candidate(BaseModel):
    label: str
    name: str
    prob: float


class NutritionView(BaseModel):
    class_: str = Field(alias="class")
    fdc_id: str | None = None
    matched_description: str | None = None
    source_url: str | None = None
    per_100g: dict[str, float | None]

    model_config = {"populate_by_name": True}


class Timings(BaseModel):
    decode_resize: int
    clip: int
    vlm: int
    total: int


EXAMPLE_RESPONSE = {
    "top3": [{"label": "pizza", "name": "Pizza", "prob": 0.91}, {"label": "lasagna", "name": "Lasagna", "prob": 0.04},
             {"label": "focaccia", "name": "Focaccia", "prob": 0.01}],
    "vlm": None, "vlm_error": None, "vlm_attempts": None,
    "disclaimer": "Estimates only. Not medical or dietary advice.",
    "dish": "pizza", "dish_name": "Pizza",
    "nutrition": {"class": "pizza", "fdc_id": "0", "matched_description": "(from USDA CSV)", "source_url": None,
                  "per_100g": {"kcal": None, "protein_g": None, "fat_g": None, "carbs_g": None, "fiber_g": None}},
    "timings_ms": {"decode_resize": 0, "clip": 0, "vlm": 0, "total": 0},
}


class AnalyzeResponse(BaseModel):
    """Shape of POST /analyze. The example is a schema illustration (placeholders), not a measured result."""

    top3: list[Candidate]
    dish: str
    dish_name: str
    nutrition: NutritionView | None = None
    vlm: VlmResult | None = None
    vlm_error: str | None = None
    vlm_attempts: int | None = None
    disclaimer: str
    timings_ms: Timings

    model_config = {"json_schema_extra": {"examples": [EXAMPLE_RESPONSE]}}
