"""Emergency-keyword refusal (runs BEFORE the model) and answer disclaimer."""
from __future__ import annotations

import re

EMERGENCY = re.compile(
    r"\b(chest pain|heart attack|stroke|can'?t breathe|cannot breathe|difficulty breathing|overdose|"
    r"suicid\w*|kill myself|end my life|self[- ]harm|severe bleeding|unconscious|not breathing|"
    r"anaphyla\w*|poison(ed|ing)?|seizure)\b", re.I)

REFUSAL = ("This sounds like it could be an emergency. I can't help with urgent situations. "
           "Call your local emergency number (911 in the US, 112 in the EU) or go to the nearest emergency "
           "department now. If you are thinking about harming yourself, call or text 988 (US) or your local crisis line.")


def is_emergency(text: str) -> bool:
    return bool(EMERGENCY.search(text))
