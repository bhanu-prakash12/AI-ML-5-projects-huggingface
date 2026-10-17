"""Three preset students (low / medium / high risk) used by the UI, the demo and the docs.

Values stay inside the validated ranges (data.RANGES). They are illustrative inputs, not real
people; the risk tier shown is whatever the trained model returns for them.
"""
from __future__ import annotations

from dropout_risk.data import RANGES
from dropout_risk.features import FEATURES

# Baseline: every feature at its range minimum, then overrides per persona.
_BASE: dict[str, float] = {c: float(RANGES[c][0]) for c in FEATURES}
_BASE |= {"application_mode": 1.0, "course": 9500.0, "daytime_evening_attendance": 1.0,
          "previous_qualification": 1.0, "previous_qualification_grade": 130.0,
          "admission_grade": 125.0, "tuition_fees_up_to_date": 1.0,
          "unemployment_rate": 10.8, "inflation_rate": 1.4, "gdp": 1.7}


def _sem(sem: str, enrolled: int, evals: int, approved: int, grade: float) -> dict[str, float]:
    p = f"curricular_units_{sem}_sem_"
    return {p + "enrolled": float(enrolled), p + "evaluations": float(evals),
            p + "approved": float(approved), p + "grade": grade}


SAMPLES: dict[str, dict[str, float]] = {
    "low": _BASE | _sem("1st", 6, 8, 6, 14.0) | _sem("2nd", 6, 8, 6, 14.0)
    | {"scholarship_holder": 1.0},
    "medium": _BASE | _sem("1st", 6, 8, 5, 12.0) | _sem("2nd", 6, 8, 5, 12.0),
    "high": _BASE | _sem("1st", 6, 6, 1, 8.0) | _sem("2nd", 6, 4, 0, 0.0)
    | {"debtor": 1.0, "tuition_fees_up_to_date": 0.0},
}

# Feature subset shown in the UI form (the rest keep the preset value).
UI_FIELDS: list[str] = [
    "curricular_units_1st_sem_approved", "curricular_units_1st_sem_grade",
    "curricular_units_2nd_sem_enrolled", "curricular_units_2nd_sem_evaluations",
    "curricular_units_2nd_sem_approved", "curricular_units_2nd_sem_grade",
    "admission_grade", "tuition_fees_up_to_date", "debtor", "scholarship_holder",
]
