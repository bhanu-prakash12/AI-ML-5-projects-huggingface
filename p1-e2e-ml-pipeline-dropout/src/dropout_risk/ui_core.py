"""Gradio-free logic for the demo UI (kept importable and testable without gradio)."""
from __future__ import annotations

from typing import Any

from dropout_risk.data import RANGES
from dropout_risk.predict import Predictor
from dropout_risk.samples import SAMPLES, UI_FIELDS

TIER_TEXT = {
    "low": "LOW risk: no action suggested by the model.",
    "medium": "MEDIUM risk: worth a check-in.",
    "high": "HIGH risk: at/above the tuned threshold; flag for support outreach.",
}


def is_binary(col: str) -> bool:
    return tuple(RANGES[col]) == (0, 1)


def example_rows() -> list[list[Any]]:
    """Rows for gr.Examples: UI field values for the low / medium / high presets."""
    return [[bool(SAMPLES[k][c]) if is_binary(c) else SAMPLES[k][c] for c in UI_FIELDS]
            for k in ("low", "medium", "high")]


def build_row(values: list[Any], columns: list[str]) -> dict[str, float]:
    """UI values + the 'medium' preset for every feature the form does not show."""
    row = {c: SAMPLES["medium"].get(c, float(RANGES[c][0])) for c in columns}
    for col, v in zip(UI_FIELDS, values, strict=True):
        if col in row:
            row[col] = float(v) if v is not None else row[col]
    return row


def score(values: list[Any], predictor: Predictor) -> dict[str, Any]:
    row = build_row(values, predictor.columns)
    return predictor.predict([row])[0]


def summary_md(pred: dict[str, Any], threshold: float) -> str:
    return (f"### {pred['risk_tier'].upper()} risk  -  P(dropout) = {pred['probability']:.1%}\n"
            f"{TIER_TEXT[pred['risk_tier']]}\n\n"
            f"_Tuned threshold: {threshold:.3f}; model `{pred['model_version']}`._")


def contrib_chart(pred: dict[str, Any]):
    """Horizontal bar chart of the top-3 SHAP contributions (log-odds)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feats = pred["top_features"][::-1]
    fig, ax = plt.subplots(figsize=(6, 2.6))
    ax.barh([f["feature"].replace("curricular_units_", "cu_") for f in feats],
            [f["contribution"] for f in feats],
            color=["#C62828" if f["contribution"] > 0 else "#2E7D32" for f in feats])
    ax.axvline(0, color="#455A64", lw=0.8)
    ax.set_xlabel("SHAP contribution (log-odds); red raises risk, green lowers it")
    fig.tight_layout()
    return fig
