"""p1 Student dropout risk: reuses the project's own Gradio form (in-process LightGBM, CPU only)."""
from __future__ import annotations

from hub import paths  # noqa: F401
from hub.loader import load_script
from hub.tabs.common import header


def render() -> None:
    from dropout_risk.config import project_path
    from dropout_risk.predict import Predictor

    header("p1", "Student dropout risk", "Tabular ML: LightGBM + per-student SHAP explanations. CPU only, ms per score.",
           "p1-e2e-ml-pipeline-dropout")
    ui = load_script("p1", "ui")
    ui.build(Predictor.load(project_path("artifacts") / "model")).render()
