"""Gradio demo: student features -> dropout probability, tier, top-3 SHAP chart (port 7861)."""
from __future__ import annotations

import _boot  # noqa: F401  (adds src/ to sys.path)
import argparse

import gradio as gr

from dropout_risk import console
from dropout_risk.config import load_config, project_path
from dropout_risk.data import RANGES
from dropout_risk.predict import Predictor
from dropout_risk.ui_core import contrib_chart, example_rows, is_binary, score, summary_md
from dropout_risk.samples import UI_FIELDS

HOW_TO = """
## Student dropout risk
**How to use:** click one of the three examples below (low / medium / high risk) or edit the
numbers, then press **Predict**. Features not shown here keep a typical preset value.

**What to expect:** a probability, a risk tier, and the three features that pushed the score most
(SHAP, log-odds). **Late-stage model** (end of year 2): not an early-warning system, and it must
not be the only input to a decision about a real person.
"""
PROS_CONS = """
- **Pros:** milliseconds per score, explainable per student, no GPU, runs offline.
- **Cons:** precision about 0.65 (1 in 3 flags is a false alarm); late features; one institution.
- **At scale:** same model behind the FastAPI `/predict/batch`; retrain on a schedule via MLflow.
"""


def build(predictor: Predictor) -> gr.Blocks:
    thr = predictor.threshold

    def run(*vals):
        pred = score(list(vals), predictor)
        return summary_md(pred, thr), contrib_chart(pred), "Done."

    with gr.Blocks(title="Student Dropout Risk") as demo:
        gr.Markdown(HOW_TO)
        inputs = []
        with gr.Row():
            for c in UI_FIELDS:
                lo, hi = RANGES[c]
                label = c.replace("curricular_units_", "").replace("_", " ")
                if is_binary(c):
                    inputs.append(gr.Checkbox(label=label))
                else:
                    inputs.append(gr.Number(label=f"{label} ({lo}..{hi})", minimum=lo, maximum=hi))
        btn = gr.Button("Predict", variant="primary")
        status = gr.Markdown("Ready.")
        with gr.Row():
            out_md = gr.Markdown()
            out_plot = gr.Plot(label="Top-3 SHAP contributions")
        btn.click(run, inputs, [out_md, out_plot, status])
        gr.Examples(example_rows(), inputs, label="Examples: low / medium / high risk student")
        with gr.Accordion("Pros / cons / at scale", open=False):
            gr.Markdown(PROS_CONS)
    return demo


def main() -> None:
    port = load_config()["ui"]["port"]
    ap = argparse.ArgumentParser(
        description="Launch the Gradio dropout-risk form.",
        epilog="Needs a trained model (python run.py train). Open http://127.0.0.1:7861. "
               "Sample: python scripts/ui.py")
    ap.add_argument("--port", type=int, default=port)
    args = ap.parse_args()
    console.setup_logging()
    console.banner("Gradio UI", "Student form -> probability, tier, SHAP chart.",
                   f"page at http://127.0.0.1:{args.port} (Ctrl+C to stop).")
    predictor = Predictor.load(project_path("artifacts") / "model")
    build(predictor).launch(server_name="127.0.0.1", server_port=args.port, show_error=True)


if __name__ == "__main__":
    main()
