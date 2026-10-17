"""p4 Fraud model monitoring: replay of the recorded run (scenario B). Simulate/Retrain need ~1 GB of data: not shipped."""
from __future__ import annotations

import json

import gradio as gr
import pandas as pd

from hub import paths  # noqa: F401
from hub.loader import project_dir
from hub.tabs.common import header

HOWTO = """**Story:** a fraud model is serving traffic. A new fraud pattern appears on day 6. Can monitoring notice it?

**How to use:** this tab **replays a real recorded run** (scenario B, 10 days of traffic, produced by the project's
simulator and Evidently monitor). Pick a window to open its full Evidently report. Running new simulations needs the
119 MB dataset, so it is left to the local project (`python run.py ui` in `p4-model-monitoring-drift/`).

**How to read the plot.** Top: share of features that drifted per day vs the alert threshold. Bottom: PR-AUC per day vs
the baseline (labels arrive a day late). Red bands = alert windows."""

TRADEOFFS = ("- **Pros:** drift share needs no labels, so it alerts the same day; PR-AUC is robust to the 0.2% fraud rate; "
             "the promotion gate stops noisy retrains.\n- **Cons:** labels arrive late; about 10 frauds a day make PR-AUC "
             "noisy; synthetic scenarios are easier than real drift.\n- **At scale:** metrics store (Prometheus/Grafana), "
             "scheduled monitor, shadow scoring, approval before promotion.")


def _run_dir():
    return project_dir("p4") / "reports" / "scenB"


def _status() -> pd.DataFrame:
    p = _run_dir() / "status.json"
    return pd.DataFrame(json.loads(p.read_text())) if p.exists() else pd.DataFrame()


def _links(df: pd.DataFrame) -> str:
    out = []
    for w, rep, a in zip(df.window, df.report, df.alert):
        if (_run_dir() / rep).exists():
            out.append(f'<a href="/gradio_api/file={(_run_dir() / rep).as_posix()}" target="_blank">'
                       f'window {w}{" (ALERT)" if a else ""}</a>')
    return ("Evidently reports (open in a new tab): " + " &middot; ".join(out)) if out else ""


def _retrain() -> str:
    p = _run_dir() / "retrain.json"
    if not p.exists():
        return ""
    from fraud_monitor import narrate

    return "\n\n".join(narrate.explain_retrain(json.loads(p.read_text())))


def render() -> None:
    from fraud_monitor import dashboard

    header("p4", "Fraud model monitoring", "Drift detection with Evidently + PSI, alert rule, champion/challenger retrain.",
           "p4-model-monitoring-drift", kind="Replay of a real run")
    gr.Markdown(HOWTO)
    df = _status()
    if df.empty:
        gr.Markdown("Recorded run not found in this build.")
        return
    gr.Plot(dashboard.figure(df))
    gr.Dataframe(dashboard.table(df), label="One row per monitored day (window)")
    gr.HTML(_links(df))
    txt = _retrain()
    if txt:
        with gr.Accordion("Retrain decision (champion vs challenger)", open=True):
            gr.Markdown(txt)
    with gr.Accordion("Pros / cons / at scale", open=False):
        gr.Markdown(TRADEOFFS)
