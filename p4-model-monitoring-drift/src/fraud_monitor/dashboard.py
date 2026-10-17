"""Dashboard helpers (pure functions are unit-tested; Gradio only wires them)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import config, narrate  # noqa: E402


LATEST_DEMO = "Latest demo run (reports/demo)"
SAVED = "Saved main run (reports/)"


def available_runs() -> dict[str, Path]:
    """Result folders that contain a status.json: the live demo run first, then saved runs under reports/."""
    rep = config.path("reports")
    runs: dict[str, Path] = {}
    for label, d in ((LATEST_DEMO, rep / "demo"), (SAVED, rep)):
        if (d / "status.json").exists():
            runs[label] = d
    for d in sorted(p for p in rep.iterdir() if p.is_dir() and p.name != "demo" and (p / "status.json").exists()):
        runs[f"Saved run: reports/{d.name}"] = d
    return runs


def load_status(out: Path | None = None) -> pd.DataFrame:
    p = (out or config.path("reports")) / "status.json"
    return pd.DataFrame(json.loads(p.read_text())) if p.exists() else pd.DataFrame()


def figure(df: pd.DataFrame):
    cfg = config.load()["monitor"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    if df.empty:
        a1.text(0.5, 0.5, "no status.json yet - click 1. Simulate then 2. Monitor", ha="center")
        return fig
    a1.plot(df.window, df.drift_share, "o-", color="#6A1B9A", label="drift share")
    a1.axhline(cfg["drift_share_threshold"], ls="--", color="grey", label="threshold")
    a1.set_ylabel("share of drifted features")
    a2.plot(df.window, df.pr_auc, "s-", color="#2E7D32", label="PR-AUC (labelled windows)")
    base = df.baseline_pr_auc.iloc[0]
    a2.axhline(base, color="grey", label="baseline")
    a2.axhline(base * (1 - cfg["pr_auc_drop_threshold"]), ls="--", color="#C62828", label="-10% alert line")
    a2.set_ylabel("PR-AUC")
    a2.set_xlabel("window (simulated day)")
    for w in df[df.alert].window:
        a1.axvline(w, color="#C62828", alpha=0.25)
        a2.axvline(w, color="#C62828", alpha=0.25)
    a1.legend(loc="upper left", fontsize=7)
    a2.legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    return fig


def table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    t = df[["window", "n", "n_fraud", "drift_share", "psi_score", "pr_auc", "alert"]].copy()
    t["why"] = df.reasons.map(lambda r: "; ".join(r)).where(df.alert, df.perf_skipped.fillna(""))
    return t.round(3)


def links(df: pd.DataFrame, out: Path | None = None) -> str:
    rep = out or config.path("reports")
    return " ".join(f'<a href="/gradio_api/file={(rep / r).as_posix()}" target="_blank">window {w}{" (ALERT)" if a else ""}</a>'
                    for w, r, a in zip(df.window, df.report, df.alert)) if not df.empty else ""


INTRO = """
## Fraud model monitoring - click-driven demo
**Story:** a fraud model is serving traffic. A new pattern appears on day 6. Can we notice, and fix it?

1. **Simulate** replays 10 days of traffic through the API (about 1 min). Pick a scenario below.
2. **Monitor** compares every day with the reference and applies the alert rule (about 1 min).
3. **Retrain** trains a challenger and promotes it only if it clearly beats the champion (about 1 min).

Everything shown comes from the run you just did (choose *Latest demo run*), or from a saved run under `reports/`.
"""

PLOT_HELP = """
**How to read the plot.** Top: share of features that drifted per day vs the alert threshold (dashed).
Bottom: PR-AUC of the model on that day (needs labels, which arrive a day late) vs the baseline and the -10% line.
Red bands = windows where an alert fired. Drift reacts the same day; PR-AUC confirms real damage a day later.
"""

SCENARIO_HELP = {
    "A": "A = Amount x3 with noise (data drift). Not recorded in reports/ yet: run it and compare.",
    "B": "B = new fraud cluster injected from the start day (drift + decay). Recorded run: alert on the injection window.",
    "C": "C = concept flip in a segment (labels change, inputs barely move). Not recorded in reports/ yet: run it.",
    "none": "none = clean traffic (checks for false alarms).",
}

TRADEOFFS = """
**Pros:** drift share needs no labels, so it alerts the same day; PR-AUC is robust to the 0.2% fraud base rate;
the promotion gate stops noisy retrains.
**Cons:** labels arrive late, PR-AUC on about 10 frauds a day is noisy, the synthetic scenarios are easier than real drift.
**At scale:** replace SQLite + HTML with a metrics store (Prometheus/Grafana), schedule the monitor, add shadow
scoring and approval steps before promotion.
"""


def build():
    import gradio as gr

    from . import pipeline

    def show(label: str | None):
        runs = available_runs()
        out = runs.get(label or "") or next(iter(runs.values()), None)
        d = load_status(out) if out else pd.DataFrame()
        return figure(d), table(d), links(d, out) if out else ""

    def choices(selected: str | None = None):
        runs = list(available_runs())
        return gr.update(choices=runs, value=selected if selected in runs else (runs[0] if runs else None))

    def do_simulate(scenario, start_day, progress=gr.Progress()):
        t0 = time.time()
        res = pipeline.simulate(scenario, int(start_day), lambda d, n: progress(d / n, desc=f"day {d}/{n}"))
        return (f"Simulated 10 days with scenario **{scenario}** from day {int(start_day)}: {res['sent']} predictions, "
                f"{res['labels_fed']} labels fed back, {time.time() - t0:.0f}s. Next: **2. Monitor**.")

    def do_monitor(progress=gr.Progress()):
        if not pipeline.demo_db().exists():
            return "Run **1. Simulate** first.", choices()
        progress(0.1, desc="running Evidently on every window (about 1 min)")
        t0 = time.time()
        rows = pipeline.run_monitor()
        lines = [narrate.window_line(r)[1] for r in rows]
        return (f"Monitored {len(rows)} windows in {time.time() - t0:.0f}s:\n\n" + "\n\n".join(f"- {x}" for x in lines)
                + "\n\nNext: **3. Retrain**."), choices(LATEST_DEMO)

    def do_retrain(promote, progress=gr.Progress()):
        if not pipeline.demo_db().exists():
            return "Run **1. Simulate** first."
        progress(0.1, desc="training challenger (about 1 min)")
        r = pipeline.run_retrain(bool(promote))
        return "\n\n".join(narrate.explain_retrain(r))

    def do_reset():
        return f"Champion alias reset to baseline version {pipeline.reset_champion()}."

    with gr.Blocks(title="Fraud model monitoring") as demo:
        gr.Markdown(INTRO)
        with gr.Row():
            scenario = gr.Radio(["A", "B", "C", "none"], value="B", label="Scenario")
            start_day = gr.Number(value=config.load()["scenario"]["start_day"], precision=0, label="Start day (1-10)")
            expect = gr.Textbox(SCENARIO_HELP["B"], label="What to expect", interactive=False)
        gr.Examples([["B", 6, SCENARIO_HELP["B"]], ["A", 6, SCENARIO_HELP["A"]], ["C", 6, SCENARIO_HELP["C"]]],
                    inputs=[scenario, start_day, expect], label="Ready samples (click one to fill the inputs)")
        scenario.change(lambda s: SCENARIO_HELP[s], scenario, expect)
        with gr.Row():
            b1 = gr.Button("1. Simulate traffic", variant="primary")
            b2 = gr.Button("2. Monitor windows", variant="primary")
            promote = gr.Checkbox(False, label="Promote winner (moves the champion alias)")
            b3 = gr.Button("3. Retrain", variant="primary")
            b4 = gr.Button("Reset champion to baseline")
        status = gr.Markdown("Status: idle. Start with **1. Simulate traffic**.")
        with gr.Row():
            runs = gr.Dropdown(list(available_runs()), value=next(iter(available_runs()), None), label="Show results from")
            refresh = gr.Button("Reload results")
        gr.Markdown(PLOT_HELP)
        plot, tbl, html = gr.Plot(), gr.Dataframe(), gr.HTML()
        with gr.Accordion("Pros / cons / at scale", open=False):
            gr.Markdown(TRADEOFFS)
        b1.click(do_simulate, [scenario, start_day], status).then(choices, None, runs)
        b2.click(do_monitor, None, [status, runs]).then(show, runs, [plot, tbl, html])
        b3.click(do_retrain, promote, status)
        b4.click(do_reset, None, status)
        refresh.click(lambda s: (choices(s), *show(s)), runs, [runs, plot, tbl, html])
        runs.change(show, runs, [plot, tbl, html])
        demo.load(lambda s: (choices(s), *show(s)), runs, [runs, plot, tbl, html])
    return demo
