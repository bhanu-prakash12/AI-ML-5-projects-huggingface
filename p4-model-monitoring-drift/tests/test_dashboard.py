import pandas as pd

from fraud_monitor import dashboard, retrain


def _df(alert2=True):
    return pd.DataFrame({
        "window": [1, 2, 3], "n": [10] * 3, "n_fraud": [5, 6, 7], "drift_share": [0.05, 0.1, 0.4],
        "psi_score": [0.01, 0.02, 0.3], "pr_auc": [0.8, 0.78, None], "baseline_pr_auc": [0.8] * 3,
        "alert": [False, False, alert2], "reasons": [[], [], ["drift_share 0.40 > 0.3"]],
        "perf_skipped": [None, None, "labels pending"], "report": [f"windows/window_0{i}.html" for i in (1, 2, 3)],
    })


def test_figure_table_links():
    d = _df()
    assert len(dashboard.figure(d).axes) == 2
    t = dashboard.table(d)
    assert t.loc[2, "why"].startswith("drift_share") and "ALERT" in dashboard.links(d)
    assert dashboard.figure(pd.DataFrame()) is not None


def test_retrain_decide():
    assert retrain.decide(0.5, 0.7, 0.8, 0.79)[0]
    assert not retrain.decide(0.5, 0.51, 0.8, 0.8)[0]          # not enough gain
    assert not retrain.decide(0.5, 0.9, 0.8, 0.5)[0]           # forgot the old pattern


def test_available_runs_and_build(tmp_path, monkeypatch):
    from fraud_monitor import config

    (tmp_path / "demo").mkdir()
    (tmp_path / "demo" / "status.json").write_text("[]")
    (tmp_path / "scenB").mkdir()
    (tmp_path / "scenB" / "status.json").write_text("[]")
    monkeypatch.setattr(config, "path", lambda k: tmp_path)
    runs = dashboard.available_runs()
    assert list(runs)[0] == dashboard.LATEST_DEMO and "Saved run: reports/scenB" in runs
    assert dashboard.build() is not None
