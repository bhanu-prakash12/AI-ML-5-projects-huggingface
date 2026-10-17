"""Tests for the demo-readiness pieces: console fallback, run.py paths, UI logic, API examples."""
import importlib.util
import sys
from pathlib import Path

from dropout_risk import console, features
from dropout_risk.data import RANGES
from dropout_risk.samples import SAMPLES, UI_FIELDS
from dropout_risk.ui_core import build_row, example_rows, summary_md

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_mod", ROOT / "run.py")
run_mod = importlib.util.module_from_spec(spec)
sys.modules["run_mod"] = run_mod
spec.loader.exec_module(run_mod)


def test_console_plain_fallback(monkeypatch, capsys):
    monkeypatch.setattr(console, "_RICH", False)
    console.banner("T", "what", "expect")
    console.step(1, 2, "go")
    console.ok("fine")
    console.table([(1, 2)], headers=("a", "b"))
    console.kv_table({"k": "v"})
    console.note("p", "c", "s")
    out = capsys.readouterr().out
    assert "== T ==" in out and "[1/2] go" in out and "OK   fine" in out and "a | b" in out
    assert "Pros:" in out and "At scale:" in out


def test_console_rich_path_runs(monkeypatch, capsys):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(console, "_console", None)
    console.banner("T", "what", "expect")
    console.fail("bad")
    assert "FAIL bad" in capsys.readouterr().out


def test_venv_python_windows_vs_posix(tmp_path):
    (tmp_path / ".venv" / "Scripts").mkdir(parents=True)
    (tmp_path / ".venv" / "Scripts" / "python.exe").write_text("")
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / ".venv" / "bin" / "python").write_text("")
    assert run_mod.venv_python(tmp_path, "nt").endswith("python.exe")
    assert run_mod.venv_python(tmp_path, "posix").endswith("bin/python")
    assert run_mod.venv_python(tmp_path / "none", "nt") == sys.executable


def test_kill_pid_uses_taskkill_on_windows(monkeypatch):
    calls = []
    monkeypatch.setattr(run_mod.subprocess, "run", lambda *a, **k: calls.append(a[0]))
    run_mod.kill_pid(1234, "nt")
    assert calls == [["taskkill", "/PID", "1234", "/T", "/F"]]


def test_stop_removes_pid_files(tmp_path, monkeypatch):
    killed = []
    monkeypatch.setattr(run_mod, "kill_pid", lambda pid, os_name=None: killed.append(pid))
    (tmp_path / "api.pid").write_text("42")
    (tmp_path / "bad.pid").write_text("x")
    run_mod.stop(tmp_path)
    assert killed == [42] and not list(tmp_path.glob("*.pid"))


def test_commands_cover_makefile_targets():
    for c in ("setup", "data", "train", "eval", "serve", "ui", "mlflow-ui", "test", "lint",
              "smoke", "verify", "demo", "stop", "clean"):
        assert c in run_mod.COMMANDS


def test_samples_in_range_and_complete():
    for name, row in SAMPLES.items():
        assert set(row) == set(features.FEATURES), name
        for c, v in row.items():
            lo, hi = RANGES[c]
            assert lo <= v <= hi, (name, c, v)


def test_ui_rows_and_summary():
    ex = example_rows()
    assert len(ex) == 3 and all(len(r) == len(UI_FIELDS) for r in ex)
    row = build_row(ex[2], features.FEATURES)
    assert set(row) == set(features.FEATURES)
    pred = {"risk_tier": "high", "probability": 0.9, "model_version": "v"}
    assert "HIGH risk" in summary_md(pred, 0.23)
