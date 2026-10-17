from pathlib import Path

import pytest

import run
from fraud_monitor import model, procs


def test_venv_python_paths():
    root = Path("/x")
    assert procs.venv_python(root, windows=True).parts[-3:] == (".venv", "Scripts", "python.exe")
    assert procs.venv_python(root, windows=False).parts[-3:] == (".venv", "bin", "python")


def test_windows_flag_is_used_when_not_explicit(monkeypatch):
    monkeypatch.setattr(procs, "is_windows", lambda: True)
    assert str(procs.venv_python(Path("/x"))).endswith("python.exe")
    assert procs.kill_cmd(42) == ["taskkill", "/PID", "42", "/T", "/F"]
    monkeypatch.setattr(procs, "is_windows", lambda: False)
    assert procs.kill_cmd(42) is None


def test_resolve_python_falls_back_to_current(tmp_path):
    assert procs.resolve_python(tmp_path) == __import__("sys").executable


def test_build_commands_use_given_python_and_no_shell_tools():
    for name in ("data", "train", "simulate", "monitor", "retrain", "ui", "smoke", "serve", "test", "lint"):
        cmd = run.build_command(name, "PY", [])
        assert cmd and cmd[0] == "PY"
        assert not any(tok in ("bash", "kill", "rm", "curl") for tok in cmd)
    assert "--workers" in run.build_command("serve", "PY", [])
    assert run.build_command("simulate", "PY", ["--reset"])[-1] == "--reset"


def test_ports_from_config():
    assert run.ports() == {"api": 8004, "ui": 7864, "mlflow": 5004}


def test_sqlite_uri_windows_style():
    assert model.sqlite_uri(Path("artifacts/mlflow.db")) == "sqlite:///artifacts/mlflow.db"
    assert model.sqlite_uri(__import__("pathlib").PureWindowsPath("C:\\a\\b.db")) == "sqlite:///C:/a/b.db"


def test_stop_pid_files_ignores_garbage(tmp_path, monkeypatch):
    killed = []
    monkeypatch.setattr(procs, "terminate", killed.append)
    (tmp_path / "a.pid").write_text("123")
    (tmp_path / "b.pid").write_text("junk")
    assert procs.stop_pid_files(tmp_path) == [123] and killed == [123] and not list(tmp_path.glob("*.pid"))


def test_demo_args_defaults():
    a = run.demo_args([])
    assert (a.scenario, a.start_day, a.promote, a.keep_champion) == ("B", 6, False, False)
    with pytest.raises(SystemExit):
        run.demo_args(["--bogus"])
