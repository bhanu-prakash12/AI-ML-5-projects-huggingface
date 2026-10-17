import importlib
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run  # noqa: E402

from study_assistant import console, device  # noqa: E402


def test_venv_python_windows_and_posix(tmp_path):
    (tmp_path / ".venv" / "Scripts").mkdir(parents=True)
    (tmp_path / ".venv" / "Scripts" / "python.exe").write_text("")
    assert run.venv_python(tmp_path, "nt").endswith("Scripts/python.exe")
    assert run.venv_python(tmp_path, "posix") == sys.executable  # no bin/python -> fallback
    (tmp_path / ".venv" / "bin").mkdir()
    (tmp_path / ".venv" / "bin" / "python").write_text("")
    assert run.venv_python(tmp_path, "posix").endswith("bin/python")


def test_kill_pid_windows_uses_taskkill(monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda args, **k: calls.append(args))
    run.kill_pid(123, "nt")
    assert calls[0][:2] == ["taskkill", "/PID"]


def test_unknown_command_and_not_used(capsys):
    assert run.main(["bogus"]) == 2
    assert run.main(["train"]) == 0
    assert "not used" in capsys.readouterr().out


def test_env_sets_offline():
    assert run.child_env()["HF_HUB_OFFLINE"] == "1"


def test_console_plain_fallback(monkeypatch, capsys):
    monkeypatch.setattr(console, "_RICH", False)
    console.banner("T", "what", "expect")
    console.step(1, 2, "go")
    console.table([[1, 2]], ["a", "b"])
    console.note("p", "c", "s")
    out = capsys.readouterr().out
    assert "=== T ===" in out and "[1/2] go" in out and "At scale: s" in out


def test_console_rich_renders(capsys):
    console.ok("fine")
    console.kv_table({"k": "v"})
    assert "fine" in capsys.readouterr().out


def test_device_order(monkeypatch):
    torch = importlib.import_module("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert device.get_device() == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert device.get_device() == "mps"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert device.get_device() == "cpu"
    assert device.dtype_name("cuda") == "float16" and device.dtype_name("cpu") == "float32"


def test_api_port_env(monkeypatch):
    from study_assistant.config import api_port

    monkeypatch.setenv("P2_API_PORT", "8002")
    assert api_port() == 8002
    monkeypatch.delenv("P2_API_PORT")
    assert api_port() == 8000
