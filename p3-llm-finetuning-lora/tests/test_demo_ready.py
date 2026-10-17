"""Console fallback, cross-platform runner logic (monkeypatched, no Windows needed), sqlite URI, API examples."""
import builtins
import importlib
import subprocess
import sys
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run  # noqa: E402
from med_assistant import api, console  # noqa: E402
from med_assistant.config import sqlite_uri  # noqa: E402


def test_console_plain_fallback(monkeypatch, capsys):
    monkeypatch.setattr(console, "_CON", None)
    console.banner("T", "what", "expect")
    console.step(1, 2, "msg")
    console.ok("fine")
    console.table([["a", 1]], ["k", "v"])
    console.kv_table({"x": 2})
    console.note("p", "c", "s")
    out = capsys.readouterr().out
    assert "== T ==" in out and "[1/2] msg" in out and "Pros:" in out and "a | 1" in out


def test_console_import_without_rich(monkeypatch):
    real = builtins.__import__

    def fake(name, *a, **k):
        if name.startswith("rich"):
            raise ImportError("no rich")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    try:
        m = importlib.reload(console)
        assert m._CON is None
    finally:
        monkeypatch.undo()
        importlib.reload(console)


def test_venv_python_per_os(tmp_path):
    assert run.venv_python(tmp_path, "nt") == tmp_path / ".venv" / "Scripts/python.exe"
    assert run.venv_python(tmp_path, "posix") == tmp_path / ".venv" / "bin/python"
    assert run.python_exe(tmp_path, "nt") == sys.executable  # venv absent -> current interpreter


def test_commands_have_no_shell_tools(tmp_path):
    for cmd in ("data", "baseline", "train", "eval", "serve", "ui", "mlflow-ui", "test", "lint", "smoke", "demo", "verify"):
        for argv in run.commands(cmd, [], tmp_path, "nt"):
            assert argv[0] not in ("bash", "sh", "make", "rm", "kill", "curl")
    v = run.commands("verify", [], tmp_path, "nt")
    assert len(v) == 3 and v[2][1].endswith("smoke.py")
    assert run.commands("train", ["--yes"], tmp_path, "nt")[0][-1] == "--yes"


def test_build_env_offline_only_for_serving(tmp_path):
    e = run.build_env("serve", {}, tmp_path)
    assert e["HF_HUB_OFFLINE"] == "1" and e["PYTHONPATH"].startswith(str(tmp_path / "src"))
    assert "HF_HUB_OFFLINE" not in run.build_env("data", {}, tmp_path)
    assert run.build_env("serve", {"HF_HUB_OFFLINE": "0"}, tmp_path)["HF_HUB_OFFLINE"] == "0"


def test_kill_pid_windows_uses_taskkill(monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append(a[0]))
    run.kill_pid(123, "nt")
    assert calls == [["taskkill", "/PID", "123", "/F", "/T"]]


def test_sqlite_uri_posix_and_windows_shape():
    u = sqlite_uri(ROOT / "mlflow.db")
    assert u.startswith("sqlite:///") and u.endswith("/mlflow.db") and "\\" not in u
    assert run.sqlite_uri(ROOT / "mlflow.db") == u
    # Windows drive paths must become sqlite:///C:/... (3 slashes + drive), never backslashes
    win = PureWindowsPath(r"C:\Users\me\p3\mlflow.db")
    assert "sqlite:///" + win.as_posix() == "sqlite:///C:/Users/me/p3/mlflow.db"


def test_device_prefers_cuda(monkeypatch):
    import torch

    from med_assistant.device import get_device, inference_dtype
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert get_device() == "cuda" and inference_dtype("cuda") == torch.float16
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert inference_dtype("cpu") == torch.float32


def test_openapi_has_examples_and_description():
    spec = api.app.openapi()
    assert "mode=base" in spec["info"]["description"]
    req = spec["components"]["schemas"]["GenReq"]
    assert len(req["examples"]) == 3
    for ex in req["examples"]:
        api.GenReq(**ex)  # examples validate against the schema
    for ex in spec["components"]["schemas"]["GenResp"]["examples"]:
        api.GenResp(**ex)
    assert spec["paths"]["/generate"]["post"]["summary"]
