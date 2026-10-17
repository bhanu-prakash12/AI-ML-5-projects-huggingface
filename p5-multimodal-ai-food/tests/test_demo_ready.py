import importlib.util
import os
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from food_lens import console, device
from food_lens.schema import EXAMPLE_RESPONSE, AnalyzeResponse

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_mod", ROOT / "run.py")
run_mod = importlib.util.module_from_spec(spec)
sys.modules["run_mod"] = run_mod
spec.loader.exec_module(run_mod)


def test_venv_python_windows_layout(tmp_path):
    exe = tmp_path / ".venv" / "Scripts" / "python.exe"
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    assert run_mod.venv_python(tmp_path, "nt") == str(exe)


def test_venv_python_posix_layout_and_fallback(tmp_path):
    exe = tmp_path / ".venv" / "bin" / "python"
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    assert run_mod.venv_python(tmp_path, "posix") == str(exe)
    assert run_mod.venv_python(tmp_path / "none", "nt") == sys.executable  # no venv yet


def test_kill_cmd_per_os():
    assert run_mod.kill_cmd(42, "nt")[0] == "taskkill" and "/T" in run_mod.kill_cmd(42, "nt")
    assert run_mod.kill_cmd(42, "posix") == ["kill", "42"]


def test_prepare_env_sets_offline_only_for_serving(monkeypatch):
    monkeypatch.setattr(os, "environ", dict(os.environ))
    os.environ.pop("HF_HUB_OFFLINE", None)
    run_mod.prepare_env("eval")
    assert "HF_HUB_OFFLINE" not in os.environ
    run_mod.prepare_env("serve")
    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_every_makefile_command_is_a_run_command():
    for c in "setup models data train eval serve ui mlflow-ui test lint smoke verify demo stop clean".split():
        assert c in run_mod.COMMANDS


def test_console_plain_fallback(monkeypatch, capsys):
    monkeypatch.setenv("NO_COLOR", "1")
    console.banner("T", "what", "expect")
    console.step(1, 2, "go")
    console.ok("fine")
    console.table([["a", 1]], ["k", "v"])
    console.note("p", "c", "s")
    out = capsys.readouterr().out
    assert "=== T ===" in out and "[1/2] go" in out and "OK   fine" in out and "At scale: s" in out


def test_console_without_rich(monkeypatch, capsys):
    monkeypatch.setattr(console, "_RICH", False)
    console.kv_table({"x": 1})
    assert "x | 1" in capsys.readouterr().out


def test_get_device_prefers_cuda_then_mps(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert device.get_device() == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert device.get_device() == "mps"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert device.get_device() == "cpu"
    assert device.inference_dtype("cpu") == torch.float32 and device.inference_dtype("cuda") == torch.float16


def test_empty_cache_is_safe_on_cpu():
    device.empty_cache("cpu")


def test_api_example_validates_and_openapi_has_it():
    AnalyzeResponse.model_validate(EXAMPLE_RESPONSE)
    with pytest.raises(ValidationError):
        AnalyzeResponse.model_validate({"top3": []})
    from food_lens.api import app

    spec_ = app.openapi()
    assert "AnalyzeResponse" in spec_["components"]["schemas"]
    ref = spec_["paths"]["/analyze"]["post"]["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
    assert ref.endswith("AnalyzeResponse")


def test_sample_assets_small():
    imgs = sorted((ROOT / "assets").glob("*.jpg"))
    assert len(imgs) == 3 and all(p.stat().st_size <= 100_000 for p in imgs)
