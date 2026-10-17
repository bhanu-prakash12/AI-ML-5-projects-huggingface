#!/usr/bin/env python3
"""Cross-platform task runner (stdlib only) mirroring the Makefile: macOS, Linux, Windows.

Usage:  python run.py <command> [args]
Commands: setup models data train eval serve ui mlflow-ui test lint smoke verify demo stop clean
Examples:
  python run.py setup          # create .venv and install pinned deps
  python run.py demo           # narrated CLIP + SmolVLM run on cached sample images
  python run.py serve          # API on :8005 (Swagger at /docs)
  python run.py ui             # Gradio on :7865 (needs `serve` running)
  python run.py verify         # lint + test + smoke
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
API_PORT, UI_PORT = 8005, 7865
OFFLINE_CMDS = {"serve", "demo", "smoke"}  # models must already be cached for these


def venv_python(root: Path = ROOT, os_name: str | None = None) -> str:
    """Venv interpreter path (Scripts/python.exe on Windows, bin/python elsewhere); falls back to sys.executable."""
    os_name = os_name or os.name
    rel = Path(".venv") / ("Scripts" if os_name == "nt" else "bin") / ("python.exe" if os_name == "nt" else "python")
    p = root / rel
    return str(p) if p.exists() else sys.executable


def kill_cmd(pid: int, os_name: str | None = None) -> list[str]:
    """Command (or marker) used to stop a PID; Windows uses taskkill, POSIX uses `kill`."""
    return ["taskkill", "/PID", str(pid), "/T", "/F"] if (os_name or os.name) == "nt" else ["kill", str(pid)]


def prepare_env(cmd: str) -> None:
    """Set process env in-Python (no inline VAR=1 cmd, which does not work in PowerShell)."""
    os.environ["PYTHONPATH"] = str(ROOT / "src")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("PYTHONUTF8", "1")  # Windows consoles default to cp1252
    if cmd in OFFLINE_CMDS:
        os.environ["HF_HUB_OFFLINE"] = "1"


def sh(args: list[str], check: bool = True) -> int:
    print("+", " ".join(args), flush=True)
    rc = subprocess.call(args, cwd=ROOT)
    if check and rc:
        sys.exit(rc)
    return rc


def py(*args: str, check: bool = True) -> int:
    return sh([venv_python(), *args], check)


def cmd_setup(_: list[str]) -> None:
    vdir = ROOT / ".venv"
    if shutil.which("uv"):
        sh(["uv", "venv", "--python", "3.11", str(vdir)])
        sh(["uv", "pip", "install", "--python", venv_python(), "-r", "requirements-dev.txt"])
    else:  # plain venv + pip (Windows without uv: use `py -3.11 run.py setup`)
        sh([sys.executable, "-m", "venv", str(vdir)])
        sh([venv_python(), "-m", "pip", "install", "-r", "requirements-dev.txt"])


def cmd_serve(a: list[str]) -> None:
    py("-m", "uvicorn", "food_lens.api:app", "--port", str(API_PORT), "--workers", "1", *a)


def cmd_stop(_: list[str]) -> None:
    run_dir = ROOT / ".run"
    for f in sorted(run_dir.glob("*.pid")):
        try:
            pid = int(f.read_text().strip())
            subprocess.call(kill_cmd(pid), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"stopped pid {pid} ({f.name})")
        except (ValueError, OSError):
            pass
        f.unlink(missing_ok=True)


def cmd_clean(_: list[str]) -> None:
    shutil.rmtree(ROOT / ".run", ignore_errors=True)
    for d in ("artifacts", "reports"):
        for f in (ROOT / d).glob("*" if d == "artifacts" else "*.json"):
            if f.name != ".gitkeep":
                shutil.rmtree(f, ignore_errors=True) if f.is_dir() else f.unlink(missing_ok=True)


def cmd_verify(a: list[str]) -> None:
    for c in ("lint", "test", "smoke"):
        print(f"\n== verify: {c} ==", flush=True)
        prepare_env(c)
        COMMANDS[c](a)
    print("\nverify ok")


COMMANDS = {
    "setup": cmd_setup,
    "models": lambda a: py("scripts/download_models.py", *a),
    "data": lambda a: py("scripts/build_nutrition.py", *a),
    "train": lambda a: print("train: not used in this project (zero-shot CLIP, no training)"),
    "eval": lambda a: py("scripts/eval_clip.py", *a),
    "serve": cmd_serve,
    "ui": lambda a: py("scripts/ui.py", *a),
    "mlflow-ui": lambda a: print("mlflow-ui: not used in this project"),
    "test": lambda a: py("-m", "pytest", "-q", *a),
    "lint": lambda a: py("-m", "ruff", "check", "src", "tests", "scripts", "run.py", *a),
    "smoke": lambda a: py("scripts/smoke.py", *a),
    "verify": cmd_verify,
    "demo": lambda a: py("scripts/demo.py", *a),
    "stop": cmd_stop,
    "clean": cmd_clean,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=sorted(COMMANDS))
    ap.add_argument("args", nargs=argparse.REMAINDER, help="extra args passed to the underlying script")
    ns = ap.parse_args(argv)
    prepare_env(ns.command)
    COMMANDS[ns.command](ns.args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
