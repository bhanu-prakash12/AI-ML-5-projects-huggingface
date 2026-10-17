#!/usr/bin/env python3
"""Cross-platform task runner (stdlib only): `python run.py <command>`; mirrors the Makefile.

Works on macOS, Linux and Windows (PowerShell / VS Code terminal). No bash, `kill`, `rm` or `time`.
"""
from __future__ import annotations

import argparse
import os
import shutil
import signal
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / ".run"
PORTS = {"API": 8000, "MLflow UI": 5001, "Gradio UI": 7861}
COMMANDS = ["setup", "data", "train", "eval", "serve", "ui", "mlflow-ui", "test", "lint",
            "smoke", "verify", "demo", "stop", "clean", "package"]


def venv_python(root: Path = ROOT, os_name: str | None = None) -> str:
    """Interpreter inside .venv (Scripts/python.exe on Windows, bin/python elsewhere).

    Falls back to the interpreter running this script when no venv exists.
    """
    win = (os_name or os.name) == "nt"
    cand = root / ".venv" / ("Scripts" if win else "bin") / ("python.exe" if win else "python")
    return str(cand) if cand.exists() else sys.executable


def task_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def run(args: list[str], check: bool = True) -> int:
    print("+", " ".join(args), flush=True)
    rc = subprocess.run(args, cwd=ROOT, env=task_env()).returncode
    if check and rc != 0:
        raise SystemExit(rc)
    return rc


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) != 0


def kill_pid(pid: int, os_name: str | None = None) -> None:
    """Terminate a process on any OS (taskkill on Windows, SIGTERM elsewhere)."""
    try:
        if (os_name or os.name) == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        else:
            os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def stop(run_dir: Path = RUN_DIR) -> int:
    n = 0
    for f in run_dir.glob("*.pid"):
        try:
            kill_pid(int(f.read_text().strip()))
            n += 1
        except ValueError:
            pass
        f.unlink(missing_ok=True)
    print(f"stopped {n} process(es) recorded in .run/*.pid")
    return 0


def clean() -> int:
    for d in (".run", "build", "mlruns"):
        shutil.rmtree(ROOT / d, ignore_errors=True)
    (ROOT / "mlflow.db").unlink(missing_ok=True)
    for d in ("data", "artifacts"):
        for p in (ROOT / d).glob("*"):
            if p.name != ".gitkeep":
                shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
    for f in ("metrics.json", "eda.md", "fairness.md", "latency.md", "resources.md",
              "train_time.txt"):
        (ROOT / "reports" / f).unlink(missing_ok=True)
    print("cleaned")
    return 0


def setup() -> int:
    """Create .venv and install dev requirements (uv if present, else venv + pip)."""
    if shutil.which("uv"):
        run(["uv", "venv", "--python", "3.11", ".venv"])
        run(["uv", "pip", "install", "--python", venv_python(), "-r", "requirements-dev.txt"])
    else:
        run([sys.executable, "-m", "venv", ".venv"])
        py = venv_python()
        run([py, "-m", "pip", "install", "-r", "requirements-dev.txt"])
    return 0


def demo() -> int:
    """Pre-demo checklist, then the narrated read-only steps; prints what to open next."""
    py = venv_python()
    print("== Pre-demo checklist ==")
    model = ROOT / "artifacts" / "model" / "model.txt"
    print(("OK   " if model.exists() else "FAIL ") + f"trained model at {model.relative_to(ROOT)}"
          + ("" if model.exists() else "  -> run: python run.py data ; python run.py train"))
    for name, port in PORTS.items():
        print(("OK   " if port_free(port) else "WARN ") + f"{name} port {port} "
              + ("free" if port_free(port) else "in use (python run.py stop, or free it)"))
    if not model.exists():
        return 1
    run([py, "scripts/evaluate.py"])
    run([py, "scripts/smoke.py"])
    print("\nNext (separate terminals):\n"
          "  python run.py serve       -> http://127.0.0.1:8000/docs\n"
          "  python run.py ui          -> http://127.0.0.1:7861\n"
          "  python run.py mlflow-ui   -> http://127.0.0.1:5001")
    return 0


def dispatch(cmd: str) -> int:
    py = venv_python()
    table = {
        "setup": setup,
        "data": lambda: run([py, "scripts/make_data.py"]) or run([py, "scripts/eda.py"]),
        "train": lambda: run([py, "scripts/train.py"]),
        "eval": lambda: run([py, "scripts/evaluate.py"]),
        "serve": lambda: run([py, "-m", "uvicorn", "dropout_risk.api:app", "--port", "8000"],
                             check=False),
        "ui": lambda: run([py, "scripts/ui.py"], check=False),
        "mlflow-ui": lambda: run([py, "-m", "mlflow", "ui", "--backend-store-uri",
                                  "sqlite:///mlflow.db", "--port", "5001"], check=False),
        "test": lambda: run([py, "-m", "pytest", "-q"]),
        "lint": lambda: run([py, "-m", "ruff", "check", "src", "tests", "scripts", "run.py"]),
        "smoke": lambda: run([py, "scripts/smoke.py"]),
        "demo": demo,
        "stop": stop,
        "clean": clean,
        "package": lambda: run([py, "scripts/build_lambda.py"]),
    }
    if cmd == "verify":
        for c in ("lint", "test", "smoke"):
            print(f"\n=== {c} ===")
            dispatch(c)
        print("\nverify OK")
        return 0
    return int(table[cmd]() or 0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Cross-platform runner for p1 (dropout risk).",
        epilog="Typical: python run.py setup ; data ; train ; eval ; serve | ui | mlflow-ui. "
               "`demo` prints a pre-demo checklist. `verify` = lint + test + smoke.")
    ap.add_argument("command", choices=COMMANDS)
    return dispatch(ap.parse_args(argv).command)


if __name__ == "__main__":
    raise SystemExit(main())
