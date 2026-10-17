#!/usr/bin/env python3
"""Cross-platform task runner for P3 (stdlib only). Mirrors the Makefile on macOS, Linux and Windows.

usage: python run.py <command> [extra args]
  setup      create .venv and install requirements-dev.txt (uses uv if available, else venv + pip)
  data       build train/val/test splits            baseline  base-model metrics
  train      LoRA fine-tune (adapter exists already) eval      base vs tuned comparison
  merge      save merged base+adapter copy          serve     API on :8003 (foreground)
  ui         Gradio side-by-side on :7863           mlflow-ui MLflow UI on :5003
  test       pytest                                 lint      ruff
  smoke      start API, check, stop                 verify    lint + test + smoke
  demo       narrated end-to-end API demo           stop      kill processes recorded in .run/*.pid
  clean      remove .run and caches (keeps reports/)
Extra args are passed to the underlying script, e.g. `python run.py train --yes`.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PORTS = {"api": 8003, "ui": 7863, "mlflow": 5003}
SERVING = {"serve", "ui", "smoke", "demo", "baseline", "eval", "merge"}  # run fully offline once models are cached


def venv_python(root: Path = ROOT, os_name: str | None = None) -> Path:
    """Interpreter inside .venv: Scripts/python.exe on Windows, bin/python elsewhere."""
    nt = (os_name or os.name) == "nt"
    return root / ".venv" / ("Scripts/python.exe" if nt else "bin/python")


def python_exe(root: Path = ROOT, os_name: str | None = None) -> str:
    vp = venv_python(root, os_name)
    return str(vp) if vp.exists() else sys.executable


def sqlite_uri(db: Path) -> str:
    """Absolute sqlite URI valid on Windows (sqlite:///C:/x/mlflow.db) and POSIX (sqlite:////x/mlflow.db)."""
    return "sqlite:///" + Path(db).resolve().as_posix()


def build_env(cmd: str, base: dict[str, str] | None = None, root: Path = ROOT) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    env["PYTHONPATH"] = str(root / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    env.setdefault("PYTHONUTF8", "1")  # Windows consoles/files default to cp1252 otherwise
    if cmd in SERVING:
        env.setdefault("HF_HUB_OFFLINE", "1")
    return env


def commands(cmd: str, extra: list[str], root: Path = ROOT, os_name: str | None = None) -> list[list[str]]:
    """Return the argv lists a command runs (pure function, unit-tested)."""
    py = python_exe(root, os_name)
    s = lambda name: [py, str(root / "scripts" / name), *extra]  # noqa: E731
    table: dict[str, list[list[str]]] = {
        "data": [s("make_data.py")], "baseline": [s("run_baseline.py")], "train": [s("train.py")],
        "eval": [s("run_compare.py")], "merge": [s("merge.py")], "ui": [s("ui.py")], "smoke": [s("smoke.py")],
        "demo": [[py, str(root / "scripts" / "smoke.py"), "--demo", *extra]],
        "serve": [[py, "-m", "uvicorn", "med_assistant.api:app", "--port", str(PORTS["api"]), "--workers", "1", *extra]],
        "mlflow-ui": [[py, "-m", "mlflow", "ui", "--backend-store-uri", sqlite_uri(root / "mlflow.db"),
                       "--port", str(PORTS["mlflow"]), *extra]],
        "test": [[py, "-m", "pytest", "-q", *extra]],
        "lint": [[py, "-m", "ruff", "check", "src", "tests", "scripts", "run.py", *extra]],
    }
    table["verify"] = table["lint"] + table["test"] + table["smoke"]
    return table[cmd]


def kill_pid(pid: int, os_name: str | None = None) -> None:
    if (os_name or os.name) == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/F", "/T"], capture_output=True, check=False)
    else:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def stop(root: Path = ROOT) -> int:
    n = 0
    for f in (root / ".run").glob("*.pid"):
        try:
            kill_pid(int(f.read_text().strip()))
            n += 1
        except (ValueError, OSError):
            pass
        f.unlink(missing_ok=True)
    print(f"stopped {n} process(es) from .run/*.pid")
    return 0


def clean(root: Path = ROOT) -> int:
    for d in (".run", ".pytest_cache", ".ruff_cache"):
        shutil.rmtree(root / d, ignore_errors=True)
    print("cleaned .run and caches (reports/ kept)")
    return 0


def setup(root: Path = ROOT) -> int:
    vp = venv_python(root)
    if not vp.exists():
        if shutil.which("uv"):
            subprocess.run(["uv", "venv", "--python", "3.11", str(root / ".venv")], check=True)
        elif os.name == "nt" and shutil.which("py"):
            subprocess.run(["py", "-3.11", "-m", "venv", str(root / ".venv")], check=True)
        else:
            subprocess.run([sys.executable, "-m", "venv", str(root / ".venv")], check=True)
    req = str(root / "requirements-dev.txt")
    if shutil.which("uv"):
        return subprocess.run(["uv", "pip", "install", "--python", str(vp), "-r", req], check=False).returncode
    return subprocess.run([str(vp), "-m", "pip", "install", "-r", req], check=False).returncode


def run_foreground(argv: list[str], env: dict[str, str], pidfile: str | None = None) -> int:
    """Run a command; record its PID in .run/ while it lives so `stop` can kill it."""
    p = subprocess.Popen(argv, cwd=ROOT, env=env)
    if pidfile:
        (ROOT / ".run").mkdir(exist_ok=True)
        (ROOT / ".run" / pidfile).write_text(str(p.pid))
    try:
        return p.wait()
    except KeyboardInterrupt:
        p.terminate()
        return p.wait()
    finally:
        if pidfile:
            (ROOT / ".run" / pidfile).unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    cmd, extra = args[0], args[1:]
    if cmd == "setup":
        return setup()
    if cmd == "stop":
        return stop()
    if cmd == "clean":
        return clean()
    if cmd not in {"data", "baseline", "train", "eval", "merge", "serve", "ui", "mlflow-ui", "test", "lint",
                   "smoke", "verify", "demo"}:
        print(f"unknown command: {cmd}\n{__doc__}")
        return 2
    env = build_env(cmd)
    pidfile = {"serve": "api.pid", "ui": "ui.pid", "mlflow-ui": "mlflow.pid"}.get(cmd)
    for argv_ in commands(cmd, extra):
        rc = run_foreground(argv_, env, pidfile)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
