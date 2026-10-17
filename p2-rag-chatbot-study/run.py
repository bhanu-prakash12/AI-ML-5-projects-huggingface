#!/usr/bin/env python3
"""Cross-platform runner (stdlib only). Mirrors the Makefile targets on macOS, Linux and Windows.

Usage: python run.py <command> [args]
  setup    create .venv and install pinned deps          data     fetch the corpus (Wikipedia + exam PDFs)
  models   download pinned models                        ingest   chunk the corpus (reports/ingest.md)
  index    ingest + embed into Chroma + probe queries    eval     run the RAG eval (reports/eval.md)
  serve    start the API (port from config/P2_API_PORT)  ui       start the Gradio chat (needs serve)
  demo     narrated end-to-end Q&A (in + out of corpus)  smoke    start API, ask 2 questions, stop
  test     pytest      lint  ruff      verify  lint + test + smoke
  stop     stop servers started from .run/*.pid          clean    remove .run, chroma, corpus, report json
  train / mlflow-ui   print "not used in this project"
Sample: python run.py index && python run.py serve
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN = ROOT / ".run"


def venv_python(root: Path = ROOT, os_name: str | None = None) -> str:
    """Interpreter inside .venv (Scripts/python.exe on Windows, bin/python elsewhere); else the current one."""
    name = os.name if os_name is None else os_name
    cand = root / ".venv" / ("Scripts/python.exe" if name == "nt" else "bin/python")
    return str(cand) if cand.exists() else sys.executable


def child_env(offline: bool = True) -> dict[str, str]:
    env = dict(os.environ)
    env.update(PYTHONPATH=str(ROOT / "src"), ANONYMIZED_TELEMETRY="False", PYTORCH_ENABLE_MPS_FALLBACK="1",
               TOKENIZERS_PARALLELISM="false")
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
    return env


def sh(args: list[str], offline: bool = True, check: bool = True) -> int:
    print("$ " + " ".join(args), flush=True)
    rc = subprocess.run(args, cwd=ROOT, env=child_env(offline)).returncode
    if check and rc:
        sys.exit(rc)
    return rc


def script(name: str, *extra: str, offline: bool = True) -> int:
    return sh([venv_python(), str(ROOT / "scripts" / name), *extra], offline=offline)


def kill_pid(pid: int, os_name: str | None = None) -> None:
    if (os.name if os_name is None else os_name) == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def cmd_stop() -> int:
    for f in RUN.glob("*.pid") if RUN.exists() else []:
        try:
            kill_pid(int(f.read_text().strip()))
        except ValueError:
            pass
        f.unlink(missing_ok=True)
        print(f"stopped {f.name}")
    return 0


def cmd_setup() -> int:
    if shutil.which("uv"):
        sh(["uv", "venv", "--python", "3.11", ".venv"])
        sh(["uv", "pip", "install", "--python", venv_python(), "-r", "requirements-dev.txt"])
    else:  # Windows without uv: py -3.11 -m venv .venv (or any python 3.11)
        sh([sys.executable, "-m", "venv", ".venv"])
        sh([venv_python(), "-m", "pip", "install", "-r", "requirements-dev.txt"])
    return 0


def cmd_clean() -> int:
    for p in (RUN, ROOT / "chroma", ROOT / "data" / "corpus"):
        shutil.rmtree(p, ignore_errors=True)
    for f in (ROOT / "reports").glob("*.json"):
        f.unlink()
    print("cleaned .run, chroma, data/corpus, reports/*.json")
    return 0


def cmd_verify() -> int:
    for c in ("lint", "test", "smoke"):
        print(f"\n== verify: {c} ==", flush=True)
        rc = COMMANDS[c]()
        if rc:
            return rc
    print("\nverify OK")
    return 0


def not_used() -> int:
    print("not used in this project")
    return 0


COMMANDS = {
    "setup": cmd_setup,
    "data": lambda: script("fetch_corpus.py", offline=False),
    "models": lambda: script("download_models.py", offline=False),
    "ingest": lambda: script("ingest.py"),
    "index": lambda: script("ingest.py") or script("probe.py"),
    "train": not_used,
    "eval": lambda: script("run_eval.py"),
    "serve": lambda: sh([venv_python(), "-m", "uvicorn", "study_assistant.api:app", "--port", str(_port()), "--workers", "1"]),
    "ui": lambda: script("ui.py"),
    "mlflow-ui": not_used,
    "demo": lambda: script("rag_demo.py"),
    "test": lambda: sh([venv_python(), "-m", "pytest", "-q"]),
    "lint": lambda: sh([venv_python(), "-m", "ruff", "check", "src", "tests", "scripts", "run.py"]),
    "smoke": lambda: script("smoke.py"),
    "verify": cmd_verify,
    "stop": cmd_stop,
    "clean": cmd_clean,
}


def _port() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from study_assistant.config import api_port

    return api_port()


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in COMMANDS:
        print(__doc__)
        return 0 if not argv or argv[0] in ("-h", "--help") else 2
    return COMMANDS[argv[0]]() or 0


if __name__ == "__main__":
    sys.exit(main())
