"""Cross-platform task runner (macOS / Linux / Windows). Same targets as the Makefile.

usage: python run.py <command> [options]
commands: setup data train simulate monitor retrain eval serve ui mlflow-ui test lint smoke verify demo stop clean
Pass-through: extra options go to the underlying script, e.g.  python run.py simulate --reset --scenario B --start-day 6
Demo (narrated, ~3 min): python run.py demo [--scenario A|B|C|none] [--start-day 6] [--promote] [--keep-champion]
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from fraud_monitor import console, procs  # noqa: E402

RUN_DIR = ROOT / ".run"
DEFAULT_PORTS = {"api": 8004, "ui": 7864, "mlflow": 5004}
COMMANDS = ["setup", "data", "train", "simulate", "monitor", "retrain", "eval", "serve", "ui", "mlflow-ui", "test", "lint",
            "smoke", "verify", "demo", "stop", "clean"]


def ports() -> dict[str, int]:
    """Ports from configs/config.yaml (regex, so it works before PyYAML is installed)."""
    try:
        txt = (ROOT / "configs" / "config.yaml").read_text()
        m = re.search(r"^ports:\s*\{([^}]*)\}", txt, re.M)
        return {k: int(v) for k, v in re.findall(r"(\w+):\s*(\d+)", m.group(1))} if m else DEFAULT_PORTS
    except OSError:
        return DEFAULT_PORTS


def env() -> dict[str, str]:
    e = dict(os.environ)
    e["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + e.get("PYTHONPATH", "")
    e["PYTHONIOENCODING"] = "utf-8"
    e.setdefault("PYTHONUNBUFFERED", "1")
    return e


def sh(cmd: list[str], extra_env: dict[str, str] | None = None) -> int:
    e = {**env(), **(extra_env or {})}
    return subprocess.call(cmd, cwd=ROOT, env=e)


def script_cmd(py: str, name: str, extra: list[str]) -> list[str]:
    return [py, str(ROOT / "scripts" / name), *extra]


def build_command(cmd: str, py: str, extra: list[str]) -> list[str] | None:
    """Command line for the simple (one-process) targets; None for targets with custom logic."""
    p = ports()
    table: dict[str, list[str]] = {
        "data": script_cmd(py, "fetch_data.py", extra),
        "train": script_cmd(py, "train_baseline.py", extra),
        "simulate": script_cmd(py, "simulate.py", extra),
        "monitor": script_cmd(py, "monitor.py", extra),
        "retrain": script_cmd(py, "retrain.py", extra),
        "ui": script_cmd(py, "ui.py", extra),
        "smoke": script_cmd(py, "smoke.py", extra),
        "serve": [py, "-m", "uvicorn", "fraud_monitor.api:app", "--port", str(p["api"]), "--workers", "1", *extra],
        "test": [py, "-m", "pytest", "-q", *extra],
        "lint": [py, "-m", "ruff", "check", "src", "tests", "scripts", "run.py", *extra],
    }
    return table.get(cmd)


def mlflow_cmd(py: str) -> list[str]:
    from fraud_monitor.model import sqlite_uri

    return [py, "-m", "mlflow", "ui", "--backend-store-uri", sqlite_uri(ROOT / "artifacts" / "mlflow.db"),
            "--port", str(ports()["mlflow"])]


def cmd_setup() -> int:
    venv = ROOT / ".venv"
    if shutil.which("uv"):
        rc = sh(["uv", "venv", "--python", "3.11", str(venv)])
        py = str(procs.venv_python(ROOT))
        return rc or sh(["uv", "pip", "install", "--python", py, "-r", "requirements-dev.txt"])
    rc = sh([sys.executable, "-m", "venv", str(venv)])
    py = str(procs.venv_python(ROOT))
    return rc or sh([py, "-m", "pip", "install", "-r", "requirements-dev.txt"])


def cmd_stop() -> int:
    killed = procs.stop_pid_files(RUN_DIR)
    console.ok(f"stopped {len(killed)} process(es) from .run/*.pid" if killed else "nothing to stop (.run/*.pid is empty)")
    return 0


def cmd_clean() -> int:
    cmd_stop()
    for d in (RUN_DIR, ROOT / "artifacts" / "mlruns", ROOT / "reports" / "windows", ROOT / "reports" / "demo"):
        shutil.rmtree(d, ignore_errors=True)
    for pat, base in (("*.db", ROOT / "artifacts"), ("*.json", ROOT / "reports"), ("alerts.jsonl", ROOT / "reports")):
        for f in base.glob(pat):
            f.unlink(missing_ok=True)
    console.ok("cleaned .run, artifacts/*.db, artifacts/mlruns, reports/{windows,demo,*.json,alerts.jsonl}")
    return 0


def cmd_verify(py: str) -> int:
    for name in ("lint", "test", "smoke"):
        console.step(["lint", "test", "smoke"].index(name) + 1, 3, name)
        rc = sh(build_command(name, py, []) or [])
        if rc:
            console.fail(f"verify stopped at {name}")
            return rc
    console.ok("verify green: lint + test + smoke")
    return 0


def demo_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="run.py demo", description="Narrated end-to-end demo: serve -> simulate -> monitor -> retrain.",
                                 epilog="Writes reports/demo/ (status.json, windows/*.html, retrain.json); open it in the dashboard "
                                        "(python run.py ui -> 'Latest demo run').")
    ap.add_argument("--scenario", default="B", help="A, B, C or none (default B)")
    ap.add_argument("--start-day", type=int, default=6)
    ap.add_argument("--promote", action="store_true", help="actually move the champion alias if the challenger wins")
    ap.add_argument("--keep-champion", action="store_true", help="do not reset the champion to the baseline model first")
    ap.add_argument("--no-retrain", action="store_true")
    return ap.parse_args(argv)


def cmd_demo(py: str, argv: list[str]) -> int:
    a = demo_args(argv)
    total = 6
    port = ports()["api"]
    base = f"http://127.0.0.1:{port}"
    db, out = RUN_DIR / "demo.db", ROOT / "reports" / "demo"
    console.banner("end-to-end demo", f"Serve the champion model, replay 10 days of traffic with scenario {a.scenario} "
                   f"(from day {a.start_day}), monitor every window, then try a retrain.",
                   "alert around the injection day; retrain challenger beats the decayed champion (about 3 minutes)")
    console.step(1, total, "checking prerequisites")
    missing = [p for p in ("data/reference.parquet", "data/stream.parquet", "reports/baseline.json") if not (ROOT / p).exists()]
    if missing:
        console.fail(f"missing {missing}: run `python run.py data` then `python run.py train` first")
        return 1
    if procs.port_in_use(port):
        console.fail(f"port {port} is busy: run `python run.py stop` (Windows: netstat -ano | findstr :{port})")
        return 1
    console.ok("data, baseline report present; port free")
    RUN_DIR.mkdir(exist_ok=True)
    for f in RUN_DIR.glob("demo.db*"):
        f.unlink(missing_ok=True)
    shutil.rmtree(out, ignore_errors=True)
    if not a.keep_champion:
        console.step(2, total, "resetting champion alias to the baseline model (same starting point every run)")
        if sh(script_cmd(py, "reset_champion.py", [])):
            return 1
    else:
        console.step(2, total, "keeping the current champion")
    console.step(3, total, "starting the API (prediction log: .run/demo.db)")
    proc = procs.start_background([py, "-m", "uvicorn", "fraud_monitor.api:app", "--port", str(port), "--workers", "1"],
                                  {**env(), "P4_DB": str(db)}, ROOT, RUN_DIR / "api.log", RUN_DIR / "api.pid")
    try:
        if not procs.wait_health(base + "/health", 90, proc):
            console.fail("API did not become healthy")
            console.hint(procs.tail(RUN_DIR / "api.log"))
            return 1
        console.ok(f"API healthy at {base}  (Swagger: {base}/docs)")
        console.step(4, total, "simulating traffic")
        sim = ["--reset", "--scenario", a.scenario, "--start-day", str(a.start_day)]
        if sh(script_cmd(py, "simulate.py", sim), {"P4_DB": str(db)}):
            return 1
        console.step(5, total, "monitoring every window")
        if sh(script_cmd(py, "monitor.py", ["--db", str(db), "--out", str(out)])):
            return 1
        if a.no_retrain:
            console.step(6, total, "retrain skipped")
        else:
            console.step(6, total, "retraining (dry run unless --promote)")
            rt = ["--db", str(db), "--out", str(out)] + (["--reload-url", base] if a.promote else ["--no-promote"])
            if sh(script_cmd(py, "retrain.py", rt), {"P4_DB": str(db)}):
                return 1
    finally:
        procs.terminate(proc.pid)
        try:
            proc.wait(10)
        except Exception:  # noqa: BLE001
            time.sleep(1)
        (RUN_DIR / "api.pid").unlink(missing_ok=True)
    console.ok("demo finished: API stopped. Next: `python run.py ui` and pick 'Latest demo run'; `python run.py mlflow-ui` for the registry")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=COMMANDS)
    ap.add_argument("rest", nargs=argparse.REMAINDER, help="options passed to the underlying script")
    a = ap.parse_args(argv)
    c, rest = a.command, a.rest
    if c == "setup":
        return cmd_setup()
    if c == "stop":
        return cmd_stop()
    if c == "clean":
        return cmd_clean()
    if c == "eval":
        console.warn("not used in this project")
        return 0
    py = procs.resolve_python(ROOT)
    if py == sys.executable and not procs.venv_python(ROOT).exists():
        console.warn("no .venv found - using the current interpreter (run `python run.py setup` first)")
    if c == "verify":
        return cmd_verify(py)
    if c == "demo":
        return cmd_demo(py, rest)
    if c == "mlflow-ui":
        return sh(mlflow_cmd(py))
    cmd = build_command(c, py, rest)
    return sh(cmd) if cmd else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
