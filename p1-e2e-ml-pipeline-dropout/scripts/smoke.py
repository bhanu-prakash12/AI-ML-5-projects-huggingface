"""Start the API in the background, wait for /health, run real requests, stop it (any OS)."""
from __future__ import annotations

import _boot  # noqa: F401  (adds src/ to sys.path)
import argparse
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import smoke_client

from dropout_risk import console

ROOT = Path(__file__).resolve().parents[1]
PORT = 8000


def healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(url + "/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def main(port: int = PORT) -> int:
    run = ROOT / ".run"
    run.mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    url = f"http://127.0.0.1:{port}"
    env = os.environ | {"PYTHONPATH": str(ROOT / "src")}
    log = (run / "api.log").open("w")
    console.step(1, 3, f"starting uvicorn on :{port} (background)")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "dropout_risk.api:app", "--port", str(port),
         "--log-level", "warning"], cwd=ROOT, env=env, stdout=log, stderr=log)
    (run / "api.pid").write_text(str(proc.pid))
    try:
        console.step(2, 3, "waiting for /health")
        for _ in range(120):
            if healthy(url) or proc.poll() is not None:
                break
            time.sleep(0.5)
        if not healthy(url):
            console.fail("API did not start; tail of .run/api.log:")
            log.flush()
            print("\n".join((run / "api.log").read_text().splitlines()[-20:]))
            return 1
        console.step(3, 3, "sending real requests")
        smoke_client.main(url)
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        (run / "api.pid").unlink(missing_ok=True)
        log.close()


if __name__ == "__main__":
    argparse.ArgumentParser(
        description="Smoke test: start the API, call it for real, stop it.",
        epilog="Needs a trained model (python run.py train). Output: console + "
               "reports/latency.md. Sample: python scripts/smoke.py").parse_args()
    console.setup_logging()
    raise SystemExit(main())
