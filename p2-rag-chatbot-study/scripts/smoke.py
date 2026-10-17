"""Smoke test: start the API (both models, one process), wait for /health, ask 2 questions, stop.

Inputs: none (uses configs/config.yaml; P2_API_PORT overrides the port).
Outputs: exit code 0 and a short table; API log in .run/api.log.
Sample: python scripts/smoke.py
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from study_assistant import console  # noqa: E402
from study_assistant.config import api_port  # noqa: E402


def post(url: str, body: dict) -> str:
    req = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read().decode()


def healthy(base: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=3) as r:
            return json.loads(r.read()).get("status") == "ok"
    except Exception:
        return False


def main() -> int:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    console.setup_logging()
    console.banner("Smoke test", "Start the API, ask one in-corpus and one out-of-corpus question, stop the API.",
                   "in-corpus answered with sources; out-of-corpus refused (score below the gate)")
    port = api_port()
    base = f"http://127.0.0.1:{port}"
    run = ROOT / ".run"
    run.mkdir(exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "HF_HUB_OFFLINE": "1", "ANONYMIZED_TELEMETRY": "False",
           "PYTORCH_ENABLE_MPS_FALLBACK": "1", "TOKENIZERS_PARALLELISM": "false"}
    console.step(1, 4, f"starting API on port {port} (first model load takes several seconds)")
    with open(run / "api.log", "w") as log:
        proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "study_assistant.api:app", "--port", str(port),
                                 "--workers", "1"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    (run / "api.pid").write_text(str(proc.pid))
    try:
        for _ in range(120):
            if healthy(base) or proc.poll() is not None:
                break
            time.sleep(2)
        if not healthy(base):
            console.fail("API not healthy; tail of .run/api.log:")
            print("\n".join((run / "api.log").read_text().splitlines()[-20:]))
            return 1
        console.ok("API healthy")
        console.step(2, 4, "in-corpus question via /ask (SSE)")
        sse = post(f"{base}/ask", {"question": "What is the bias-variance tradeoff?"})
        if '"refused": false' not in sse or "event: sources" not in sse:
            console.fail("in-corpus question not answered with sources")
            print(sse[-400:])
            return 1
        console.ok("answered with sources")
        console.step(3, 4, "out-of-corpus question via /ask/simple (JSON)")
        out = json.loads(post(f"{base}/ask/simple", {"question": "Who won the 1998 football world cup?"}))
        if not out["refused"]:
            console.fail(f"not refused: {out['answer'][:200]}")
            return 1
        console.ok(f"refused (top score {out['top_score']})")
        console.step(4, 4, "stopping API")
        console.table([["in-corpus", "answered + sources"], ["out-of-corpus", f"refused, top={out['top_score']}"]],
                      ["question", "result"])
        console.ok("smoke ok")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        (run / "api.pid").unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
