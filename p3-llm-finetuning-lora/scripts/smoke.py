"""Start the API (one model + adapter), wait for /health, check base, tuned, emergency, bad mode; stop.

usage: python scripts/smoke.py [--demo] [--port 8003]
  default : quiet checks, exit code 0/1 (used by `make smoke` / `python run.py smoke`)
  --demo  : same checks, narrated with the actual answers side by side (used by `python run.py demo`)
Requires artifacts/adapter (the adapter already exists; `python run.py train` recreates it).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from med_assistant import console  # noqa: E402

Q = "What are the symptoms of type 2 diabetes?"


def call(port: int, path: str, body: dict | None = None, timeout: int = 300) -> tuple[int, dict | None]:
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, None
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
        return 0, None


def wait_healthy(port: int, proc: subprocess.Popen, tries: int = 120) -> bool:
    for _ in range(tries):
        if proc.poll() is not None:
            return False
        st, j = call(port, "/health", timeout=5)
        if st == 200 and j and j.get("status") == "ok":
            return bool(j.get("adapter_loaded"))
        time.sleep(2)
    return False


def start_api(port: int) -> subprocess.Popen:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    (ROOT / ".run").mkdir(exist_ok=True)
    log = open(ROOT / ".run" / "api.log", "w", encoding="utf-8")  # noqa: SIM115 (lives as long as the child)
    p = subprocess.Popen([sys.executable, "-m", "uvicorn", "med_assistant.api:app", "--port", str(port),
                          "--workers", "1"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    (ROOT / ".run" / "api.pid").write_text(str(p.pid))
    return p


def stop_api(p: subprocess.Popen) -> None:
    if p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            p.kill()
    (ROOT / ".run" / "api.pid").unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="narrate each step and print the answers")
    ap.add_argument("--port", type=int, default=8003)
    a = ap.parse_args()
    demo, port = a.demo, a.port
    if demo:
        console.banner("DEMO: base vs LoRA-tuned vs safety refusal",
                       "Starts the API (ONE model, adapter toggled per request) and sends 4 requests.",
                       "tuned and base answers differ; the chest-pain question is refused with no model call; bad mode -> HTTP 422.")
    if not (ROOT / "artifacts" / "adapter").exists():
        console.fail("no adapter: run `python run.py train` first")
        return 1
    if call(port, "/health", timeout=2)[0] == 200:
        console.fail(f"port {port} already in use (another API running?). Run `python run.py stop` or free the port.")
        return 1
    console.step(1, 5, f"starting API on :{port} and loading model + adapter (~20-60 s; slower on CPU)") if demo else None
    proc = start_api(port)
    try:
        if not wait_healthy(port, proc):
            console.fail("API not healthy; tail of .run/api.log:")
            log = (ROOT / ".run" / "api.log").read_text(encoding="utf-8", errors="replace").splitlines()
            print("\n".join(log[-20:]))
            return 1
        if demo:
            console.ok("API healthy, adapter loaded")
            console.step(2, 5, f'tuned model: "{Q}"')
        t = call(port, "/generate", {"question": Q, "mode": "tuned", "max_new_tokens": 60})
        if demo:
            console.step(3, 5, "same question, base model (adapter switched off)")
        b = call(port, "/generate", {"question": Q, "mode": "base", "max_new_tokens": 60})
        if demo:
            console.step(4, 5, "emergency question -> safety filter (expect refused=true)")
        e = call(port, "/generate", {"question": "I have severe chest pain and cannot breathe", "mode": "tuned"})
        c = call(port, "/generate", {"question": "hello there", "mode": "x"})
        if demo:
            console.step(5, 5, "invalid mode -> validation error")
            tj, bj, ej = t[1] or {}, b[1] or {}, e[1] or {}
            console.table([["tuned", tj.get("answer", "")[:220], tj.get("seconds")],
                           ["base", bj.get("answer", "")[:220], bj.get("seconds")],
                           ["emergency", ej.get("answer", "")[:120], ej.get("seconds")]],
                          ["mode", "answer (truncated)", "seconds"], "Answers")
            console.hint("Honest result: reports/compare.md shows NO measurable ROUGE-L gain; the adapter changed style, not accuracy.")
        failures = []
        if not (t[0] == 200 and t[1] and not t[1]["refused"] and t[1]["mode"] == "tuned"):
            failures.append(f"tuned: {t}")
        if not (b[0] == 200 and b[1] and not b[1]["refused"] and b[1]["mode"] == "base"):
            failures.append(f"base: {b}")
        if t[1] and b[1] and t[1]["answer"] == b[1]["answer"]:
            failures.append("base and tuned identical")
        if not (e[0] == 200 and e[1] and e[1]["refused"]):
            failures.append(f"emergency not refused: {e}")
        if c[0] != 422:
            failures.append(f"bad mode returned {c[0]}")
        if failures:
            for f in failures:
                console.fail(f"smoke FAIL: {f}")
            return 1
        console.ok("smoke ok: tuned+base answered (differ), emergency refused, bad mode 422")
        if demo:
            console.note("one process, one model, both modes; safety check before any generation",
                         "keyword refusal is crude; 0.5B answers can be wrong; requests are serialized by a lock",
                         "vLLM multi-LoRA behind a gateway with safety filter and eval gates")
        return 0
    finally:
        stop_api(proc)


if __name__ == "__main__":
    sys.exit(main())
