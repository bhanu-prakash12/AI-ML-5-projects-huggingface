"""Start the API on a temp prediction DB, call /health, /predict, /feedback once, stop it (cross-platform)."""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraud_monitor import console, examples, procs  # noqa: E402


def main() -> int:
    argparse.ArgumentParser(description=__doc__, epilog="Outputs: nothing persistent (temp DB .run/smoke.db is removed).\n"
                            "Example: python run.py smoke").parse_args()
    try:
        import yaml
        port = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())["ports"]["api"]
    except Exception:  # noqa: BLE001
        port = 8004
    run_dir = ROOT / ".run"
    run_dir.mkdir(exist_ok=True)
    db = run_dir / "smoke.db"
    for f in run_dir.glob("smoke.db*"):
        f.unlink(missing_ok=True)
    base = f"http://127.0.0.1:{port}"
    console.banner("smoke test", f"Start the API on port {port} with a temporary prediction DB, score one row, send its label.",
                   "OK for /health, /predict (logged) and /feedback (joined)")
    if procs.port_in_use(port):
        console.fail(f"port {port} is busy - run `python run.py stop` or free the port")
        return 1
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "P4_DB": str(db), "PYTHONUNBUFFERED": "1"}
    pid_path = run_dir / "api.pid"
    console.step(1, 4, "starting uvicorn")
    proc = procs.start_background([procs.resolve_python(ROOT), "-m", "uvicorn", "fraud_monitor.api:app", "--port", str(port),
                                   "--workers", "1"], env, ROOT, run_dir / "api.log", pid_path)
    try:
        console.step(2, 4, "waiting for /health (model load takes a few seconds)")
        if not procs.wait_health(base + "/health", 90, proc):
            console.fail("API not healthy")
            console.hint(procs.tail(run_dir / "api.log"))
            return 1
        console.ok("/health ok")
        console.step(3, 4, "POST /predict with a real legit reference row (all features)")
        body = examples.predict_examples()[0]
        body = {**body, "request_id": "smoke1"}
        out = procs.http_json(base + "/predict", "POST", body)
        if "score" not in out:
            console.fail(f"no score in {out}")
            return 1
        console.kv_table({"request_id": out["request_id"], "score": f"{out['score']:.6f}", "is_fraud": out["is_fraud"],
                          "model_version": out["model_version"]})
        console.step(4, 4, "POST /feedback with the true label")
        fb = procs.http_json(base + "/feedback", "POST", {"request_id": "smoke1", "label": 0})
        if fb.get("matched") != 1:
            console.fail(f"feedback not matched: {fb}")
            return 1
        console.ok("smoke ok: /health, /predict (logged), /feedback (joined)")
        return 0
    except Exception as e:  # noqa: BLE001
        console.fail(f"smoke failed: {e!r}")
        console.hint(procs.tail(run_dir / "api.log"))
        return 1
    finally:
        procs.terminate(proc.pid)
        try:
            proc.wait(10)
        except Exception:  # noqa: BLE001
            pass
        pid_path.unlink(missing_ok=True)
        for f in run_dir.glob("smoke.db*"):
            f.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
