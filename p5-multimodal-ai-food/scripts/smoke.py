"""Smoke test: start the API, wait for /health, analyze 3 real images, check 413, start the UI, then stop all.

Inputs:  data/nutrition.csv, cached eval images (data/eval_sample) or assets/*.jpg, cached models.
Outputs: console report + .run/api.log, .run/ui.log. Exit code 0 = ok.
Example: python run.py smoke   (or: python scripts/smoke.py)
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
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from food_lens import console  # noqa: E402
from run import kill_cmd, venv_python  # noqa: E402

API, UI = 8005, 7865


def http(url: str, data: bytes | None = None, headers: dict | None = None, timeout: int = 120) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=data, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except (urllib.error.URLError, OSError):
        return 0, b""


def multipart(fname: str, payload: bytes) -> tuple[bytes, dict]:
    b = uuid.uuid4().hex
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{fname}\"\r\n"
            "Content-Type: image/jpeg\r\n\r\n").encode() + payload + f"\r\n--{b}--\r\n".encode()
    return body, {"Content-Type": f"multipart/form-data; boundary={b}"}


def wait(url: str, tries: int, pause: float, want: str | None = None) -> bool:
    for _ in range(tries):
        code, body = http(url, timeout=5)
        if code == 200 and (want is None or want.encode() in body):
            return True
        time.sleep(pause)
    return False


def images() -> list[Path]:
    cached = sorted((ROOT / "data" / "eval_sample").glob("*.jpg"))
    if len(cached) > 201:
        return [cached[0], cached[100], cached[200]]
    return sorted((ROOT / "assets").glob("*.jpg"))[:3]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-ui", action="store_true", help="only test the API")
    args = ap.parse_args()
    console.setup_logging()
    console.banner("p5 smoke test", "Starts uvicorn on :8005, analyzes 3 photos, checks the 5 MB limit, starts the UI on :7865.",
                   "Each image prints dish + timings (~7 s/image with the VLM); ends with 'smoke ok'.")
    if not (ROOT / "data" / "nutrition.csv").exists():
        console.warn("smoke stub: data/nutrition.csv missing (run `python run.py data`)")
        return 0
    imgs = images()
    run_dir = ROOT / ".run"
    run_dir.mkdir(exist_ok=True)
    os.environ.update(PYTHONPATH=str(ROOT / "src"), HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                      PYTORCH_ENABLE_MPS_FALLBACK="1", PYTHONUTF8="1")
    procs: list[subprocess.Popen] = []
    try:
        console.step(1, 4, "starting API (loads CLIP + SmolVLM)")
        with open(run_dir / "api.log", "w") as log:
            p = subprocess.Popen([venv_python(), "-m", "uvicorn", "food_lens.api:app", "--port", str(API), "--workers", "1"],
                                 cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        procs.append(p)
        (run_dir / "api.pid").write_text(str(p.pid))
        if not wait(f"http://127.0.0.1:{API}/health", 150, 2, '"ok"'):
            console.fail("API not healthy; tail of .run/api.log:")
            print("\n".join((run_dir / "api.log").read_text(errors="replace").splitlines()[-20:]))
            return 1
        console.ok("API healthy")
        console.step(2, 4, f"analyzing {len(imgs)} images")
        rows = []
        for f in imgs:
            body, hdr = multipart(f.name, f.read_bytes())
            code, out = http(f"http://127.0.0.1:{API}/analyze", body, hdr)
            if code != 200:
                console.fail(f"/analyze {f.name} -> {code}")
                return 1
            d = json.loads(out)
            assert len(d["top3"]) == 3 and d["timings_ms"]["total"] > 0 and d["disclaimer"]
            rows.append([f.name, d["dish"], d["timings_ms"]["total"], "ok" if d["vlm"] else "vlm_err"])
        console.table(rows, ["image", "dish", "total ms", "vlm"], "analyze results")
        console.step(3, 4, "oversize upload must be rejected")
        body, hdr = multipart("big.jpg", b"a" * 6_000_000)
        code, _ = http(f"http://127.0.0.1:{API}/analyze", body, hdr)
        if code != 413:
            console.fail(f"oversize not rejected ({code})")
            return 1
        console.ok("6 MB upload -> 413")
        if not args.skip_ui:
            console.step(4, 4, "starting Gradio UI")
            with open(run_dir / "ui.log", "w") as log:
                u = subprocess.Popen([venv_python(), "scripts/ui.py"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            procs.append(u)
            (run_dir / "ui.pid").write_text(str(u.pid))
            if not wait(f"http://127.0.0.1:{UI}", 30, 1):
                console.fail("UI not up; see .run/ui.log")
                return 1
            console.ok(f"UI HTTP 200 on :{UI}")
            import importlib.util
            spec = importlib.util.spec_from_file_location("ui", ROOT / "scripts" / "ui.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            top, nut, md, st = mod.analyze(str(imgs[0]))
            assert len(top) == 3 and "Recipe ideas" in md and "Timings" in st
            console.ok(f"ui.analyze end-to-end ok ({len(nut)} nutrient rows)")
        console.ok(f"smoke ok: {len(rows)} images analyzed, oversize rejected")
        return 0
    finally:
        for p in procs:
            if p.poll() is None:
                subprocess.call(kill_cmd(p.pid), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for f in ("api.pid", "ui.pid"):
            (run_dir / f).unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
