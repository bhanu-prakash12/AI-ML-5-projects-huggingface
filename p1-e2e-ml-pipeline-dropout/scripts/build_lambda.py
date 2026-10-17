"""Build the arm64 Lambda zip WITHOUT Docker (optional, local only; nothing is deployed).

Needs `uv` on PATH. Works on Windows too (uv cross-installs aarch64 Linux wheels), but the zip is
only useful if you plan a Lambda deploy; the demo does not need it.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / "build"


def _size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main() -> int:
    shutil.rmtree(B / "lambda", ignore_errors=True)
    (B / "lambda.zip").unlink(missing_ok=True)
    out = B / "lambda"
    (out / "configs").mkdir(parents=True)
    (out / "artifacts" / "model").mkdir(parents=True)
    subprocess.run(
        ["uv", "pip", "install", "--python", sys.executable, "--python-platform",
         "aarch64-manylinux_2_17", "--python-version", "3.11", "--only-binary", ":all:",
         "--target", str(out), "-r", str(ROOT / "requirements-lambda.txt")], check=True)
    shutil.copytree(ROOT / "src" / "dropout_risk", out / "dropout_risk")
    shutil.copy(ROOT / "configs" / "config.yaml", out / "configs")
    for f in ("model.txt", "preprocess.json"):
        shutil.copy(ROOT / "artifacts" / "model" / f, out / "artifacts" / "model")
    for d in list(out.rglob("__pycache__")) + list(out.rglob("*.dist-info")):
        shutil.rmtree(d, ignore_errors=True)
    with zipfile.ZipFile(B / "lambda.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in out.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(out))
    unz, zp = _size(out) / 2**20, (B / "lambda.zip").stat().st_size / 2**20
    print(f"unzipped_mb={unz:.0f} zip_mb={zp:.0f} (limit: 250 MB unzipped, 50 MB zip direct upload)")
    if unz >= 250:
        print("FAIL: unzipped >= 250 MB")
        return 1
    return 0


if __name__ == "__main__":
    argparse.ArgumentParser(
        description="Build build/lambda.zip for AWS Lambda (arm64, python3.11), local only.",
        epilog="Input: artifacts/model, requirements-lambda.txt, uv. Output: build/lambda.zip. "
               "Sample: python scripts/build_lambda.py").parse_args()
    raise SystemExit(main())
