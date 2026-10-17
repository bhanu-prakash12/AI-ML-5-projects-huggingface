"""Stage the hub for local run / upload: hf-hub/ + only the runtime files of p1..p5 (no data, venv, mlruns, big HTML).

usage: python scripts/stage_hub.py [--out .run/hub_stage]
Prereq: p5 data (`python run.py data` in p5) so data/nutrition.csv and labels.json exist.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HUB = ROOT / "hf-hub"
IGN = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")
SMALL_REPORT = 300_000  # bytes

# project -> (folder, [dirs/files copied verbatim], required paths)
PROJECTS = {
    "p1": ("p1-e2e-ml-pipeline-dropout", ["src/dropout_risk", "configs/config.yaml", "artifacts/model", "scripts/ui.py",
                                           "scripts/_boot.py", "reports/metrics.json", "reports/fairness.md"],
           ["artifacts/model/model.txt"]),
    "p2": ("p2-rag-chatbot-study", ["src/study_assistant", "configs/config.yaml", "chroma", "reports/eval.md"],
           ["chroma/chroma.sqlite3"]),
    "p3": ("p3-llm-finetuning-lora", ["src/med_assistant", "configs/config.yaml", "artifacts/adapter",
                                       "reports/compare.json", "reports/compare.md", "reports/train.md"],
           ["artifacts/adapter/adapter_model.safetensors"]),
    "p4": ("p4-model-monitoring-drift", ["src/fraud_monitor", "configs/config.yaml", "reports/baseline.json",
                                          "reports/scenB/status.json", "reports/scenB/alerts.jsonl",
                                          "reports/scenB/retrain.json",
                                          "reports/scenB/windows/window_05.html", "reports/scenB/windows/window_06.html"],
           ["reports/scenB/status.json"]),
    "p5": ("p5-multimodal-ai-food", ["src/food_lens", "configs/config.yaml", "scripts/ui.py", "data/nutrition.csv",
                                      "data/labels.json", "artifacts/clip_text_emb.npy", "assets/sample-output.png"],
           ["data/nutrition.csv", "data/labels.json"]),
}


def copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, ignore=IGN, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / ".run" / "hub_stage"))
    out = Path(ap.parse_args().out)
    if out.exists():
        shutil.rmtree(out)
    for f in ("app.py", "README.md", "requirements.txt"):
        copy(HUB / f, out / f)
    copy(HUB / "hub", out / "hub")
    for pid, (folder, items, required) in PROJECTS.items():
        base = ROOT / folder
        missing = [r for r in required if not (base / r).exists()]
        if missing:
            sys.exit(f"{pid}: missing required {missing} (run the project's data/train step first)")
        for it in items:
            if (base / it).exists():
                copy(base / it, out / "projects" / pid / it)
            else:
                print(f"skip (absent): {folder}/{it}")
        if pid == "p5":  # sample photos (<=100 KB each, as in p5's own deploy script)
            for jpg in (base / "assets").glob("sample_*.jpg"):
                if jpg.stat().st_size <= 100_000:
                    copy(jpg, out / "projects" / pid / "assets" / jpg.name)
    (out / "projects" / "p4" / "reports" / "demo").mkdir(parents=True, exist_ok=True)
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"staged {sum(1 for f in out.rglob('*') if f.is_file())} files, {total / 1e6:.1f} MB -> {out}")


if __name__ == "__main__":
    main()
