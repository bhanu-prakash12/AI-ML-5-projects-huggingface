"""Stage the Food Lens app for a Hugging Face Space and upload it.

Inputs:  HF_TOKEN (env, write access), a Space id like user/ai-food; run `python run.py data` first
         (data/nutrition.csv and data/labels.json are gitignored and must exist).
Outputs: a staging folder (default .run/hf_space) and, unless --no-upload, a new commit on the Space.
Example: HF_TOKEN=hf_xxx python scripts/deploy_space.py --space user/ai-food
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from food_lens import console  # noqa: E402

COPY = [  # (source relative to project root, destination in the Space)
    ("deploy/hf_space/app.py", "app.py"), ("deploy/hf_space/README.md", "README.md"),
    ("deploy/hf_space/requirements.txt", "requirements.txt"), ("src/food_lens", "src/food_lens"),
    ("configs/config.yaml", "configs/config.yaml"), ("scripts/ui.py", "scripts/ui.py"),
    ("data/nutrition.csv", "data/nutrition.csv"), ("data/labels.json", "data/labels.json"),
    ("artifacts/clip_text_emb.npy", "artifacts/clip_text_emb.npy"),  # optional cache
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", required=True, help="Space id, e.g. user/ai-food")
    ap.add_argument("--stage", default=str(ROOT / ".run" / "hf_space"), help="staging folder")
    ap.add_argument("--no-upload", action="store_true", help="only build the staging folder")
    a = ap.parse_args()
    console.setup_logging()
    console.banner("deploy: Hugging Face Space", "Copy the app files into a staging folder and upload them to the Space.",
                   "a new commit on the Space, which then rebuilds (a few minutes)")
    stage = Path(a.stage)
    console.step(1, 3, f"staging into {stage}")
    shutil.rmtree(stage, ignore_errors=True)
    for src, dst in COPY:
        s, d = ROOT / src, stage / dst
        if not s.exists():
            if src.startswith("artifacts/"):
                continue
            console.fail(f"missing {src} (run `python run.py data` first?)")
            return 1
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(s, d, ignore=shutil.ignore_patterns("__pycache__")) if s.is_dir() else shutil.copy2(s, d)
    for p in (ROOT / "assets").glob("*.jpg"):
        if p.stat().st_size <= 100_000:  # only the small bundled samples
            (stage / "assets").mkdir(exist_ok=True)
            shutil.copy2(p, stage / "assets" / p.name)
    console.ok(f"{sum(1 for f in stage.rglob('*') if f.is_file())} files staged")
    if a.no_upload:
        return 0
    token = os.environ.get("HF_TOKEN")
    if not token:
        console.fail("set HF_TOKEN (a write token from https://huggingface.co/settings/tokens); it is never written to disk")
        return 1
    from huggingface_hub import HfApi

    console.step(2, 3, f"uploading to {a.space}")
    api = HfApi(token=token)
    api.upload_folder(folder_path=str(stage), repo_id=a.space, repo_type="space", commit_message="Deploy Food Lens")
    console.step(3, 3, "checking the Space status")
    rt = api.get_space_runtime(a.space)
    console.ok(f"stage: {rt.stage} (hardware requested: {rt.raw.get('hardware', {}).get('requested')})")
    console.ok(f"Logs and status: https://huggingface.co/spaces/{a.space}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
