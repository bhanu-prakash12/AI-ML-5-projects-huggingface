"""Create (if needed) and upload the hub Space. Token comes from the HF_TOKEN env var only; never written anywhere.

usage: HF_TOKEN=... python scripts/deploy_hub.py --space <user>/ai-eng-projects [--hardware zero-a10g] [--no-upload]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / ".run" / "hub_stage"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True)
    ap.add_argument("--hardware", default="zero-a10g")
    ap.add_argument("--no-upload", action="store_true")
    a = ap.parse_args()
    subprocess.run([sys.executable, str(ROOT / "scripts" / "stage_hub.py"), "--out", str(STAGE)], check=True)
    if a.no_upload:
        return
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("set HF_TOKEN (write access) in the environment")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    print("authenticated as", api.whoami()["name"])
    if not api.repo_exists(a.space, repo_type="space"):
        api.create_repo(a.space, repo_type="space", space_sdk="gradio")
    cur = api.get_space_runtime(a.space)
    print("current:", cur.stage, cur.hardware)
    if cur.hardware != a.hardware:
        try:
            api.request_space_hardware(a.space, a.hardware)
            print("hardware requested:", a.hardware)
        except Exception as e:  # noqa: BLE001
            print(f"could not set hardware {a.hardware} via API ({type(e).__name__}); set it in the Space settings")
    # delete_patterns removes stale files of the previous app (old root-level src/, scripts/, data/ ...)
    api.upload_folder(folder_path=str(STAGE), repo_id=a.space, repo_type="space", delete_patterns=["*"],
                      commit_message="Deploy AI Engineering Projects hub")
    for _ in range(60):
        st = api.get_space_runtime(a.space)
        print("stage:", st.stage, "hardware:", st.hardware)
        if st.stage in ("RUNNING", "RUNTIME_ERROR", "BUILD_ERROR", "CONFIG_ERROR", "NO_APP_FILE"):
            break
        time.sleep(20)
    print(f"https://huggingface.co/spaces/{a.space}")


if __name__ == "__main__":
    main()
