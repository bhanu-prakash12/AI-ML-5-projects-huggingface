"""Download the 3 pinned-revision models (embedding, LLM, fallback LLM) into the Hugging Face cache.

Inputs: pinned names + revisions in configs/config.yaml. Outputs: files in the HF cache (reused if present).
Sample: python scripts/download_models.py   (needs network once; afterwards run with HF_HUB_OFFLINE=1)
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from huggingface_hub import snapshot_download  # noqa: E402

from study_assistant import console  # noqa: E402
from study_assistant.config import load_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("download_models")


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    console.banner("Download models", "Fetch pinned model revisions into the HF cache.", "cached models are reused; first download is the slow part")
    cfg = load_config()
    for key in ("embedding", "llm", "llm_fallback"):
        m = cfg["models"][key]
        if m["revision"] in (None, "TBD"):
            sys.exit(f"revision for {key} not pinned")
        path = snapshot_download(m["name"], revision=m["revision"])
        console.ok(f"{m['name']}@{m['revision'][:8]} -> {path}")
    console.note("pinned revisions make runs reproducible and offline-capable", "multi-GB download once", "a model registry or artifact cache replaces per-machine downloads")


if __name__ == "__main__":
    main()
