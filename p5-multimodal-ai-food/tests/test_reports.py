import json
from pathlib import Path

import pytest

R = Path(__file__).resolve().parents[1] / "reports" / "clip_eval.json"


@pytest.mark.skipif(not R.exists(), reason="run make eval first")
def test_clip_eval_report_sane():
    d = json.loads(R.read_text())
    assert d["n_images"] >= 100 and d["top5"] >= d["top1"] > 0.5
