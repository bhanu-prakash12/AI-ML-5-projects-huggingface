import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("ui", Path(__file__).parents[1] / "scripts" / "ui.py")
ui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ui)


def test_analyze_without_image_is_friendly():
    assert "first" in ui.analyze(None)[3]


def test_analyze_api_down_reports_error(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, "API", "http://127.0.0.1:1")
    f = tmp_path / "x.jpg"
    f.write_bytes(b"x")
    assert "API unreachable" in ui.analyze(str(f))[3]


def test_build_blocks():
    assert ui.build() is not None


def test_shrink_downsizes_large_image(tmp_path):
    from PIL import Image

    src = tmp_path / "big.png"
    Image.new("RGB", (3000, 2000), "red").save(src)
    out = Path(ui.shrink(str(src)))
    with Image.open(out) as im:
        assert max(im.size) <= 1280 and im.format == "JPEG"


def test_shrink_passes_through_unreadable(tmp_path):
    f = tmp_path / "x.heic"
    f.write_bytes(b"not an image")
    assert ui.shrink(str(f)) == str(f)
