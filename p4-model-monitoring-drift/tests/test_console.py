from fraud_monitor import console, narrate


def _all(fn_calls, capsys):
    for f in fn_calls:
        f()
    return capsys.readouterr().out


def _calls():
    return [
        lambda: console.banner("T", "does a thing", "a result"),
        lambda: console.step(1, 2, "step one"),
        lambda: console.ok("good"), lambda: console.warn("hmm"), lambda: console.fail("bad"), lambda: console.hint("tip"),
        lambda: console.kv_table({"a": 1}, title="kv"),
        lambda: console.table([{"x": 1, "y": "z"}], title="tbl", highlight=lambda r: "red"),
        lambda: console.note("p", "c", "s"),
    ]


def test_plain_fallback_when_rich_missing(monkeypatch, capsys):
    monkeypatch.setattr(console, "_RICH", False)
    out = _all(_calls(), capsys)
    for word in ("does a thing", "[1/2] step one", "OK   good", "FAIL bad", "a: 1", "x | y", "Pros: p", "At scale: s"):
        assert word in out


def test_rich_path_and_no_color(monkeypatch, capsys):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(console, "_con", None)
    out = _all(_calls(), capsys)
    assert "does a thing" in out and "good" in out and "\x1b[" not in out  # no ANSI when piped / NO_COLOR


def test_window_line_alert_and_ok():
    r = {"window": 6, "drift_share": 0.34, "psi_score": 0.1, "pr_auc": 0.4, "baseline_pr_auc": 0.8, "alert": True,
         "reasons": ["drift_share 0.34 > 0.3"], "perf_skipped": None}
    lvl, txt = narrate.window_line(r)
    assert lvl == "fail" and "ALERT" in txt and "-50%" in txt
    lvl, txt = narrate.window_line({**r, "alert": False, "pr_auc": None, "perf_skipped": "labels pending"})
    assert lvl == "ok" and "labels pending" in txt
