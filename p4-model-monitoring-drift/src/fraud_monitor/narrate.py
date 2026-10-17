"""Plain-English one-liners per monitoring window (pure functions; numbers come straight from status rows)."""
from __future__ import annotations

from typing import Any, Mapping


def window_line(r: Mapping[str, Any]) -> tuple[str, str]:
    """Return (level, text); level in ok|warn|fail for colouring."""
    pr = r.get("pr_auc")
    base = r.get("baseline_pr_auc")
    head = f"window {int(r['window']):02d}: drift share {r['drift_share']:.2f}, prediction PSI {r['psi_score']:.3f}"
    if pr is not None and pr == pr:
        drop = f" ({(pr / base - 1) * 100:+.0f}% vs baseline {base:.3f})" if base else ""
        head += f", PR-AUC {pr:.3f}{drop}"
    if r.get("alert"):
        return "fail", f"{head} -> ALERT: {'; '.join(r.get('reasons') or [])}"
    skipped = r.get("perf_skipped")
    return "ok", f"{head} -> ok" + (f" ({skipped})" if skipped else "")


def explain_retrain(res: Mapping[str, Any]) -> list[str]:
    ch, cr = res["champion"], res["challenger"]
    verdict = "challenger passes the gate" if res["promoted"] else "challenger rejected"
    lines = [f"evaluated on window {res['eval_window']} ({res['n_new_labelled_rows']} new labelled rows, "
             f"{res['n_new_frauds']} frauds)",
             f"champion v{ch['version']} PR-AUC {ch['window']['pr_auc']:.3f} vs challenger v{cr['version']} "
             f"{cr['window']['pr_auc']:.3f} -> {verdict}: {res['reason']}"]
    if res["promoted"] and not res.get("alias_updated", True):
        lines.append("alias NOT moved (dry run: pass --promote / tick the box to switch the champion)")
    return lines
