"""Eval over eval/qa.jsonl -> reports/eval.json + reports/eval.md (indicative only).

Inputs: eval/qa.jsonl (31 auto-drafted questions), Chroma index. Outputs: recall/refusal table, reports/eval.{json,md}.
Sample: python scripts/run_eval.py   (runs the LLM on every question; a few minutes)
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from study_assistant import console  # noqa: E402
from study_assistant.config import load_config, resolve  # noqa: E402
from study_assistant.index import get_store  # noqa: E402
from study_assistant.rag import ask, label, retrieve  # noqa: E402


def load_qa(path: Path) -> list[dict]:
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    return [r for r in rows if "question" in r]


def pick_threshold(top: list[tuple[float, bool]]) -> float:
    """Gate accuracy sweep: refuse iff top < thr. Ties -> the largest-margin midpoint."""
    cands = sorted({round(s, 3) for s, _ in top})
    cands = [(a + b) / 2 for a, b in zip(cands, cands[1:], strict=False)]
    best = max(cands, key=lambda t: (sum((s >= t) == ans for s, ans in top), -abs(t - 0.65)))
    return round(best, 3)


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    console.setup_logging(logging.WARNING)
    console.banner("RAG eval", "Retrieval metrics, refusal-gate sweep, then answer every question.",
                   "recall@4, MRR, refusal accuracy (indicative only: questions were auto-drafted, not human-reviewed)")
    cfg = load_config()
    console.step(1, 3, "loading index and computing retrieval metrics")
    store = get_store(cfg)
    qa = load_qa(resolve(cfg, "eval") / "qa.jsonl")
    ans_qs = [q for q in qa if q["answerable"]]
    # retrieval metrics + gate sweep (no LLM)
    rets, rr, hit = {}, [], []
    for q in qa:
        rets[q["id"]] = r = retrieve(store, q["question"], cfg)
        if q["answerable"]:
            gold = set(q["gold_chunk_ids"])
            hit.append(bool(gold & set(r.ids)))
            ranks = [i + 1 for i, c in enumerate(r.ids) if c in gold]
            rr.append(1 / min(ranks) if ranks else 0.0)
    tops = [(rets[q["id"]].top, q["answerable"]) for q in qa]
    old = cfg["retrieval"]["score_threshold"]
    new = pick_threshold(tops)
    gate_acc = lambda t: sum((s >= t) == a for s, a in tops) / len(tops)  # noqa: E731
    cfg["retrieval"]["score_threshold"] = new
    per = []
    console.step(2, 3, f"answering {len(qa)} questions (LLM on every one)")
    for q in qa:
        r = ask(store, q["question"], cfg=cfg)
        resolved = [c for c in r.get("model_cited", []) if c in {label(d) for d in rets[q["id"]].docs}]
        per.append({"id": q["id"], "type": q["type"], "answerable": q["answerable"], "top": r["top_score"],
                    "refused": r["refused"], "gate": r["gate"], "model_cited": r.get("model_cited", []),
                    "resolved": resolved, "answer": r["answer"][:300]})
        console.hint(f"{q['id']} {q['type']} top={r['top_score']} {'refused' if r['refused'] else 'answered'}")
    answered = [p for p in per if p["answerable"] and not p["refused"]]
    cited = [p for p in answered if p["resolved"]]
    all_cites = sum(len(p["model_cited"]) for p in answered)
    ok_cites = sum(len(p["resolved"]) for p in answered)
    un = [p for p in per if not p["answerable"]]
    out = {
        "note": "auto-drafted questions, not human-reviewed; indicative only (questions drawn from the indexed chunks)",
        "n_questions": len(qa), "n_answerable": len(ans_qs), "n_unanswerable": len(un),
        "recall_at_4": round(sum(hit) / len(hit), 3), "mrr": round(sum(rr) / len(rr), 3),
        "citation_rate_model": round(len(cited) / max(len(answered), 1), 3),
        "citation_resolve_rate": round(ok_cites / max(all_cites, 1), 3),
        "sources_fallback_rate": round(sum(1 for p in answered if not p["model_cited"]) / max(len(answered), 1), 3),
        "refusal_accuracy_unanswerable": round(sum(p["refused"] for p in un) / len(un), 3),
        "answer_rate_answerable": round(len(answered) / len(ans_qs), 3),
        "overall_refusal_accuracy": round(sum(p["refused"] != p["answerable"] for p in per) / len(per), 3),
        "threshold": {"old": old, "new": new, "gate_acc_old": round(gate_acc(old), 3), "gate_acc_new": round(gate_acc(new), 3),
                      "min_top_answerable": round(min(s for s, a in tops if a), 3),
                      "max_top_unanswerable": round(max(s for s, a in tops if not a), 3)},
        "per_question": per,
    }
    rep = resolve(cfg, "reports")
    (rep / "eval.json").write_text(json.dumps(out, indent=1))
    keys = [k for k in out if k not in ("per_question", "note", "threshold")]
    (rep / "eval.md").write_text("# Eval (indicative only)\n\n" + out["note"] + "\n\n" +
                                 "\n".join(f"- {k}: {out[k]}" for k in keys) + f"\n- threshold: {out['threshold']}\n")
    console.step(3, 3, "results (also in reports/eval.md)")
    console.table([[k, out[k]] for k in keys], ["metric", "value"], title="Eval (indicative only)")
    console.kv_table(out["threshold"], title="Refusal gate threshold")
    console.note("measures retrieval and refusal separately from generation", "small n, in-sample gate tuning",
                 "held-out human-reviewed set and continuous eval in CI")


if __name__ == "__main__":
    main()
