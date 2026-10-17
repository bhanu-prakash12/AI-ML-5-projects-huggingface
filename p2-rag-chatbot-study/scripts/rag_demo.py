"""Narrated RAG demo: 3 in-corpus + 2 out-of-corpus questions -> reports/rag_demo.md.

Inputs: Chroma index (run `run.py index` first). Outputs: console table with refusal-gate score per query, reports/rag_demo.md.
Sample: python scripts/rag_demo.py   (loads both models; first load ~10 s)
"""
import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from study_assistant import console  # noqa: E402
from study_assistant.config import load_config, resolve  # noqa: E402
from study_assistant.index import get_store  # noqa: E402
from study_assistant.rag import ask  # noqa: E402

IN = ["What is the bias-variance tradeoff?", "How does gradient descent update model parameters?",
      "What does a support vector machine maximize?"]
OUT = ["Who won the FIFA World Cup in 2018?", "What is the recipe for chocolate chip cookies?"]


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    console.setup_logging(logging.WARNING)
    console.banner("RAG demo", "Ask in-corpus and out-of-corpus questions; the refusal gate compares the best similarity to a threshold.",
                   "in-corpus: answered with citations; out-of-corpus: refused without calling the LLM")
    cfg = load_config()
    console.step(1, 2, "loading embedding model + index (first load ~10 s)")
    store = get_store(cfg)
    thr = cfg["retrieval"]["score_threshold"]
    console.step(2, 2, f"asking {len(IN) + len(OUT)} questions (gate threshold {thr})")
    rows = []
    out = ["# RAG demo", ""]
    for kind, qs in (("in-corpus", IN), ("out-of-corpus", OUT)):
        for q in qs:
            t0 = time.time()
            r = ask(store, q, cfg=cfg)
            out += [f"## [{kind}] {q}", "", f"- top_score: {r['top_score']}, refused: {r['refused']}, gate: {r['gate']}, "
                    f"citations: {r.get('citations', [])}, {time.time() - t0:.1f}s", "", r["answer"], ""]
            console.hint(f"[{kind}] {q}")
            console.hint(f"  gate score {r['top_score']} vs threshold {thr} -> {'REFUSED' if r['refused'] else 'answered'}")
            rows.append([kind, q, r["top_score"], "refused" if r["refused"] else "answered", ", ".join(r.get("citations") or [])])
    (resolve(cfg, "reports") / "rag_demo.md").write_text("\n".join(out))
    console.table(rows, ["kind", "question", "top score", "outcome", "citations"], title="Results")
    console.note("answers grounded in your notes, with citations and a cheap refusal gate", "1.5B model: weak at multi-chunk synthesis",
                 "GPU serving (vLLM), semantic cache and a reranker")


if __name__ == "__main__":
    main()
