"""Embed the corpus into Chroma (only unseen chunks) and probe 3 queries -> reports/probe.md.

Inputs: data/corpus. Outputs: chroma/ index, reports/probe.md. First run downloads/loads bge-small (~30 s embedding).
Sample: python scripts/probe.py
"""
import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from study_assistant import console  # noqa: E402
from study_assistant.config import load_config, resolve  # noqa: E402
from study_assistant.index import add_chunks, get_store, search  # noqa: E402
from study_assistant.ingest import ingest_corpus  # noqa: E402

QUERIES = [
    "What is the bias-variance tradeoff?",
    "How does gradient descent update model parameters?",
    "Explain how a support vector machine finds a decision boundary",
]


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    console.setup_logging(logging.WARNING)
    console.banner("Index + probe", "Embed chunks into Chroma and show the top-4 hits for 3 queries.",
                   "similarity roughly 0.78-0.89 for in-corpus queries (see reports/probe.md)")
    cfg = load_config()
    console.step(1, 3, "chunking corpus")
    ch = cfg["chunking"]["notes"]
    chunks, _ = ingest_corpus(resolve(cfg, "corpus"), ch["size"], ch["overlap"])
    console.step(2, 3, "loading embedding model (first load can take a while) and indexing unseen chunks")
    t0 = time.time()
    store = get_store(cfg)
    added = add_chunks(store, chunks)
    total = store._collection.count()
    instr = cfg["models"]["embedding"]["query_instruction"]
    out = ["# Probe report", "", f"- chunks in index: {total} (added this run: {added}, {time.time() - t0:.0f}s)",
           f"- metric: cosine similarity, k={cfg['retrieval']['k']}", ""]
    console.step(3, 3, "probing 3 queries")
    for q in QUERIES:
        out += [f"## {q}", "", "| # | sim | source | page | snippet |", "|---|---|---|---|---|"]
        hits = search(store, q, cfg["retrieval"]["k"], instruction=instr)
        console.table([[i, f"{s:.3f}", d.metadata.get("source"), d.metadata.get("page", "")] for i, (d, s) in enumerate(hits, 1)],
                      ["#", "sim", "source", "page"], title=q)
        for i, (d, s) in enumerate(hits, 1):
            snip = d.page_content[:110].replace("\n", " ").replace("|", "/")
            out.append(f"| {i} | {s:.3f} | {d.metadata.get('source')} | {d.metadata.get('page', '')} | {snip} |")
        out.append("")
    (resolve(cfg, "reports") / "probe.md").write_text("\n".join(out))
    console.kv_table({"chunks in index": total, "added this run": added})
    console.note("embedded Chroma: no server, one folder", "single-process, single-writer", "client/server vector DB (Qdrant/Weaviate) and batch embedding")


if __name__ == "__main__":
    main()
