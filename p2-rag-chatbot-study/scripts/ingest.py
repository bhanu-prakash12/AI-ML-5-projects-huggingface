"""Chunk the corpus and write reports/ingest.md (no embedding; `run.py index` embeds into Chroma).

Inputs: data/corpus (from `run.py data`). Outputs: reports/ingest.md, data/chunk_ids.json.
Sample: python scripts/ingest.py
"""
import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from study_assistant import console  # noqa: E402
from study_assistant.config import load_config, resolve  # noqa: E402
from study_assistant.ingest import ingest_corpus  # noqa: E402


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    console.setup_logging(logging.WARNING)
    console.banner("Ingest", "Split notes and exam PDFs into content-hashed chunks.", "~2.7K chunks; a re-run adds 0 new IDs")
    cfg = load_config()
    ch = cfg["chunking"]["notes"]
    console.step(1, 3, f"chunking corpus (size {ch['size']}, overlap {ch['overlap']})")
    chunks, stats = ingest_corpus(resolve(cfg, "corpus"), ch["size"], ch["overlap"])
    ids_file = resolve(cfg, "data") / "chunk_ids.json"
    prev = set(json.loads(ids_file.read_text())) if ids_file.exists() else set()
    now = {c.id for c in chunks}
    added = len(now - prev)
    ids_file.write_text(json.dumps(sorted(now)))
    lens = sorted(len(c.text) for c in chunks)
    by_src = Counter(c.metadata["source"] for c in chunks)
    out = ["# Ingest report", "", f"- total chunks: {stats['total_chunks']}",
           f"- notes docs: {stats['notes_docs']}, exam docs: {stats['exam_docs']}",
           f"- exam question chunks: {stats['question_chunks']}, fallback chunks: {stats['fallback_chunks']}, "
           f"fallback rate: {stats['fallback_rate']}",
           f"- duplicates dropped in-run: {stats['duplicates_dropped']}",
           f"- new chunk IDs vs previous run: {added} (previous run had {len(prev)})",
           f"- chunk length chars: min {lens[0]}, median {lens[len(lens)//2]}, max {lens[-1]}" if lens else "- no chunks",
           "", "| source | chunks |", "|---|---|"]
    out += [f"| {s} | {n} |" for s, n in sorted(by_src.items())]
    (resolve(cfg, "reports") / "ingest.md").write_text("\n".join(out) + "\n")
    console.step(2, 3, "result")
    console.kv_table({"total chunks": stats["total_chunks"], "new IDs vs previous run": added,
                      "notes / exam docs": f"{stats['notes_docs']} / {stats['exam_docs']}",
                      "exam fallback rate": stats["fallback_rate"]})
    console.step(3, 3, "wrote reports/ingest.md")
    console.note("content-hash IDs make re-ingest idempotent", "exam PDFs often fall back to plain recursive splitting",
                 "ingestion moves to an async queue with incremental updates")


if __name__ == "__main__":
    main()
