"""Phase 1: build splits in data/*.jsonl and reports/data.md."""
import argparse
import statistics as st

import pandas as pd
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from med_assistant import console
from med_assistant.config import ROOT, load_config, resolve
from med_assistant.data import clean_and_dedupe, select_and_split, write_jsonl



def main() -> None:
    argparse.ArgumentParser(
        description="Build deterministic train/val/test splits from MedQuAD (clean, dedupe, <=384 tokens, chat template).",
        epilog="Input: pinned HF dataset (cached after first download). Outputs: data/*.jsonl, reports/data.md.\n"
               "Sample: python run.py data", formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    console.setup_logging()
    console.banner("DATA: build splits", "Download (cached) the pinned MedQuAD CSV, clean, dedupe, keep rows that fit 384 tokens, split 300/50/50.",
                   "data/train|val|test.jsonl and reports/data.md; rerun gives identical files (seed 42).")
    cfg = load_config()
    console.step(1, 3, "loading dataset (first run downloads ~22 MB)")
    ds, m = cfg["dataset"], cfg["model"]
    raw = pd.read_csv(hf_hub_download(ds["name"], ds["file"], repo_type="dataset", revision=ds["revision"]))
    console.step(2, 3, "cleaning, deduping, token-length filtering")
    clean = clean_and_dedupe(raw, ds["min_answer_chars"])
    tok = AutoTokenizer.from_pretrained(m["name"], revision=m["revision"])
    tpl = lambda msgs: tok.apply_chat_template(msgs, tokenize=False)  # noqa: E731
    splits = select_and_split(clean, cfg["system_prompt"], tpl, lambda t: len(tok.encode(t)),
                              ds["max_tokens"], cfg["sizes"], cfg["seed"])
    for k, rows in splits.items():
        write_jsonl(rows, resolve(cfg, "data") / f"{k}.jsonl")
    L = {k: [r["n_tokens"] for r in v] for k, v in splits.items()}
    out = [f"# Data report ({cfg['profile']} profile)", "",
           f"- Source: {ds['name']} @ {ds['revision'][:7]} (MedQuAD; license note in DECISIONS.md)",
           f"- Raw rows {len(raw)}, after clean+dedupe {len(clean)}",
           f"- Filter: full chat text <= {ds['max_tokens']} tokens (Qwen tokenizer), seed {cfg['seed']}", "",
           "| split | rows | tokens min | median | mean | max |", "|---|---|---|---|---|---|"]
    for k, v in L.items():
        out.append(f"| {k} | {len(v)} | {min(v)} | {st.median(v):.0f} | {st.mean(v):.0f} | {max(v)} |")
    out += ["", "## Samples (train)", ""]
    for r in splits["train"][:3]:
        out += [f"**Q ({r['qtype']}):** {r['question']}", "", f"**A:** {r['answer'][:300]}...", ""]
    out += ["## Rendered chat template (first train row)", "", "```", splits["train"][0]["text"][:700], "```"]
    (ROOT / "reports" / "data.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    console.step(3, 3, "wrote data/*.jsonl and reports/data.md")
    console.table([[k, len(v)] for k, v in splits.items()], ["split", "rows"], "Splits")
    console.note("deterministic, leak-free splits; test set never used for training",
                 "length filter biases the sample toward shorter answers",
                 "use the standard profile (2000/200/100) or the full dataset with a streaming loader")


if __name__ == "__main__":
    main()
