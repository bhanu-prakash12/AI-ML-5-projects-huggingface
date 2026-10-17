"""Phase 2: base model on test set (greedy, max_new_tokens 200) + 10 fixed prompts -> reports/baseline.json."""
import argparse
import json
import time

import yaml

from med_assistant import console
from med_assistant.config import ROOT, load_config, resolve
from med_assistant.data import build_messages, read_jsonl
from med_assistant.llm import generate_batch, load
from med_assistant.metrics import avg_words, rouge_l



def main() -> None:
    argparse.ArgumentParser(
        description="Run the BASE model (no adapter) on the held-out test set + 10 fixed medical prompts.",
        epilog="Inputs: data/test.jsonl, configs/prompts.yaml. Output: reports/baseline.json (ROUGE-L, avg words, answers).\n"
               "Sample: python run.py baseline", formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    console.setup_logging()
    console.banner("BASELINE: base model, no fine-tuning",
                   "Greedy answers from Qwen2.5-0.5B-Instruct on the test questions; ROUGE-L vs the MedQuAD reference answer.",
                   "reports/baseline.json; ROUGE-L is the number the tuned model must beat.")
    cfg = load_config()
    console.step(1, 4, "reading test split and prompts")
    test = read_jsonl(resolve(cfg, "data") / "test.jsonl")
    prompts = yaml.safe_load(open(ROOT / "configs" / "prompts.yaml", encoding="utf-8"))["domain"]
    console.step(2, 4, "loading base model (first load can take ~1 min; downloads once, then cached)")
    tok, model, dev = load(cfg)
    console.ok(f"model on device: {dev}")
    console.step(3, 4, f"generating {len(test)} test answers + {len(prompts)} qualitative answers (slow on CPU)")
    n = cfg["baseline"]["max_new_tokens"]
    t0 = time.time()
    preds = generate_batch(tok, model, dev, [build_messages(cfg["system_prompt"], r["question"]) for r in test], n)
    qual = generate_batch(tok, model, dev, [build_messages(cfg["system_prompt"], p) for p in prompts], n)
    res = {"model": cfg["model"]["name"], "n_test": len(test), "max_new_tokens": n,
           "rougeL": rouge_l(preds, [r["answer"] for r in test]), "avg_words": avg_words(preds),
           "ref_avg_words": avg_words([r["answer"] for r in test]), "seconds": round(time.time() - t0, 1),
           "test_preds": preds, "qualitative": [{"prompt": p, "answer": a} for p, a in zip(prompts, qual, strict=True)]}
    console.step(4, 4, "writing reports/baseline.json")
    (ROOT / "reports" / "baseline.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    console.kv_table({k: v for k, v in res.items() if k not in ("test_preds", "qualitative")}, "Baseline result")
    console.note("one number (ROUGE-L) to compare against; cheap, no training needed",
                 "ROUGE-L measures overlap with one reference text, not medical correctness",
                 "use a larger test set and an LLM judge or clinician review before trusting a score")


if __name__ == "__main__":
    main()
