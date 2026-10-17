"""Phase 4: base vs tuned on test set + 10 prompts + forgetting check -> reports/compare.md/json.
One model in memory: base+adapter, base answers via disable_adapter()."""
import argparse
import json
import time

import yaml

from med_assistant import console
from med_assistant.config import ROOT, load_config, resolve
from med_assistant.data import build_messages, read_jsonl
from med_assistant.llm import base_mode, generate_batch, load
from med_assistant.metrics import avg_words, paired_bootstrap, rouge_l, rouge_l_each

EXPECT = {"What is the capital of France?": "paris", "What is 17 multiplied by 6?": "102",
          "Who wrote the play Romeo and Juliet?": "shakespeare", 'Translate "good morning" into Spanish.': "buenos"}


def cell(s: str, n: int = 260) -> str:
    s = " ".join(s.split()).replace("|", "/")
    return (s[:n] + "...") if len(s) > n else s


def main() -> None:
    argparse.ArgumentParser(
        description="Compare BASE vs LoRA-TUNED on the test set + 10 general prompts (forgetting check). One model in memory.",
        epilog="Inputs: data/test.jsonl, artifacts/adapter, reports/baseline.json. Outputs: reports/compare.md|json.\n"
               "Sample: python run.py eval   (a hand-written '## Interpretation' section in compare.md is preserved)", formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    console.setup_logging()
    console.banner("EVAL: base vs tuned",
                   "Same questions, greedy decoding, adapter on vs off; paired bootstrap on per-question ROUGE-L.",
                   "An honest result: ROUGE-L difference near zero (CI spans 0), shorter tuned answers, mild drift.")
    cfg = load_config()
    console.step(1, 4, "loading base model + adapter (first load ~1 min)")
    n = cfg["baseline"]["max_new_tokens"]
    sysm = cfg["system_prompt"]
    test = read_jsonl(resolve(cfg, "data") / "test.jsonl")
    P = yaml.safe_load(open(ROOT / "configs" / "prompts.yaml", encoding="utf-8"))
    tok, model, dev = load(cfg, resolve(cfg, "artifacts") / "adapter")
    console.step(2, 4, "generating answers with base, then tuned (slow on CPU)")
    qs = [build_messages(sysm, r["question"]) for r in test]
    refs = [r["answer"] for r in test]
    out: dict = {}
    t0 = time.time()
    for name, base in (("base", True), ("tuned", False)):
        with base_mode(model, base):
            out[name] = {"test": generate_batch(tok, model, dev, qs, n),
                         "domain": generate_batch(tok, model, dev, [build_messages(sysm, p) for p in P["domain"]], n),
                         "general": generate_batch(tok, model, dev, [build_messages(sysm, p) for p in P["general"]], n)}
    sec = time.time() - t0
    console.step(3, 4, "scoring (ROUGE-L, paired bootstrap, forgetting check)")
    each = {k: rouge_l_each(out[k]["test"], refs) for k in out}
    stats = paired_bootstrap(each["base"], each["tuned"])
    base_json = json.loads((ROOT / "reports" / "baseline.json").read_text(encoding="utf-8"))
    res = {"n_test": len(test), "rougeL": {k: rouge_l(out[k]["test"], refs) for k in out},
           "avg_words": {k: avg_words(out[k]["test"]) for k in out}, "ref_avg_words": avg_words(refs),
           "paired": stats, "baseline_json_rougeL": base_json["rougeL"], "seconds": round(sec, 1)}
    gen_sim = rouge_l(out["tuned"]["general"], out["base"]["general"])
    ok = {k: sum(EXPECT[p].lower() in a.lower() for p, a in zip(P["general"], out[k]["general"], strict=True)
                 if p in EXPECT) for k in out}
    res["forgetting"] = {"tuned_vs_base_rougeL_on_general": gen_sim, "keyword_correct_of_4": ok,
                         "avg_words": {k: avg_words(out[k]["general"]) for k in out}}
    (ROOT / "reports" / "compare.json").write_text(json.dumps({**res, "outputs": out}, indent=1, ensure_ascii=False), encoding="utf-8")
    R, W = res["rougeL"], res["avg_words"]
    md = ["# Base vs tuned (LoRA)", "", f"Test set n={len(test)}, greedy, max_new_tokens {n}. ROUGE-L is stemmed F1 against the "
          "MedQuAD reference answer: it measures overlap with one reference text, not medical correctness.", "",
          "| metric | base | tuned |", "|---|---|---|", f"| ROUGE-L | {R['base']:.3f} | {R['tuned']:.3f} |",
          f"| avg words (ref {res['ref_avg_words']:.0f}) | {W['base']:.0f} | {W['tuned']:.0f} |", "",
          f"Paired per-question diff (tuned - base): mean {stats['mean_diff']:+.3f}, 95% bootstrap CI "
          f"[{stats['ci_lo']:+.3f}, {stats['ci_hi']:+.3f}]; tuned wins {stats['wins']}, ties {stats['ties']}, "
          f"losses {stats['losses']}.", "",
          f"Sanity: baseline.json ROUGE-L {base_json['rougeL']:.3f} (separate unbatched-adapter run) vs base here {R['base']:.3f}.", "",
          "## Forgetting check (10 general prompts)", "",
          f"- ROUGE-L of tuned answers vs base answers: {gen_sim:.3f} (1.0 = unchanged)",
          f"- Keyword-correct on 4 checkable prompts (Paris, 102, Shakespeare, buenos): base {ok['base']}/4, tuned {ok['tuned']}/4",
          f"- avg words: base {res['forgetting']['avg_words']['base']:.0f}, tuned {res['forgetting']['avg_words']['tuned']:.0f}", "",
          "| prompt | base | tuned |", "|---|---|---|"]
    md += [f"| {p} | {cell(b, 160)} | {cell(t, 160)} |" for p, b, t in zip(P["general"], out["base"]["general"], out["tuned"]["general"], strict=True)]
    md += ["", "## Qualitative (10 fixed medical prompts, side by side)", "", "| prompt | base | tuned |", "|---|---|---|"]
    md += [f"| {p} | {cell(b)} | {cell(t)} |" for p, b, t in zip(P["domain"], out["base"]["domain"], out["tuned"]["domain"], strict=True)]
    cmp_md = ROOT / "reports" / "compare.md"
    kept = ""  # keep the hand-written interpretation across re-runs
    if cmp_md.exists() and "## Interpretation" in (prev := cmp_md.read_text(encoding="utf-8")):
        kept = "\n" + prev[prev.index("## Interpretation"):]
    cmp_md.write_text("\n".join(md) + "\n" + kept, encoding="utf-8")
    console.step(4, 4, "wrote reports/compare.md and compare.json")
    d = stats["mean_diff"]
    verdict = ("no measurable gain (CI spans zero)" if stats["ci_lo"] <= 0 <= stats["ci_hi"]
               else ("tuned better" if d > 0 else "tuned worse"))
    console.table([["ROUGE-L", f"{R['base']:.3f}", f"{R['tuned']:.3f}"],
                   ["avg words (ref %.0f)" % res["ref_avg_words"], f"{W['base']:.0f}", f"{W['tuned']:.0f}"],
                   ["general keyword-correct /4", ok["base"], ok["tuned"]]], ["metric", "base", "tuned"], "Base vs tuned")
    (console.ok if "no measurable" not in verdict else console.warn)(
        f"paired diff {d:+.3f} [{stats['ci_lo']:+.3f}, {stats['ci_hi']:+.3f}]: {verdict}")
    console.note("shows style adaptation (shorter, MedQuAD-like) with a proper paired test",
                 "ROUGE-L is not correctness; tuned answers can drift or reason worse (n=50 / n=10 is small)",
                 "add LLM-judge or clinician review, a bigger test set, and gate adapters on it in CI")


if __name__ == "__main__":
    main()
