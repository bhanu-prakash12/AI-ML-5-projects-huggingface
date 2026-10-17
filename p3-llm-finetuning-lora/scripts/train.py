"""Phase 3: LoRA SFT with TRL on the train split; MLflow tracking; resume; ETA gate.

usage: train.py [--yes]   (--yes continues even when ETA > eta_ask_minutes)
"""
import argparse
import json
import os
import sys
import time

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

from med_assistant import console
from med_assistant.config import ROOT, load_config, resolve, sqlite_uri
from med_assistant.data import read_jsonl
from med_assistant.train_utils import eta_minutes, loss_trend

RESPONSE_TEMPLATE = "<|im_start|>assistant\n"


class EtaGate(TrainerCallback):
    def __init__(self, limit_min: float, yes: bool) -> None:
        self.limit, self.yes, self.t0, self.aborted = limit_min, yes, time.time(), False

    def on_step_end(self, args, state, control, **kw):
        if state.global_step == 1:
            eta = eta_minutes(time.time() - self.t0, state.max_steps, 1)
            console.warn(f"ETA: ~{eta:.1f} min for {state.max_steps} steps (limit {self.limit} min)")
            if eta > self.limit and not self.yes:
                console.fail("ETA exceeds limit: rerun with --yes to continue")
                self.aborted, control.should_training_stop = True, True
        return control


def peak_ram_gb() -> float | None:
    """Peak RSS in GB where the stdlib can tell (macOS/Linux); None on Windows."""
    try:
        import resource
    except ImportError:
        return None
    div = 1e9 if sys.platform == "darwin" else 1e6  # ru_maxrss is bytes on macOS, KB on Linux
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / div, 2)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="LoRA supervised fine-tuning of Qwen2.5-0.5B-Instruct on the MedQuAD train split (MLflow tracked).",
        epilog="Inputs: data/train.jsonl, data/val.jsonl. Outputs: artifacts/adapter, reports/train.md|json, mlflow.db.\n"
               "Sample: python run.py train   (the adapter already exists; retraining overwrites it)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yes", action="store_true", help="continue even when the ETA exceeds the limit (needed on slow CPUs)")
    yes = ap.parse_args().yes
    console.setup_logging()
    console.banner("TRAIN: LoRA adapter",
                   "Freeze the 0.5B base model and learn small low-rank (r=8) matrices on 300 medical Q&A pairs.",
                   "ETA after step 1, falling train/val loss, adapter saved to artifacts/adapter (~19 MB).")
    cfg = load_config()
    t, lo, m = cfg["train"], cfg["lora"], cfg["model"]
    art = resolve(cfg, "artifacts")
    ck = art / "checkpoints"
    os.environ.setdefault("MLFLOW_TRACKING_URI", sqlite_uri(ROOT / "mlflow.db"))
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", cfg["mlflow"]["experiment"])
    console.step(1, 4, "loading base model in fp32 (training is fp32 on every device)")
    tok = AutoTokenizer.from_pretrained(m["name"], revision=m["revision"])
    model = AutoModelForCausalLM.from_pretrained(m["name"], revision=m["revision"], torch_dtype=torch.float32)
    from datasets import Dataset

    data = {k: Dataset.from_list([{"text": r["text"]} for r in read_jsonl(resolve(cfg, "data") / f"{k}.jsonl")])
            for k in ("train", "val")}
    sft = SFTConfig(
        output_dir=str(ck), num_train_epochs=t["epochs"], per_device_train_batch_size=t["batch"],
        per_device_eval_batch_size=2, gradient_accumulation_steps=t["grad_accum"], learning_rate=t["lr"],
        lr_scheduler_type="cosine", warmup_ratio=0.05, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, bf16=False, fp16=False, seed=cfg["seed"],
        logging_steps=1, save_steps=t["save_steps"], save_total_limit=2, eval_strategy="steps",
        eval_steps=t["save_steps"], report_to="mlflow", run_name=f"lora-r{lo['r']}-{cfg['profile']}",
        max_seq_length=t["max_seq_len"], dataset_text_field="text", packing=False,
    )
    peft_cfg = LoraConfig(r=lo["r"], lora_alpha=lo["alpha"], lora_dropout=lo["dropout"],
                          target_modules=lo["targets"], task_type="CAUSAL_LM")
    gate = EtaGate(t["eta_ask_minutes"], yes)
    trainer = SFTTrainer(model=model, args=sft, train_dataset=data["train"], eval_dataset=data["val"],
                         tokenizer=tok, peft_config=peft_cfg, callbacks=[gate],
                         data_collator=DataCollatorForCompletionOnlyLM(RESPONSE_TEMPLATE, tokenizer=tok))
    trainable, total = trainer.model.get_nb_trainable_parameters()
    console.ok(f"trainable params {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    console.step(2, 4, "evaluating before training, then training (ETA printed after step 1; CPU-only machines are much slower)")
    resume = ck.exists() and any(ck.glob("checkpoint-*"))
    eval0 = None if resume else trainer.evaluate()["eval_loss"]
    t0 = time.time()
    trainer.train(resume_from_checkpoint=True if resume else None)
    if gate.aborted:
        sys.exit(3)
    secs = time.time() - t0
    console.step(3, 4, "final evaluation and saving adapter")
    final = trainer.evaluate()["eval_loss"]
    trainer.model.save_pretrained(art / "adapter")
    tok.save_pretrained(art / "adapter")
    hist = trainer.state.log_history
    losses = [h["loss"] for h in hist if "loss" in h]
    evals = [{"step": h["step"], "eval_loss": h["eval_loss"]} for h in hist if "eval_loss" in h]
    trend = loss_trend(losses)
    rep = {"train_seconds": round(secs, 1), "steps": trainer.state.global_step, "trainable_params": trainable,
           "total_params": total, "eval_loss_before": eval0, "eval_loss_after": final, "loss_trend": trend,
           "losses": losses, "evals": evals, "resumed": resume,
           "peak_ram_gb": peak_ram_gb()}
    (ROOT / "reports" / "train.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    md = ["# Training report", "", f"- LoRA r={lo['r']} alpha={lo['alpha']} on {lo['targets']}; trainable {trainable:,} of {total:,} "
          f"({100 * trainable / total:.2f}%)", f"- lr {t['lr']}, {t['epochs']} epoch, batch {t['batch']} x accum {t['grad_accum']}, fp32, "
          f"grad checkpointing; steps {rep['steps']}; wall {secs / 60:.1f} min (this run); peak RAM {rep['peak_ram_gb'] if rep['peak_ram_gb'] is not None else 'n/a (Windows)'} GB",
          f"- Train loss: first-5 mean {trend['first']:.3f} -> last-5 mean {trend['last']:.3f} (falling: {trend['falling']})",
          f"- Val loss: before {eval0} -> after {final:.3f}", "", "| step | eval_loss |", "|---|---|"]
    md += [f"| {e['step']} | {e['eval_loss']:.3f} |" for e in evals]
    md += ["", "Train loss per step: " + ", ".join(f"{x:.2f}" for x in losses)]
    (ROOT / "reports" / "train.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    console.step(4, 4, "done: reports/train.md written; view curves with `python run.py mlflow-ui` (http://127.0.0.1:5003)")
    console.kv_table({k: rep[k] for k in ("steps", "eval_loss_before", "eval_loss_after", "loss_trend")}, "Training result")
    console.note("adapter is tiny (~19 MB), trains on a laptop, base weights untouched",
                 "300 examples / 37 steps is a demo; loss falling does not mean answers got better (see eval)",
                 "cloud GPU + more data (QLoRA/LoRA), eval gate in CI before shipping an adapter")


if __name__ == "__main__":
    main()
