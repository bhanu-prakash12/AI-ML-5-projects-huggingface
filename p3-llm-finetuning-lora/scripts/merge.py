"""Save a merged (base + adapter) copy to artifacts/merged; check it reproduces the adapter's greedy output."""
from med_assistant import console
from med_assistant.config import load_config, resolve
from med_assistant.data import build_messages
from med_assistant.llm import generate_batch, load

console.banner("MERGE", "Fold the LoRA adapter into the base weights and save artifacts/merged (~1.9 GB, optional).", "greedy output identical to the adapter model: True")
cfg = load_config()
art = resolve(cfg, "artifacts")
tok, model, dev = load(cfg, art / "adapter")
msgs = [build_messages(cfg["system_prompt"], "What are the symptoms of type 2 diabetes?")]
before = generate_batch(tok, model, dev, msgs, 60, 1)[0]
merged = model.merge_and_unload()
merged.save_pretrained(art / "merged")
tok.save_pretrained(art / "merged")
after = generate_batch(tok, merged, dev, msgs, 60, 1)[0]
(console.ok if before == after else console.fail)(f"merged saved; greedy output identical to adapter model: {before == after}")
