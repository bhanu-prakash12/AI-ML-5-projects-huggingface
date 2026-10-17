"""Model loading (base, optionally + LoRA adapter) and batched greedy generation. fp32 on MPS/CPU, fp16 on CUDA."""
from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from med_assistant.config import load_config
from med_assistant.device import get_device, inference_dtype

log = logging.getLogger(__name__)


def load(cfg: dict[str, Any] | None = None, adapter: Path | str | None = None):
    """Returns (tokenizer, model, device). Adapter, if given, is attached to the base (PeftModel)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = cfg or load_config()
    m = cfg["model"]
    dev = get_device()
    tok = AutoTokenizer.from_pretrained(m["name"], revision=m["revision"], padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(m["name"], revision=m["revision"], torch_dtype=inference_dtype(dev))
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
    model.to(dev).eval()
    log.info("loaded %s adapter=%s on %s", m["name"], bool(adapter), dev)
    return tok, model, dev


def generate_batch(tok, model, dev: str, msgs_list: list[list[dict[str, str]]],
                   max_new_tokens: int = 200, batch_size: int = 10) -> list[str]:
    """Greedy decoding (deterministic); left-padded batches."""
    import torch

    outs: list[str] = []
    for i in range(0, len(msgs_list), batch_size):
        texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                 for m in msgs_list[i:i + batch_size]]
        inp = tok(texts, return_tensors="pt", padding=True).to(dev)
        with torch.inference_mode():
            o = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                               top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
        outs += [tok.decode(x[inp["input_ids"].shape[1]:], skip_special_tokens=True).strip() for x in o]
    return outs


@contextlib.contextmanager
def base_mode(model, base: bool) -> Iterator[None]:
    """Run the block as the base model (adapter disabled) or the tuned model."""
    if base and hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            yield
    else:
        yield
