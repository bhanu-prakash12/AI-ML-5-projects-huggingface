"""load(), generate(), stream() with transformers (Qwen), in-process, pinned revision."""
from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from typing import Any

from study_assistant.config import load_config
from study_assistant.device import dtype_name, get_device

log = logging.getLogger(__name__)
_STATE: dict[str, Any] = {}
_LOCK = threading.Lock()


def load(cfg: dict[str, Any] | None = None, fallback: bool = False):
    """Lazy singleton: fp16 on cuda/mps, fp32 on CPU. Returns (tokenizer, model)."""
    with _LOCK:
        if "model" not in _STATE:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            cfg = cfg or load_config()
            m = cfg["models"]["llm_fallback" if fallback else "llm"]
            dev = get_device()
            dtype = getattr(torch, dtype_name(dev))
            tok = AutoTokenizer.from_pretrained(m["name"], revision=m["revision"])
            model = AutoModelForCausalLM.from_pretrained(m["name"], revision=m["revision"], torch_dtype=dtype)
            model.to(dev).eval()
            _STATE.update(tok=tok, model=model, device=dev)
            log.info("llm loaded: %s on %s", m["name"], dev)
        return _STATE["tok"], _STATE["model"]


def count_tokens(text: str) -> int:
    tok, _ = load()
    return len(tok.encode(text))


def _inputs(messages: list[dict[str, str]]):
    tok, model = load()
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tok, model, tok(text, return_tensors="pt").to(_STATE["device"])


def _gen_kwargs(tok, max_new_tokens: int, temperature: float) -> dict[str, Any]:
    kw: dict[str, Any] = {"max_new_tokens": max_new_tokens, "pad_token_id": tok.eos_token_id}
    if temperature > 0:
        kw.update(do_sample=True, temperature=temperature)
    else:
        kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    return kw


def generate(messages: list[dict[str, str]], max_new_tokens: int = 400, temperature: float = 0.0) -> str:
    import torch

    tok, model, inp = _inputs(messages)
    with torch.inference_mode():
        out = model.generate(**inp, **_gen_kwargs(tok, max_new_tokens, temperature))
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def stream(messages: list[dict[str, str]], max_new_tokens: int = 400, temperature: float = 0.0) -> Iterator[str]:
    import torch
    from transformers import TextIteratorStreamer

    tok, model, inp = _inputs(messages)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)

    def run() -> None:
        with torch.inference_mode():
            model.generate(**inp, streamer=streamer, **_gen_kwargs(tok, max_new_tokens, temperature))

    t = threading.Thread(target=run, daemon=True)
    t.start()
    yield from streamer
    t.join()
