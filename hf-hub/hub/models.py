"""ONE Qwen2.5-0.5B-Instruct in memory for p2 and p3 (no duplicate weights).

p3 owns the base+LoRA model (`PeftModel`). p2 gets the same object injected into its `llm._STATE` and runs it with the
adapter disabled (= the plain base model). Both revisions are asserted equal so they cannot drift apart.
"""
from __future__ import annotations

import logging
import threading

from hub import paths  # noqa: F401

log = logging.getLogger(__name__)
LOCK = threading.RLock()  # adapter toggling is global model state: one generation at a time
_S: dict = {}


def _load(cfg, adapter):
    """Same as med_assistant.llm.load, but the adapter file is read on CPU (ZeroGPU: cuda is emulated at startup)."""
    from med_assistant.device import get_device, inference_dtype
    from transformers import AutoModelForCausalLM, AutoTokenizer

    m = cfg["model"]
    dev = get_device()
    tok = AutoTokenizer.from_pretrained(m["name"], revision=m["revision"], padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(m["name"], revision=m["revision"], torch_dtype=inference_dtype(dev))
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter), torch_device="cpu")
    return tok, model.to(dev).eval(), dev


def shared_llm():
    """Returns (p3_cfg, tokenizer, model, device, has_adapter). Lazy singleton."""
    with LOCK:
        if "llm" not in _S:
            from med_assistant.config import load_config, resolve
            from study_assistant import llm as p2_llm
            from study_assistant.config import load_config as p2_cfg

            cfg = load_config()
            fb = p2_cfg()["models"]["llm_fallback"]
            assert (fb["name"], fb["revision"]) == (cfg["model"]["name"], cfg["model"]["revision"]), \
                "p2 fallback LLM and p3 base must be the same pinned model to share weights"
            adapter = resolve(cfg, "artifacts") / "adapter"
            has = adapter.exists()
            tok, model, dev = _load(cfg, adapter if has else None)
            _S["llm"] = (cfg, tok, model, dev, has)
            p2_llm._STATE.update(tok=tok, model=model, device=dev)  # p2's lazy singleton now finds it
            log.info("shared LLM loaded on %s (adapter=%s)", dev, has)
        return _S["llm"]


_P2: dict = {}


def p2_store():
    """Embeddings (bge-small) + Chroma store for p2, loaded once."""
    with LOCK:
        if "store" not in _P2:
            from study_assistant.config import load_config
            from study_assistant.index import get_embeddings, get_store

            cfg = load_config()
            emb = get_embeddings(cfg)
            _P2.update(cfg=cfg, store=get_store(cfg, emb))
        return _P2["cfg"], _P2["store"]
