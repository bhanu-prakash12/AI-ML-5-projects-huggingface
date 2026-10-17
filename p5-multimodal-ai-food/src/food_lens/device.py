"""get_device(): cuda > mps > cpu; dtype by device (fp16 on cuda/mps inference, fp32 on cpu)."""
from __future__ import annotations


def get_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def inference_dtype(device: str):
    import torch

    return torch.float16 if device in ("cuda", "mps") else torch.float32


def empty_cache(device: str) -> None:
    """Free accelerator cache; guarded so CPU-only / Windows installs never touch torch.mps."""
    import torch

    if device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
