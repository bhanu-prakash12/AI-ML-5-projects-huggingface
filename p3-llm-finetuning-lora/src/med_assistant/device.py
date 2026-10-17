"""get_device(): cuda > mps > cpu; dtype by device."""


def get_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    return "mps" if mps is not None and mps.is_available() else "cpu"


def inference_dtype(dev: str):
    """fp16 on CUDA; fp32 on MPS/CPU (MPS fp32 kept so the reported numbers stay reproducible)."""
    import torch

    return torch.float16 if dev == "cuda" else torch.float32
