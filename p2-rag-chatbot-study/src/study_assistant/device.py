"""get_device(): cuda > mps > cpu; dtype_for(): fp16 on cuda/mps for inference, fp32 on cpu."""


def get_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def dtype_name(device: str) -> str:
    return "float16" if device in ("cuda", "mps") else "float32"
