"""CLIP zero-shot classifier with prompt ensembling and a cached text-embedding matrix."""
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
from PIL import Image

from .config import ROOT, load_config
from .device import empty_cache, get_device

TEMPLATES = [
    "a photo of {}, a type of food.",
    "a close-up photo of {}.",
    "a photo of a plate of {}.",
    "a homemade dish of {}.",
    "a restaurant photo of {}.",
]


def pretty(label: str) -> str:
    return label.replace("_", " ")


class ClipClassifier:
    def __init__(self, labels: list[str], cfg: dict | None = None, device: str | None = None):
        import open_clip
        import torch
        from huggingface_hub import snapshot_download

        self.cfg = cfg or load_config()
        m = self.cfg["models"]["clip"]
        self.device = device or get_device()
        self.labels = labels
        path = snapshot_download(m["repo"], revision=m["revision"], allow_patterns=[m["weights"]])
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            m["arch"], pretrained=str(Path(path) / m["weights"])
        )
        self.model.eval().to(self.device)
        self.tokenizer = open_clip.get_tokenizer(m["arch"])
        self._torch = torch
        self.text_emb = self._load_or_build_text()

    def _cache_path(self) -> Path:
        return ROOT / self.cfg["paths"]["artifacts"] / "clip_text_emb.npy"

    def _load_or_build_text(self) -> np.ndarray:
        p = self._cache_path()
        if p.exists():
            arr = np.load(p)
            if arr.shape[0] == len(self.labels):
                return arr
        torch = self._torch
        embs = []
        with torch.inference_mode():
            for lab in self.labels:
                toks = self.tokenizer([t.format(pretty(lab)) for t in TEMPLATES]).to(self.device)
                e = self.model.encode_text(toks).float()
                e = e / e.norm(dim=-1, keepdim=True)
                e = e.mean(0)  # prompt ensembling: average then renormalise
                embs.append((e / e.norm()).cpu().numpy())
        arr = np.stack(embs).astype("float32")
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, arr)
        return arr

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        torch = self._torch
        x = torch.stack([self.preprocess(im.convert("RGB")) for im in images]).to(self.device)
        with torch.inference_mode():
            e = self.model.encode_image(x).float()
            e = e / e.norm(dim=-1, keepdim=True)
        return e.cpu().numpy()

    def topk(self, images: list[Image.Image], k: int = 5) -> list[list[tuple[str, float]]]:
        logits = 100.0 * self.encode_images(images) @ self.text_emb.T
        logits -= logits.max(1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(1, keepdims=True)
        out = []
        for row in probs:
            idx = np.argsort(-row)[:k]
            out.append([(self.labels[i], float(row[i])) for i in idx])
        return out

    def close(self) -> None:
        """Free the model (small profile: before loading the VLM)."""
        del self.model
        gc.collect()
        empty_cache(self.device)
