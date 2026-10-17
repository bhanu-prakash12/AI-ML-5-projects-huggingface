"""Config loader: YAML + env override (P4_PROFILE)."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def load() -> dict[str, Any]:
    cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
    profile = os.environ.get("P4_PROFILE", cfg["profile"])
    cfg = {**cfg, **cfg["profiles"][profile], "profile": profile}
    return cfg


def path(key: str) -> Path:
    if key == "db" and os.environ.get("P4_DB"):
        return Path(os.environ["P4_DB"])
    p = ROOT / load()["paths"][key]
    p.parent.mkdir(parents=True, exist_ok=True) if "." in p.name else p.mkdir(parents=True, exist_ok=True)
    return p
