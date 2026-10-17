"""YAML config; ROOT-relative paths."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path | None = None) -> dict[str, Any]:
    with open(path or ROOT / "configs" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(cfg: dict[str, Any], key: str) -> Path:
    return ROOT / cfg["paths"][key]
