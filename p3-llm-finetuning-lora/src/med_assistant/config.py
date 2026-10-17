"""YAML config + profile selection (env P3_PROFILE overrides)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "config.yaml"


def load_config(path: Path | None = None) -> dict[str, Any]:
    with open(path or CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["profile"] = os.environ.get("P3_PROFILE", cfg["profile"])
    cfg["sizes"] = cfg["profiles"][cfg["profile"]]
    return cfg


def resolve(cfg: dict[str, Any], key: str) -> Path:
    return ROOT / cfg["paths"][key]


def sqlite_uri(db: Path) -> str:
    """Absolute sqlite URI that is valid on Windows (C:\\x -> sqlite:///C:/x) and POSIX (/x -> sqlite:////x)."""
    return "sqlite:///" + Path(db).resolve().as_posix()
