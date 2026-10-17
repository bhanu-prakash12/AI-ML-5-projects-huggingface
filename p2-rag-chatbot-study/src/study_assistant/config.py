"""YAML config + env overrides."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "config.yaml"


def load_config(path: Path | None = None) -> dict[str, Any]:
    with open(path or CONFIG_PATH) as f:
        return yaml.safe_load(f)


def resolve(cfg: dict[str, Any], key: str) -> Path:
    """Resolve a configured path relative to the project root."""
    return ROOT / cfg["paths"][key]


def api_port(cfg: dict[str, Any] | None = None) -> int:
    """API port: P2_API_PORT env override, else config api.port."""
    return int(os.environ.get("P2_API_PORT") or (cfg or load_config())["api"]["port"])


def ui_port(cfg: dict[str, Any] | None = None) -> int:
    return int(os.environ.get("P2_UI_PORT") or (cfg or load_config())["ui"]["port"])
