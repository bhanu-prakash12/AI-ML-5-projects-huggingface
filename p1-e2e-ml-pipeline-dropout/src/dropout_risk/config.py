"""Load YAML config + env overrides (pydantic-settings)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Env(BaseSettings):
    """Environment overrides (prefix P1_)."""

    model_config = SettingsConfigDict(env_prefix="P1_", env_file=ROOT / ".env", extra="ignore")
    profile: str = "small"
    config_path: Path = ROOT / "configs" / "config.yaml"


@lru_cache
def load_config() -> dict[str, Any]:
    env = Env()
    cfg = yaml.safe_load(env.config_path.read_text())
    cfg["profile"] = env.profile
    cfg["runtime"] = cfg["profiles"][env.profile]
    return cfg


def project_path(key: str) -> Path:
    """Resolve a configured `paths.<key>` directory against the project root."""
    p = ROOT / load_config()["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p
