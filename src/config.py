from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load the dependency-free JSON subset of YAML used by frozen configs."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Config must be an object: {config_path}")
    version = value.get("config_version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError(f"Config is missing config_version: {config_path}")
    return value
