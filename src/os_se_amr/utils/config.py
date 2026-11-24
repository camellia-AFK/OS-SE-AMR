"""YAML configuration loading, path resolution, and serialization."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to PROJECT_ROOT if not already absolute."""
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""
    resolved = resolve_path(path)
    with resolved.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {} if data is None else dict(data)


def save_yaml(cfg: dict[str, Any], path: str | Path) -> Path:
    """Save configuration dictionary to YAML, creating parent directories."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return out_path


def load_stage_config(stage_yaml: str | Path) -> dict[str, Any]:
    """Load a stage YAML config and merge referenced split_config into cfg['split']."""
    cfg = copy.deepcopy(load_yaml(stage_yaml))
    if "split_config" in cfg and cfg["split_config"]:
        cfg["split"] = load_yaml(cfg["split_config"])
    return cfg
