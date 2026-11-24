"""Model checkpoint saving and loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    metrics: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save model state, optimizer state, epoch, and evaluation metrics."""
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "metrics": metrics,
        "extra": extra or {},
    }
    torch.save(payload, ckpt_path)
    return ckpt_path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load model weights from checkpoint and return full checkpoint metadata."""
    ckpt = torch.load(Path(path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=strict)
    return ckpt
