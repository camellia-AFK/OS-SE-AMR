"""RadioML2016.10a dataset loader and PyTorch Dataset wrapper."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from os_se_amr.utils.config import resolve_path


def load_rml2016_dict(path: str | Path) -> dict[tuple[str, int], np.ndarray]:
    """Load RadioML2016.10a pickle dict and normalize array layouts to float32."""
    resolved = resolve_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset file not found: {resolved}")

    with resolved.open("rb") as f:
        raw = pickle.load(f, encoding="latin1")

    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Expected non-empty dict in {resolved}, got {type(raw)}")

    cleaned: dict[tuple[str, int], np.ndarray] = {}
    for key, arr in raw.items():
        mod, snr = key
        arr_f32 = np.ascontiguousarray(arr, dtype=np.float32)
        if arr_f32.ndim != 3 or arr_f32.shape[1:] != (2, 128):
            raise ValueError(
                f"Unexpected sample tensor shape for key {key}: {arr_f32.shape}; expected (N, 2, 128)"
            )
        cleaned[(str(mod), int(snr))] = arr_f32

    return cleaned


class RMLDataset(Dataset):
    """In-memory PyTorch Dataset holding raw I/Q waveforms, class labels, and SNRs."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        snrs: np.ndarray,
        class_names: list[str] | None = None,
    ) -> None:
        if not (len(x) == len(y) == len(snrs)):
            raise ValueError(f"Length mismatch: x={len(x)}, y={len(y)}, snrs={len(snrs)}")
        self.x = np.ascontiguousarray(x, dtype=np.float32)
        self.y = np.ascontiguousarray(y, dtype=np.int64)
        self.snrs = np.ascontiguousarray(snrs, dtype=np.int64)
        self.class_names = list(class_names) if class_names is not None else []

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        return torch.from_numpy(self.x[idx]), int(self.y[idx]), int(self.snrs[idx])
