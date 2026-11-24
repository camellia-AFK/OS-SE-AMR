"""Rejection sample buffer for storing intercepted unknown signal candidates."""

from __future__ import annotations

import numpy as np


class RejectionBuffer:
    """In-memory buffer holding intercepted raw I/Q waveforms and optional metadata."""

    def __init__(self) -> None:
        self._waveforms: list[np.ndarray] = []
        self._snr_ests: list[np.ndarray] = []
        self._true_labels: list[np.ndarray] = []

    @property
    def size(self) -> int:
        return sum(len(x) for x in self._waveforms)

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        x_iq: np.ndarray,
        snr_est: np.ndarray | None = None,
        true_labels: np.ndarray | None = None,
    ) -> None:
        arr = np.ascontiguousarray(x_iq, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[1] != 2 or arr.shape[2] <= 0:
            raise ValueError(f"Expected (N, 2, L) waveforms, got {arr.shape}")
        if len(arr) == 0:
            return
        self._waveforms.append(arr)
        if snr_est is not None:
            self._snr_ests.append(np.asarray(snr_est, dtype=np.float32).reshape(-1))
        if true_labels is not None:
            self._true_labels.append(np.asarray(true_labels, dtype=np.int64).reshape(-1))

    def get_waveforms(self) -> np.ndarray:
        if not self._waveforms:
            return np.empty((0, 2, 128), dtype=np.float32)
        return np.concatenate(self._waveforms, axis=0)

    def get_snr_ests(self) -> np.ndarray | None:
        if not self._snr_ests:
            return None
        return np.concatenate(self._snr_ests, axis=0)

    def get_true_labels(self) -> np.ndarray | None:
        if not self._true_labels:
            return None
        return np.concatenate(self._true_labels, axis=0)

    def clear(self) -> None:
        self._waveforms.clear()
        self._snr_ests.clear()
        self._true_labels.clear()
