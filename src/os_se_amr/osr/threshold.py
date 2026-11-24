"""OSR decision threshold calibration using Validation known scores."""

from __future__ import annotations

import numpy as np


def calibrate_threshold(val_known_scores: np.ndarray, target_tpr: float = 0.95) -> float:
    """Calibrate rejection threshold tau such that P(score >= tau | Val Known) ~= target_tpr.

    Strict anti-leakage contract: pass ONLY Session 0 Validation known-class scores.
    """
    scores = np.asarray(val_known_scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        raise ValueError("val_known_scores must not be empty")
    if not (0.0 < target_tpr < 1.0):
        raise ValueError(f"target_tpr must be in (0, 1), got {target_tpr}")
    if not np.all(np.isfinite(scores)):
        raise ValueError("val_known_scores contains NaN or Inf")
    return float(np.quantile(scores, 1.0 - target_tpr))
