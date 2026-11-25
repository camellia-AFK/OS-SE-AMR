"""Static Open-Set Recognition (OSR) metrics: AUROC, FPR@TPR95, and OSCR.

Global scoring polarity invariant:
- Higher score => more likely Known (positive class = 1)
- Lower score  => more likely Unknown (negative class = 0)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def _to_1d_float(arr: np.ndarray | list[float], name: str) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64).reshape(-1)
    if out.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains NaN or Inf values")
    return out


def compute_auroc(known_scores: np.ndarray, unknown_scores: np.ndarray) -> float:
    """Compute AUROC separating Known (1) from Unknown (0) using normality scores."""
    k = _to_1d_float(known_scores, "known_scores")
    u = _to_1d_float(unknown_scores, "unknown_scores")
    y_true = np.concatenate([np.ones(len(k), dtype=np.int64), np.zeros(len(u), dtype=np.int64)])
    y_score = np.concatenate([k, u])
    return float(roc_auc_score(y_true, y_score))


def compute_fpr_at_tpr(
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
    target_tpr: float = 0.95,
) -> tuple[float, float]:
    """Compute Unknown False Positive Rate at target Known TPR via linear ROC interpolation.

    Returns:
        (fpr_at_target_tpr, interpolated_threshold)
    """
    if not (0.0 < target_tpr <= 1.0):
        raise ValueError(f"target_tpr must be in (0, 1], got {target_tpr}")

    k = _to_1d_float(known_scores, "known_scores")
    u = _to_1d_float(unknown_scores, "unknown_scores")
    y_true = np.concatenate([np.ones(len(k), dtype=np.int64), np.zeros(len(u), dtype=np.int64)])
    y_score = np.concatenate([k, u])

    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)

    idx = int(np.searchsorted(tpr, target_tpr, side="left"))
    if idx >= len(tpr):
        return float(fpr[-1]), float(thresholds[-1])
    if idx == 0 or tpr[idx] == target_tpr or tpr[idx] == tpr[idx - 1]:
        thr = float(thresholds[idx]) if np.isfinite(thresholds[idx]) else float(y_score.max())
        return float(fpr[idx]), thr

    alpha = (target_tpr - tpr[idx - 1]) / (tpr[idx] - tpr[idx - 1])
    fpr_interp = float(fpr[idx - 1] + alpha * (fpr[idx] - fpr[idx - 1]))
    thr_prev = thresholds[idx - 1] if np.isfinite(thresholds[idx - 1]) else y_score.max()
    thr_curr = thresholds[idx] if np.isfinite(thresholds[idx]) else y_score.min()
    thr_interp = float(thr_prev + alpha * (thr_curr - thr_prev))
    return fpr_interp, thr_interp


def compute_oscr(
    known_scores: np.ndarray,
    known_preds: np.ndarray,
    known_labels: np.ndarray,
    unknown_scores: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute Open-Set Classification Rate (OSCR) area and curve coordinates.

    Vectorized O(N log N) computation handling tied scores rigorously by evaluating
    cumulative CCR and FPR only at unique threshold boundaries.
    """
    k_scores = _to_1d_float(known_scores, "known_scores")
    u_scores = _to_1d_float(unknown_scores, "unknown_scores")
    k_preds = np.asarray(known_preds, dtype=np.int64).reshape(-1)
    k_labels = np.asarray(known_labels, dtype=np.int64).reshape(-1)

    n_k = len(k_scores)
    n_u = len(u_scores)
    if len(k_preds) != n_k or len(k_labels) != n_k:
        raise ValueError("known_scores, known_preds, and known_labels must share the same length")

    all_scores = np.concatenate([k_scores, u_scores])
    cc_flags = np.concatenate([(k_preds == k_labels).astype(np.float64), np.zeros(n_u, dtype=np.float64)])
    fp_flags = np.concatenate([np.zeros(n_k, dtype=np.float64), np.ones(n_u, dtype=np.float64)])

    order = np.argsort(-all_scores, kind="mergesort")
    sorted_scores = all_scores[order]
    cum_cc = np.cumsum(cc_flags[order]) / float(n_k)
    cum_fp = np.cumsum(fp_flags[order]) / float(n_u)

    # Evaluate only at the final index of each distinct score block (tie-safe)
    is_last = np.r_[sorted_scores[:-1] != sorted_scores[1:], True]
    fpr_pts = cum_fp[is_last]
    ccr_pts = cum_cc[is_last]

    fpr_curve = np.concatenate([[0.0], fpr_pts])
    ccr_curve = np.concatenate([[0.0], ccr_pts])

    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    oscr = float(_trapz(ccr_curve, fpr_curve))
    return oscr, fpr_curve, ccr_curve


def _subsample_curve(x: np.ndarray, y: np.ndarray, max_points: int = 400) -> dict[str, list[float]]:
    if len(x) <= max_points:
        return {"x": x.tolist(), "y": y.tolist()}
    idx = np.unique(np.linspace(0, len(x) - 1, num=max_points, dtype=np.int64))
    return {"x": x[idx].tolist(), "y": y[idx].tolist()}


def evaluate_osr(
    known_scores: np.ndarray,
    known_preds: np.ndarray,
    known_labels: np.ndarray,
    unknown_scores: np.ndarray,
    unknown_labels: np.ndarray | None = None,
    unknown_class_names: dict[int, str] | None = None,
    calibrated_threshold: float | None = None,
    target_tpr: float = 0.95,
) -> dict[str, Any]:
    """Run full static OSR evaluation and return summary metrics plus subsampled curves."""
    k_scores = _to_1d_float(known_scores, "known_scores")
    u_scores = _to_1d_float(unknown_scores, "unknown_scores")
    k_preds = np.asarray(known_preds, dtype=np.int64).reshape(-1)
    k_labels = np.asarray(known_labels, dtype=np.int64).reshape(-1)

    known_acc = float((k_preds == k_labels).mean())
    auroc = compute_auroc(k_scores, u_scores)
    fpr95, roc_thr95 = compute_fpr_at_tpr(k_scores, u_scores, target_tpr=target_tpr)
    oscr, oscr_fpr, oscr_ccr = compute_oscr(k_scores, k_preds, k_labels, u_scores)

    y_true = np.concatenate([np.ones(len(k_scores), dtype=np.int64), np.zeros(len(u_scores), dtype=np.int64)])
    y_score = np.concatenate([k_scores, u_scores])
    roc_fpr, roc_tpr, _ = roc_curve(y_true, y_score, pos_label=1)

    result: dict[str, Any] = {
        "known_acc": known_acc,
        "auroc": auroc,
        "fpr_at_tpr95": fpr95,
        "oscr": oscr,
        "roc_threshold_at_tpr95": roc_thr95,
        "num_known_samples": int(len(k_scores)),
        "num_unknown_samples": int(len(u_scores)),
        "curves": {
            "roc": _subsample_curve(roc_fpr, roc_tpr),
            "oscr": _subsample_curve(oscr_fpr, oscr_ccr),
        },
    }

    if calibrated_threshold is not None:
        thr = float(calibrated_threshold)
        result["calibrated_threshold"] = thr
        result["val_calibrated_tpr"] = float((k_scores >= thr).mean())
        result["val_calibrated_fpr"] = float((u_scores >= thr).mean())

    if unknown_labels is not None:
        u_labels = np.asarray(unknown_labels, dtype=np.int64).reshape(-1)
        per_unknown: dict[str, float] = {}
        for uid in sorted(np.unique(u_labels).tolist()):
            mask = u_labels == uid
            cls_key = (
                unknown_class_names[uid]
                if (unknown_class_names is not None and uid in unknown_class_names)
                else str(uid)
            )
            per_unknown[cls_key] = compute_auroc(k_scores, u_scores[mask])
        result["per_unknown_class_auroc"] = per_unknown

    return result
