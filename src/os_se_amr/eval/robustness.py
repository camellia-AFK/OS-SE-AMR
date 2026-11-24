"""Cross-SNR environment drift evaluation for OSR boundaries and continual learning (DOCO protocol)."""

from __future__ import annotations

from typing import Any

import numpy as np

from os_se_amr.eval.cil_metrics import evaluate_cil_session
from os_se_amr.eval.osr_metrics import compute_auroc, compute_fpr_at_tpr, compute_oscr

DEFAULT_SNR_REGIMES: dict[str, list[int]] = {
    "high_snr": [10, 12, 14, 16, 18],
    "mid_low_snr": [0, 2, 4, 6],
    "neg_snr": [-10, -8, -6, -4, -2],
}


def _compute_osr_slice(
    k_scores: np.ndarray,
    k_preds: np.ndarray,
    k_labels: np.ndarray,
    u_scores: np.ndarray,
    calibrated_threshold: float,
    target_tpr: float,
) -> dict[str, Any]:
    known_acc = float((k_preds == k_labels).mean())
    known_tpr = float((k_scores >= calibrated_threshold).mean())
    known_frr = 1.0 - known_tpr
    unknown_fpr = float((u_scores >= calibrated_threshold).mean())
    auroc = compute_auroc(k_scores, u_scores)
    fpr95, _ = compute_fpr_at_tpr(k_scores, u_scores, target_tpr=target_tpr)
    oscr, _, _ = compute_oscr(k_scores, k_preds, k_labels, u_scores)
    return {
        "n_known": int(len(k_scores)),
        "n_unknown": int(len(u_scores)),
        "known_acc_pct": round(known_acc * 100.0, 2),
        "known_tpr_pct": round(known_tpr * 100.0, 2),
        "known_frr_pct": round(known_frr * 100.0, 2),
        "unknown_fpr_pct": round(unknown_fpr * 100.0, 2),
        "auroc_pct": round(auroc * 100.0, 2),
        "fpr_at_tpr95_pct": round(fpr95 * 100.0, 2),
        "oscr_pct": round(oscr * 100.0, 2),
    }


def evaluate_snr_osr_drift(
    known_scores: np.ndarray,
    known_preds: np.ndarray,
    known_labels: np.ndarray,
    known_snrs: np.ndarray,
    unknown_scores: np.ndarray,
    unknown_labels: np.ndarray,
    unknown_snrs: np.ndarray,
    calibrated_threshold: float,
    regimes: dict[str, list[int]] | None = None,
    target_tpr: float = 0.95,
) -> dict[str, Any]:
    """Evaluate OSR known false-rejection drift & unknown separability across SNR regimes and levels."""
    ks = np.asarray(known_scores, dtype=np.float64).reshape(-1)
    kp = np.asarray(known_preds, dtype=np.int64).reshape(-1)
    kl = np.asarray(known_labels, dtype=np.int64).reshape(-1)
    ksnr = np.asarray(known_snrs, dtype=np.int64).reshape(-1)

    us = np.asarray(unknown_scores, dtype=np.float64).reshape(-1)
    _ = np.asarray(unknown_labels, dtype=np.int64).reshape(-1)
    usnr = np.asarray(unknown_snrs, dtype=np.int64).reshape(-1)

    use_regimes = DEFAULT_SNR_REGIMES if regimes is None else {k: [int(s) for s in v] for k, v in regimes.items()}
    thr = float(calibrated_threshold)

    regime_metrics: dict[str, dict[str, Any]] = {}
    for reg_name, snr_list in use_regimes.items():
        km = np.isin(ksnr, snr_list)
        um = np.isin(usnr, snr_list)
        if not km.any() or not um.any():
            continue
        res = _compute_osr_slice(ks[km], kp[km], kl[km], us[um], thr, target_tpr)
        res["snrs"] = snr_list
        regime_metrics[reg_name] = res

    per_snr_metrics: dict[str, dict[str, Any]] = {}
    common_snrs = sorted(set(np.unique(ksnr).tolist()) & set(np.unique(usnr).tolist()))
    for snr_val in common_snrs:
        km = ksnr == snr_val
        um = usnr == snr_val
        if not km.any() or not um.any():
            continue
        per_snr_metrics[str(int(snr_val))] = _compute_osr_slice(ks[km], kp[km], kl[km], us[um], thr, target_tpr)

    drift_summary: dict[str, float | None] = {
        "frr_drift_mid_low_pct": None,
        "frr_drift_neg_pct": None,
        "auroc_drop_mid_low_pct": None,
        "auroc_drop_neg_pct": None,
        "oscr_drop_mid_low_pct": None,
        "oscr_drop_neg_pct": None,
    }
    if "high_snr" in regime_metrics:
        hi = regime_metrics["high_snr"]
        if "mid_low_snr" in regime_metrics:
            ml = regime_metrics["mid_low_snr"]
            drift_summary["frr_drift_mid_low_pct"] = round(ml["known_frr_pct"] - hi["known_frr_pct"], 2)
            drift_summary["auroc_drop_mid_low_pct"] = round(hi["auroc_pct"] - ml["auroc_pct"], 2)
            drift_summary["oscr_drop_mid_low_pct"] = round(hi["oscr_pct"] - ml["oscr_pct"], 2)
        if "neg_snr" in regime_metrics:
            ng = regime_metrics["neg_snr"]
            drift_summary["frr_drift_neg_pct"] = round(ng["known_frr_pct"] - hi["known_frr_pct"], 2)
            drift_summary["auroc_drop_neg_pct"] = round(hi["auroc_pct"] - ng["auroc_pct"], 2)
            drift_summary["oscr_drop_neg_pct"] = round(hi["oscr_pct"] - ng["oscr_pct"], 2)

    return {
        "calibrated_threshold": round(thr, 6),
        "regimes": regime_metrics,
        "per_snr": per_snr_metrics,
        "drift_summary": drift_summary,
    }


def evaluate_snr_cil_drift(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    snrs: np.ndarray,
    num_base_classes: int,
    current_true_classes: list[int],
    fixed_head_to_true: dict[int, int],
    regimes: dict[str, list[int]] | None = None,
    base_acc_0_by_regime: dict[str, float] | None = None,
    per_snr_base_acc_0: dict[int, float] | None = None,
) -> dict[str, Any]:
    """Evaluate post-evolution CIL accuracy degradation across SNR regimes with a fixed High-SNR mapping."""
    yt = np.asarray(y_true, dtype=np.int64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    snrs_arr = np.asarray(snrs, dtype=np.int64).reshape(-1)
    if not (len(yt) == len(yp) == len(snrs_arr)):
        raise ValueError("y_true, y_pred, and snrs must have identical lengths")

    use_regimes = DEFAULT_SNR_REGIMES if regimes is None else {k: [int(s) for s in v] for k, v in regimes.items()}
    mapping = {int(k): int(v) for k, v in fixed_head_to_true.items()}

    regime_metrics: dict[str, dict[str, Any]] = {}
    for reg_name, snr_list in use_regimes.items():
        mask = np.isin(snrs_arr, snr_list)
        if not mask.any():
            continue
        b0 = None if base_acc_0_by_regime is None else base_acc_0_by_regime.get(reg_name)
        res = evaluate_cil_session(
            y_true=yt[mask],
            y_pred=yp[mask],
            num_base_classes=num_base_classes,
            current_true_classes=current_true_classes,
            head_to_true=mapping,
            base_acc_0=b0,
        )
        regime_metrics[reg_name] = {
            "snrs": snr_list,
            "n_samples": int(mask.sum()),
            "a_all_pct": round(res["a_all"] * 100.0, 2),
            "a_old_pct": round(res["a_old"] * 100.0, 2),
            "a_new_pct": round((res["a_new"] or 0.0) * 100.0, 2) if res["a_new"] is not None else None,
            "delta_a_old_pct": round(res["delta_a_old"] * 100.0, 2) if b0 is not None else None,
        }

    per_snr_metrics: dict[str, dict[str, Any]] = {}
    for snr_val in sorted(np.unique(snrs_arr).tolist()):
        mask = snrs_arr == snr_val
        b0 = None if per_snr_base_acc_0 is None else per_snr_base_acc_0.get(int(snr_val))
        res = evaluate_cil_session(
            y_true=yt[mask],
            y_pred=yp[mask],
            num_base_classes=num_base_classes,
            current_true_classes=current_true_classes,
            head_to_true=mapping,
            base_acc_0=b0,
        )
        per_snr_metrics[str(int(snr_val))] = {
            "n_samples": int(mask.sum()),
            "a_all_pct": round(res["a_all"] * 100.0, 2),
            "a_old_pct": round(res["a_old"] * 100.0, 2),
            "a_new_pct": round((res["a_new"] or 0.0) * 100.0, 2) if res["a_new"] is not None else None,
            "delta_a_old_pct": round(res["delta_a_old"] * 100.0, 2) if b0 is not None else None,
        }

    drift_summary: dict[str, float | None] = {
        "a_all_drop_mid_low_pct": None,
        "a_all_drop_neg_pct": None,
        "a_old_drop_mid_low_pct": None,
        "a_old_drop_neg_pct": None,
    }
    if "high_snr" in regime_metrics:
        hi = regime_metrics["high_snr"]
        if "mid_low_snr" in regime_metrics:
            ml = regime_metrics["mid_low_snr"]
            drift_summary["a_all_drop_mid_low_pct"] = round(hi["a_all_pct"] - ml["a_all_pct"], 2)
            drift_summary["a_old_drop_mid_low_pct"] = round(hi["a_old_pct"] - ml["a_old_pct"], 2)
        if "neg_snr" in regime_metrics:
            ng = regime_metrics["neg_snr"]
            drift_summary["a_all_drop_neg_pct"] = round(hi["a_all_pct"] - ng["a_all_pct"], 2)
            drift_summary["a_old_drop_neg_pct"] = round(hi["a_old_pct"] - ng["a_old_pct"], 2)

    return {
        "fixed_head_to_true": mapping,
        "regimes": regime_metrics,
        "per_snr": per_snr_metrics,
        "drift_summary": drift_summary,
    }
