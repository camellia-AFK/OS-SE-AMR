"""Locked-known Hungarian 1-to-1 cluster matching and unsupervised clustering diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def locked_known_hungarian_match(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_known: int,
    new_true_labels: list[int] | None = None,
    new_pred_heads: list[int] | None = None,
) -> dict[str, Any]:
    """Solve optimal 1-to-1 assignment for novel classifier heads while keeping 0..num_known-1 locked.

    Under-clustering (K_hat < K_true): unmatched true novel classes get 0% accuracy.
    Over-clustering (K_hat > K_true): unmatched extra heads stay unmapped (treated as wrong in strict 1-to-1).
    """
    yt = np.asarray(y_true, dtype=np.int64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if len(yt) != len(yp):
        raise ValueError(f"Length mismatch: y_true={len(yt)}, y_pred={len(yp)}")

    if new_true_labels is None:
        true_novel = sorted({int(c) for c in np.unique(yt).tolist() if c >= num_known})
    else:
        true_novel = sorted({int(c) for c in new_true_labels})

    if new_pred_heads is None:
        pred_novel = sorted({int(h) for h in np.unique(yp).tolist() if h >= num_known})
    else:
        pred_novel = sorted({int(h) for h in new_pred_heads})

    k_hat = len(pred_novel)
    k_true = len(true_novel)
    head_to_true: dict[int, int] = {}
    matched_pairs: list[dict[str, int]] = []
    contingency = np.zeros((k_hat, k_true), dtype=np.int64)

    if k_hat > 0 and k_true > 0:
        for i, head in enumerate(pred_novel):
            mask_h = yp == head
            for j, lbl in enumerate(true_novel):
                contingency[i, j] = int((mask_h & (yt == lbl)).sum())

        row_ind, col_ind = linear_sum_assignment(-contingency)
        for r, c in zip(row_ind, col_ind):
            head_id = pred_novel[int(r)]
            true_id = true_novel[int(c)]
            hits = int(contingency[r, c])
            head_to_true[head_id] = true_id
            matched_pairs.append({"pred_head": head_id, "true_label": true_id, "hits": hits})

    return {
        "head_to_true": head_to_true,
        "k_hat": k_hat,
        "k_true": k_true,
        "delta_k": abs(k_hat - k_true),
        "matched_pairs": matched_pairs,
        "contingency_matrix": contingency.tolist(),
    }


def apply_locked_mapping(
    y_pred: np.ndarray,
    num_known: int,
    head_to_true: dict[int, int],
) -> np.ndarray:
    """Map predicted heads using locked known indices and Hungarian-matched novel indices."""
    yp = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    mapped = np.full_like(yp, fill_value=-1)

    known_mask = (yp >= 0) & (yp < num_known)
    mapped[known_mask] = yp[known_mask]

    for pred_head, true_lbl in head_to_true.items():
        mapped[yp == int(pred_head)] = int(true_lbl)

    return mapped


def evaluate_clustering_diagnostics(
    y_true: np.ndarray,
    cluster_ids: np.ndarray,
) -> dict[str, Any]:
    """Compute unsupervised clustering quality metrics over non-noise samples (cluster_ids >= 0)."""
    yt = np.asarray(y_true, dtype=np.int64).reshape(-1)
    cid = np.asarray(cluster_ids, dtype=np.int64).reshape(-1)
    if len(yt) != len(cid):
        raise ValueError(f"Length mismatch: y_true={len(yt)}, cluster_ids={len(cid)}")

    valid = cid >= 0
    noise_ratio = float(1.0 - valid.mean()) if len(cid) > 0 else 1.0
    if not valid.any():
        return {
            "k_hat": 0,
            "k_true": int(len(np.unique(yt))),
            "noise_ratio": noise_ratio,
            "nmi": 0.0,
            "ari": 0.0,
            "many_to_1_purity": 0.0,
            "strict_1to1_acc": 0.0,
        }

    yt_v = yt[valid]
    cid_v = cid[valid]
    uniq_clusters = sorted(np.unique(cid_v).tolist())
    uniq_true = sorted(np.unique(yt_v).tolist())

    nmi = float(normalized_mutual_info_score(yt_v, cid_v))
    ari = float(adjusted_rand_score(yt_v, cid_v))

    w = np.zeros((len(uniq_clusters), len(uniq_true)), dtype=np.int64)
    for i, cl in enumerate(uniq_clusters):
        mask_c = cid_v == cl
        for j, tl in enumerate(uniq_true):
            w[i, j] = int((mask_c & (yt_v == tl)).sum())

    purity = float(w.max(axis=1).sum()) / float(len(yt_v))
    row_ind, col_ind = linear_sum_assignment(-w)
    strict_acc = float(w[row_ind, col_ind].sum()) / float(len(yt_v))

    return {
        "k_hat": len(uniq_clusters),
        "k_true": len(uniq_true),
        "noise_ratio": noise_ratio,
        "nmi": nmi,
        "ari": ari,
        "many_to_1_purity": purity,
        "strict_1to1_acc": strict_acc,
    }
