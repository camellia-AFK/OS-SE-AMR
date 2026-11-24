"""Continual incremental learning evaluation metrics (A_all, A_old, A_new, FM)."""

from __future__ import annotations

from typing import Any

import numpy as np

from os_se_amr.eval.hungarian import apply_locked_mapping, locked_known_hungarian_match


def evaluate_cil_session(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_base_classes: int,
    current_true_classes: list[int] | None = None,
    prev_per_class_accs: list[dict[int, float]] | None = None,
    base_acc_0: float | None = None,
    head_to_true: dict[int, int] | None = None,
    num_locked_heads: int | None = None,
) -> dict[str, Any]:
    """Evaluate all-class, old-class, novel-class accuracies and forgetting measures at session t."""
    yt = np.asarray(y_true, dtype=np.int64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if len(yt) != len(yp):
        raise ValueError(f"Length mismatch: y_true={len(yt)}, y_pred={len(yp)}")

    classes_present = (
        sorted({int(c) for c in np.unique(yt).tolist()})
        if current_true_classes is None
        else sorted({int(c) for c in current_true_classes})
    )
    lock_limit = int(num_base_classes) if num_locked_heads is None else int(num_locked_heads)

    match_info: dict[str, Any]
    if head_to_true is None:
        novel_true = [c for c in classes_present if c >= lock_limit]
        match_info = locked_known_hungarian_match(
            y_true=yt,
            y_pred=yp,
            num_known=lock_limit,
            new_true_labels=novel_true,
        )
        mapping = match_info["head_to_true"]
    else:
        mapping = {int(k): int(v) for k, v in head_to_true.items()}
        match_info = {
            "head_to_true": mapping,
            "k_hat": len(mapping),
            "k_true": len([c for c in classes_present if c >= num_base_classes]),
            "delta_k": abs(len(mapping) - len([c for c in classes_present if c >= num_base_classes])),
        }

    y_mapped = apply_locked_mapping(yp, num_known=lock_limit, head_to_true=mapping)

    a_all = float((y_mapped == yt).mean()) if len(yt) > 0 else 0.0

    old_mask = yt < num_base_classes
    a_old = float((y_mapped[old_mask] == yt[old_mask]).mean()) if old_mask.any() else 0.0

    new_mask = yt >= num_base_classes
    a_new = float((y_mapped[new_mask] == yt[new_mask]).mean()) if new_mask.any() else None

    per_class_acc: dict[int, float] = {}
    for c in classes_present:
        c_mask = yt == c
        per_class_acc[c] = float((y_mapped[c_mask] == yt[c_mask]).mean()) if c_mask.any() else 0.0

    delta_a_old = float(base_acc_0 - a_old) if base_acc_0 is not None else 0.0

    fm_cls = 0.0
    if prev_per_class_accs:
        forgetting_vals = []
        for c in range(num_base_classes):
            past_max = max(prev_map.get(c, 0.0) for prev_map in prev_per_class_accs)
            forgetting_vals.append(past_max - per_class_acc.get(c, 0.0))
        fm_cls = float(np.mean(forgetting_vals))

    return {
        "a_all": a_all,
        "a_old": a_old,
        "a_new": a_new,
        "delta_a_old": delta_a_old,
        "fm_cls": fm_cls,
        "per_class_acc": per_class_acc,
        "head_to_true": mapping,
        "match_info": match_info,
    }
