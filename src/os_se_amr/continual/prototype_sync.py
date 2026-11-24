"""Post-update prototype refresh and Mahalanobis scorer synchronization."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from os_se_amr.continual.replay import ReplayBuffer
from os_se_amr.data.transforms import extract_rf_physics_features
from os_se_amr.models.base import BaseBackbone
from os_se_amr.osr.mahalanobis import MahalanobisPrototypeScorer


@torch.no_grad()
def _extract_ms_arrays(
    backbone: BaseBackbone,
    x_iq: np.ndarray | torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
    include_physics: bool = False,
) -> dict[str, np.ndarray]:
    backbone.eval()
    x_t = torch.as_tensor(x_iq, dtype=torch.float32)
    buffers: dict[str, list[np.ndarray]] = {}
    for start in range(0, len(x_t), batch_size):
        xb = x_t[start : start + batch_size].to(device, non_blocking=True)
        ms = backbone.extract_multiscale_features(xb)
        for k, v in ms.items():
            buffers.setdefault(k, []).append(v.cpu().numpy())
        if include_physics:
            buffers.setdefault("physics", []).append(extract_rf_physics_features(xb).cpu().numpy())
    return {k: np.concatenate(v_list, axis=0) for k, v_list in buffers.items()}


def sync_prototypes_and_mahalanobis(
    backbone: BaseBackbone,
    replay_buffer: ReplayBuffer,
    new_core_x: np.ndarray | torch.Tensor,
    new_core_labels: np.ndarray | torch.Tensor,
    prev_scorer: MahalanobisPrototypeScorer | None,
    device: str | torch.device,
    batch_size: int = 256,
    layers: tuple[str, ...] = ("stem", "pre_ln"),
    weights: dict[str, float] | None = None,
    num_old_classes: int | None = None,
    val_feats: dict[str, np.ndarray] | None = None,
    include_physics: bool | None = None,
) -> dict[str, Any]:
    """Refresh old and newly discovered class prototypes after incremental backbone updates."""
    dev = torch.device(device)
    old_x, old_y = replay_buffer.as_tensors()
    if len(old_x) == 0:
        raise ValueError("replay_buffer must contain old-class exemplars for prototype sync")

    old_classes = replay_buffer.class_ids
    new_x_np = np.asarray(new_core_x, dtype=np.float32)
    new_local_np = np.asarray(new_core_labels, dtype=np.int64).reshape(-1)
    if len(new_x_np) == 0 or len(new_local_np) == 0:
        raise ValueError("new_core_x and new_core_labels must be non-empty")

    new_local_min = int(new_local_np.min())
    new_local_max = int(new_local_np.max())
    if num_old_classes is None:
        if new_local_min == 0:
            k_new = new_local_max + 1
            if backbone.num_classes > k_new and len(old_classes) >= backbone.num_classes:
                num_old = int(backbone.num_classes - k_new)
            else:
                num_old = len(old_classes)
        else:
            num_old = new_local_min
    else:
        num_old = int(num_old_classes)

    old_mask = old_y < num_old
    old_x_filt = old_x[old_mask]
    old_y_filt = old_y[old_mask]
    if len(old_x_filt) == 0:
        raise ValueError(f"No old-class exemplars with class_id < {num_old} in replay_buffer")

    new_global_np = (new_local_np + num_old) if new_local_min < num_old else new_local_np

    all_x = np.concatenate([old_x_filt.numpy(), new_x_np], axis=0)
    all_y = np.concatenate([old_y_filt.numpy(), new_global_np], axis=0)

    if prev_scorer is not None:
        use_layers = prev_scorer.layers
        use_weights = prev_scorer.weights
    else:
        use_layers = layers
        use_weights = weights if weights is not None else {"stem": 0.7, "pre_ln": 0.3}

    use_physics = ("physics" in use_layers) if include_physics is None else bool(include_physics)
    ms_feats = _extract_ms_arrays(backbone, all_x, device=dev, batch_size=batch_size, include_physics=use_physics)

    refreshed = MahalanobisPrototypeScorer(layers=use_layers, weights=use_weights)
    refreshed.fit_multiscale(ms_feats, all_y, val_feats=val_feats)

    shift_pre_ln = 0.0
    shift_stem = 0.0
    if prev_scorer is not None:
        if "pre_ln" in prev_scorer.means and "pre_ln" in refreshed.means:
            n_cmp = min(num_old, len(prev_scorer.means["pre_ln"]), len(refreshed.means["pre_ln"]))
            if n_cmp > 0:
                shift_pre_ln = float(
                    np.linalg.norm(refreshed.means["pre_ln"][:n_cmp] - prev_scorer.means["pre_ln"][:n_cmp], axis=1).mean()
                )
        if "stem" in prev_scorer.means and "stem" in refreshed.means:
            n_cmp = min(num_old, len(prev_scorer.means["stem"]), len(refreshed.means["stem"]))
            if n_cmp > 0:
                shift_stem = float(
                    np.linalg.norm(refreshed.means["stem"][:n_cmp] - prev_scorer.means["stem"][:n_cmp], axis=1).mean()
                )

    return {
        "scorer": refreshed,
        "old_proto_shift_pre_ln": shift_pre_ln,
        "old_proto_shift_stem": shift_stem,
        "num_synced_classes": len(refreshed.classes),
    }
