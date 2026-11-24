"""Replay buffer storing representative raw I/Q exemplars selected via Herding or random."""

from __future__ import annotations

import numpy as np
import torch


def _herding_select(features: np.ndarray, k: int) -> np.ndarray:
    """iCaRL greedy herding in L2-normalized feature space."""
    n = len(features)
    k = min(k, n)
    if k <= 0:
        return np.empty(0, dtype=np.int64)

    f64 = np.asarray(features, dtype=np.float64)
    norms = np.maximum(np.linalg.norm(f64, axis=1, keepdims=True), 1e-8)
    f_norm = f64 / norms
    class_mean = f_norm.mean(axis=0)

    selected: list[int] = []
    available = np.ones(n, dtype=bool)
    selected_sum = np.zeros_like(class_mean)

    for step in range(1, k + 1):
        cand_indices = np.where(available)[0]
        cand_means = (selected_sum[None, :] + f_norm[cand_indices]) / float(step)
        dists = np.linalg.norm(cand_means - class_mean[None, :], axis=1)
        best_local = int(np.argmin(dists))
        best_idx = int(cand_indices[best_local])
        selected.append(best_idx)
        available[best_idx] = False
        selected_sum += f_norm[best_idx]

    return np.asarray(selected, dtype=np.int64)


class ReplayBuffer:
    """Fixed-budget exemplar buffer preserving raw I/Q waveforms per seen class."""

    def __init__(
        self,
        exemplars_per_class: int = 100,
        selection_strategy: str = "herding",
    ) -> None:
        if selection_strategy not in ("herding", "random"):
            raise ValueError(f"Unsupported selection_strategy='{selection_strategy}'")
        self.exemplars_per_class = int(exemplars_per_class)
        self.selection_strategy = selection_strategy
        self._storage: dict[int, np.ndarray] = {}

    @property
    def class_ids(self) -> list[int]:
        return sorted(self._storage.keys())

    def __len__(self) -> int:
        return sum(len(arr) for arr in self._storage.values())

    def add_class_exemplars(
        self,
        class_id: int,
        x_iq: np.ndarray,
        features: np.ndarray | None = None,
        snrs: np.ndarray | None = None,
        seed: int = 42,
    ) -> None:
        """Select and store up to `exemplars_per_class` raw I/Q exemplars for `class_id`.

        When `snrs` is provided alongside herding, allocates the budget uniformly across
        SNR strata before running herding within each SNR bin.
        """
        x_arr = np.ascontiguousarray(x_iq, dtype=np.float32)
        if x_arr.ndim != 3 or x_arr.shape[1] != 2 or x_arr.shape[2] <= 0:
            raise ValueError(f"Expected (N, 2, L) waveforms, got {x_arr.shape}")
        if len(x_arr) == 0:
            return

        k = min(self.exemplars_per_class, len(x_arr))
        if self.selection_strategy == "herding":
            if features is None:
                raise ValueError("features must be provided when selection_strategy='herding'")
            if len(features) != len(x_arr):
                raise ValueError(f"Length mismatch: x_iq={len(x_arr)}, features={len(features)}")

            if snrs is not None:
                snrs_arr = np.asarray(snrs, dtype=np.int64).reshape(-1)
                if len(snrs_arr) != len(x_arr):
                    raise ValueError(f"Length mismatch: x_iq={len(x_arr)}, snrs={len(snrs_arr)}")
                uniq_snrs = sorted(np.unique(snrs_arr).tolist())
                n_strata = len(uniq_snrs)
                base_q, rem = divmod(k, n_strata)
                chosen_parts: list[np.ndarray] = []
                for idx_s, s_val in enumerate(uniq_snrs):
                    s_indices = np.where(snrs_arr == s_val)[0]
                    target_s = base_q + (1 if idx_s < rem else 0)
                    if target_s <= 0 or len(s_indices) == 0:
                        continue
                    local_sel = _herding_select(features[s_indices], min(target_s, len(s_indices)))
                    chosen_parts.append(s_indices[local_sel])
                chosen = np.concatenate(chosen_parts) if chosen_parts else _herding_select(features, k)
                # Top up with global herding if any stratum had fewer samples than quota
                if len(chosen) < k:
                    used = set(chosen.tolist())
                    rem_idx = np.asarray([i for i in range(len(x_arr)) if i not in used], dtype=np.int64)
                    extra = _herding_select(features[rem_idx], k - len(chosen))
                    chosen = np.concatenate([chosen, rem_idx[extra]])
            else:
                chosen = _herding_select(features, k)
        else:
            rng = np.random.default_rng(seed + int(class_id))
            chosen = np.sort(rng.choice(len(x_arr), size=k, replace=False))

        self._storage[int(class_id)] = np.ascontiguousarray(x_arr[chosen], dtype=np.float32)

    def as_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return stacked (x_iq[M, 2, 128], labels[M]) tensors."""
        if not self._storage:
            return torch.empty((0, 2, 128), dtype=torch.float32), torch.empty((0,), dtype=torch.long)

        xs, ys = [], []
        for cid in self.class_ids:
            arr = self._storage[cid]
            xs.append(arr)
            ys.append(np.full(len(arr), cid, dtype=np.int64))

        return torch.from_numpy(np.concatenate(xs, axis=0)), torch.from_numpy(np.concatenate(ys, axis=0))
