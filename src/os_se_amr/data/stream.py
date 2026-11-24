"""Multi-session streaming signal arrival generator strictly drawing from Train splits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from os_se_amr.data.splitter import SplitBundle


@dataclass
class StreamBatch:
    """Single arrival session waveform batch (labels kept solely for offline evaluation)."""

    x_iq: np.ndarray
    true_labels: np.ndarray
    snrs: np.ndarray
    session_id: int
    arriving_classes: list[str]
    mode: str


def _sample_balanced_from_classes(
    bundle: SplitBundle,
    classes: list[str],
    total_per_class: int,
    rng: np.random.Generator,
    snrs: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample `total_per_class` train waveforms balanced across target SNRs for each class."""
    xs, ys, ss = [], [], []
    target_snrs = bundle.main_snrs if snrs is None else [int(s) for s in snrs]
    if not target_snrs:
        raise ValueError("target_snrs must be non-empty")
    n_snrs = len(target_snrs)
    base_per_snr = max(1, total_per_class // n_snrs)
    remainder = total_per_class - base_per_snr * n_snrs

    for cls_name in classes:
        label_id = bundle.all_to_idx[cls_name]
        for s_idx, snr in enumerate(target_snrs):
            key = (cls_name, snr)
            avail_idx = bundle.indices[key]["train"]
            k = base_per_snr + (1 if s_idx < remainder else 0)
            k = min(k, len(avail_idx))
            chosen = rng.choice(avail_idx, size=k, replace=False)
            xs.append(bundle.raw_dict[key][chosen])
            ys.append(np.full(k, label_id, dtype=np.int64))
            ss.append(np.full(k, snr, dtype=np.int64))

    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), np.concatenate(ss, axis=0)


def build_arrival_sessions(
    bundle: SplitBundle,
    order_name: str = "order_a",
    samples_per_class: int = 1000,
    mode: str = "decoupled",
    known_ratio: float = 0.5,
    snrs: list[int] | None = None,
    seed: int = 42,
) -> list[StreamBatch]:
    """Construct ordered streaming sessions strictly from the Train partition.

    Args:
        bundle: Leakage-free SplitBundle.
        order_name: Key inside split_cfg['arrival_orders'] ('order_a' or 'order_b').
        samples_per_class: Waveforms sampled per arriving novel class.
        mode: 'decoupled' (unknown classes only, Stage 2) or 'mixed' (known + unknown stream, Stage 4).
        known_ratio: Fraction of known samples when mode == 'mixed'.
        snrs: Optional SNR list filter (defaults to bundle.main_snrs).
        seed: Deterministic RNG seed.
    """
    orders = bundle.split_cfg.get("arrival_orders", {})
    if order_name not in orders:
        raise KeyError(f"Arrival order '{order_name}' not found in split_cfg['arrival_orders']")
    if mode not in ("decoupled", "mixed"):
        raise ValueError(f"Unsupported stream mode '{mode}'; expected 'decoupled' or 'mixed'")

    rng = np.random.default_rng(seed)
    seen_classes = list(bundle.known_classes)
    sessions: list[StreamBatch] = []

    for sess_idx, arriving in enumerate(orders[order_name], start=1):
        arriving_list = [str(c) for c in arriving]
        x_unk, y_unk, s_unk = _sample_balanced_from_classes(
            bundle=bundle,
            classes=arriving_list,
            total_per_class=samples_per_class,
            rng=rng,
            snrs=snrs,
        )

        if mode == "mixed" and 0.0 < known_ratio < 1.0:
            total_known = int(round(len(x_unk) * (known_ratio / (1.0 - known_ratio))))
            per_known = max(1, total_known // len(seen_classes))
            x_k, y_k, s_k = _sample_balanced_from_classes(
                bundle=bundle,
                classes=seen_classes,
                total_per_class=per_known,
                rng=rng,
                snrs=snrs,
            )
            x_all = np.concatenate([x_k, x_unk], axis=0)
            y_all = np.concatenate([y_k, y_unk], axis=0)
            s_all = np.concatenate([s_k, s_unk], axis=0)
        else:
            x_all, y_all, s_all = x_unk, y_unk, s_unk

        perm = rng.permutation(len(x_all))
        sessions.append(
            StreamBatch(
                x_iq=np.ascontiguousarray(x_all[perm], dtype=np.float32),
                true_labels=np.ascontiguousarray(y_all[perm], dtype=np.int64),
                snrs=np.ascontiguousarray(s_all[perm], dtype=np.int64),
                session_id=sess_idx,
                arriving_classes=arriving_list,
                mode=mode,
            )
        )
        seen_classes.extend(arriving_list)

    return sessions
