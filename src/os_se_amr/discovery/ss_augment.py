"""Intra-cluster TLDNN Segment Substitution (SS-IQ & SS-PhaseAligned) for few-shot novel classes."""

from __future__ import annotations

import numpy as np

VALID_SS_MODES = ("none", "ss_iq", "ss_phase_aligned")


def ss_iq_pair(x_a: np.ndarray, x_b: np.ndarray, cut: int | None = None) -> np.ndarray:
    """Direct time-domain half-frame substitution between two I/Q waveforms (2, L)."""
    arr_a = np.asarray(x_a, dtype=np.float32)
    arr_b = np.asarray(x_b, dtype=np.float32)
    if arr_a.ndim != 2 or arr_a.shape != arr_b.shape or arr_a.shape[0] != 2:
        raise ValueError(f"Expected matching (2, L) waveforms, got {arr_a.shape} and {arr_b.shape}")
    c = (arr_a.shape[-1] // 2) if cut is None else int(cut)
    return np.concatenate([arr_a[:, :c], arr_b[:, c:]], axis=1).astype(np.float32)


def ss_phase_aligned_pair(x_a: np.ndarray, x_b: np.ndarray, cut: int | None = None) -> np.ndarray:
    """Phase-continuous intra-cluster substitution aligning seam instantaneous phase at cut."""
    arr_a = np.asarray(x_a, dtype=np.float32)
    arr_b = np.asarray(x_b, dtype=np.float32)
    if arr_a.ndim != 2 or arr_a.shape != arr_b.shape or arr_a.shape[0] != 2:
        raise ValueError(f"Expected matching (2, L) waveforms, got {arr_a.shape} and {arr_b.shape}")
    c = (arr_a.shape[-1] // 2) if cut is None else int(cut)
    if c <= 0 or c >= arr_a.shape[-1]:
        raise ValueError(f"cut must be in (0, {arr_a.shape[-1]}), got {c}")

    theta_a = float(np.arctan2(arr_a[1, c - 1], arr_a[0, c - 1]))
    theta_b = float(np.arctan2(arr_b[1, c], arr_b[0, c]))
    d_theta = theta_a - theta_b
    cos_d = float(np.cos(d_theta))
    sin_d = float(np.sin(d_theta))

    i_rot = cos_d * arr_b[0, c:] - sin_d * arr_b[1, c:]
    q_rot = sin_d * arr_b[0, c:] + cos_d * arr_b[1, c:]
    right = np.stack([i_rot, q_rot], axis=0)
    return np.concatenate([arr_a[:, :c], right], axis=1).astype(np.float32)


def _augment_cluster_vectorized(
    x_c: np.ndarray,
    n_syn: int,
    mode: str,
    rng: np.random.Generator,
    cut: int,
) -> np.ndarray:
    n_c = len(x_c)
    idx_a = rng.integers(0, n_c, size=n_syn, dtype=np.int64)
    offsets = rng.integers(1, n_c, size=n_syn, dtype=np.int64)
    idx_b = (idx_a + offsets) % n_c

    left = x_c[idx_a, :, :cut]
    right = x_c[idx_b, :, cut:].copy()

    if mode == "ss_phase_aligned":
        theta_a = np.arctan2(x_c[idx_a, 1, cut - 1], x_c[idx_a, 0, cut - 1])
        theta_b = np.arctan2(x_c[idx_b, 1, cut], x_c[idx_b, 0, cut])
        d_theta = (theta_a - theta_b)[:, None]
        cos_d = np.cos(d_theta)
        sin_d = np.sin(d_theta)

        r_i = right[:, 0, :].copy()
        r_q = right[:, 1, :].copy()
        right[:, 0, :] = cos_d * r_i - sin_d * r_q
        right[:, 1, :] = sin_d * r_i + cos_d * r_q

    return np.concatenate([left, right], axis=2).astype(np.float32)


def augment_core_clusters(
    x_core: np.ndarray,
    cluster_labels: np.ndarray,
    mode: str = "ss_phase_aligned",
    expand_factor: int = 4,
    max_samples_per_cluster: int = 2000,
    seed: int = 42,
    per_cluster_silhouette: dict[int, float] | None = None,
    per_cluster_cohesion: dict[int, float] | None = None,
    min_cluster_silhouette: float | None = None,
    min_cluster_cohesion: float | None = None,
    max_core_size_for_ss: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand high-purity core clusters strictly intra-cluster using TLDNN-SS.

    No cross-cluster stitching is ever performed; noise labels (< 0) are ignored.
    Optional quality/size gating selectively skips clusters that already have ample
    samples or exhibit weak geometric cohesion/silhouette.
    """
    if mode not in VALID_SS_MODES:
        raise ValueError(f"Unsupported SS mode='{mode}'; choose from {VALID_SS_MODES}")

    x_arr = np.ascontiguousarray(x_core, dtype=np.float32)
    y_arr = np.asarray(cluster_labels, dtype=np.int64).reshape(-1)
    if len(x_arr) != len(y_arr):
        raise ValueError(f"Length mismatch: x_core={len(x_arr)}, cluster_labels={len(y_arr)}")
    if len(x_arr) == 0 or mode == "none" or int(expand_factor) <= 1:
        return x_arr.copy(), y_arr.copy()

    rng = np.random.default_rng(int(seed))
    cut = x_arr.shape[-1] // 2
    out_x = [x_arr]
    out_y = [y_arr]

    valid_cids = sorted(int(c) for c in np.unique(y_arr).tolist() if c >= 0)
    n_clusters = len(valid_cids)

    for cid in valid_cids:
        c_mask = y_arr == cid
        x_c = x_arr[c_mask]
        n_c = len(x_c)
        if n_c < 2:
            continue
        if max_core_size_for_ss is not None and n_c > int(max_core_size_for_ss):
            continue
        if min_cluster_cohesion is not None and per_cluster_cohesion is not None:
            if per_cluster_cohesion.get(cid, 0.0) < float(min_cluster_cohesion):
                continue
        if n_clusters >= 2 and min_cluster_silhouette is not None and per_cluster_silhouette is not None:
            if per_cluster_silhouette.get(cid, 0.0) < float(min_cluster_silhouette):
                continue

        target_total = min(int(max_samples_per_cluster), n_c * int(expand_factor))
        n_syn = max(0, target_total - n_c)
        if n_syn == 0:
            continue

        x_syn = _augment_cluster_vectorized(x_c=x_c, n_syn=n_syn, mode=mode, rng=rng, cut=cut)
        out_x.append(x_syn)
        out_y.append(np.full(n_syn, cid, dtype=np.int64))

    return np.ascontiguousarray(np.concatenate(out_x, axis=0), dtype=np.float32), np.concatenate(out_y, axis=0)
