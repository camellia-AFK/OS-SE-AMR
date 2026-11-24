"""Unsupervised discovery comparison baselines: K-Means and parameter-free FINCH (CVPR 2019)."""

from __future__ import annotations

import math

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

from os_se_amr.discovery.hdbscan_cluster import DiscoveryResult, HDBSCANDiscoverer, _l2_normalize


def _finalize_partition_result(
    x_clust: np.ndarray,
    ref_64: np.ndarray,
    labels: np.ndarray,
    min_cluster_size: int,
    core_top_ratio: float,
    silhouette_threshold: float,
) -> DiscoveryResult:
    """Filter undersized clusters, remap IDs 0..K-1, compute silhouettes & core exemplars."""
    n_samples = len(x_clust)
    raw_labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    kept_ids = [
        int(cid)
        for cid in sorted(np.unique(raw_labels).tolist())
        if cid >= 0 and int((raw_labels == cid).sum()) >= min_cluster_size
    ]
    remapped = np.full(n_samples, -1, dtype=np.int64)
    for new_id, oid in enumerate(kept_ids):
        remapped[raw_labels == oid] = new_id

    k_hat = len(kept_ids)
    valid = remapped >= 0
    noise_ratio = float(1.0 - valid.mean())
    if k_hat == 0:
        return DiscoveryResult(
            accepted=False,
            reason="no_valid_clusters",
            cluster_ids=remapped,
            membership_probs=np.zeros(n_samples, dtype=np.float32),
            k_hat=0,
            noise_ratio=noise_ratio,
            silhouette=-1.0,
            per_cluster_silhouette={},
            core_mask=np.zeros(n_samples, dtype=bool),
            prototypes_64d=np.empty((0, ref_64.shape[1]), dtype=np.float32),
            cluster_sizes={},
            cluster_ids_unique=[],
        )

    cids = list(range(k_hat))
    cluster_sizes = {c: int((remapped == c).sum()) for c in cids}
    per_sil: dict[int, float] = {}
    if k_hat >= 2:
        sil = float(silhouette_score(x_clust[valid], remapped[valid]))
        ssamp = silhouette_samples(x_clust[valid], remapped[valid])
        vl = remapped[valid]
        for c in cids:
            per_sil[c] = float(ssamp[vl == c].mean())
        accepted = sil > silhouette_threshold
        reason = "accepted" if accepted else "silhouette_below_threshold"
    else:
        sil = 0.0
        per_sil[0] = 0.0
        accepted = True
        reason = "accepted_single_cluster"

    probs = np.zeros(n_samples, dtype=np.float32)
    core_mask = np.zeros(n_samples, dtype=bool)
    protos = np.zeros((k_hat, ref_64.shape[1]), dtype=np.float32)
    cohesions: dict[int, float] = {}

    for c in cids:
        c_idx = np.where(remapped == c)[0]
        cent = x_clust[c_idx].mean(axis=0, keepdims=True)
        dists = np.linalg.norm(x_clust[c_idx] - cent, axis=1)
        order = np.argsort(dists)
        k_core = max(1, min(len(c_idx), int(math.ceil(core_top_ratio * len(c_idx)))))
        sel = c_idx[order[:k_core]]
        core_mask[sel] = True
        probs[c_idx] = 1.0
        protos[c] = ref_64[sel].mean(axis=0)
        cu = cent[0] / max(float(np.linalg.norm(cent[0])), 1e-8)
        cohesions[c] = float((x_clust[sel] @ cu).mean())

    return DiscoveryResult(
        accepted=accepted,
        reason=reason,
        cluster_ids=remapped,
        membership_probs=probs,
        k_hat=k_hat,
        noise_ratio=noise_ratio,
        silhouette=sil,
        per_cluster_silhouette=per_sil,
        core_mask=core_mask,
        prototypes_64d=protos,
        cluster_sizes=cluster_sizes,
        cluster_ids_unique=cids,
        cluster_cohesions=cohesions,
        raw_k_hat=k_hat,
        merged_groups=[[c] for c in cids],
    )


class KMeansDiscoverer:
    """K-Means baseline with fixed K or silhouette-optimal K over a bounded range."""

    def __init__(
        self,
        n_clusters: int | None = None,
        k_range: tuple[int, int] = (2, 6),
        min_cluster_size: int = 10,
        core_top_ratio: float = 0.60,
        silhouette_threshold: float = 0.0,
        feature_mode: str = "stem_preln",
        seed: int = 42,
    ) -> None:
        self.n_clusters = int(n_clusters) if n_clusters is not None else None
        self.k_range = (int(k_range[0]), int(k_range[1]))
        self.min_cluster_size = int(min_cluster_size)
        self.core_top_ratio = float(core_top_ratio)
        self.silhouette_threshold = float(silhouette_threshold)
        self.feature_mode = feature_mode
        self.seed = int(seed)
        self._space_builder = HDBSCANDiscoverer(feature_mode=feature_mode)

    def discover(self, feats: dict[str, np.ndarray] | np.ndarray) -> DiscoveryResult:
        x_clust, ref_64 = self._space_builder.build_clustering_space(feats)
        if len(x_clust) < self.min_cluster_size:
            return _finalize_partition_result(
                x_clust, ref_64, np.full(len(x_clust), -1), self.min_cluster_size, self.core_top_ratio, self.silhouette_threshold
            )

        if self.n_clusters is not None:
            k_opt = min(self.n_clusters, len(x_clust))
            labels = KMeans(n_clusters=k_opt, n_init=10, random_state=self.seed).fit_predict(x_clust)
        else:
            best_sil, best_lbl = -2.0, np.zeros(len(x_clust), dtype=np.int64)
            k_lo, k_hi = max(2, self.k_range[0]), min(self.k_range[1], len(x_clust) - 1)
            for k in range(k_lo, max(k_lo, k_hi) + 1):
                lbl = KMeans(n_clusters=k, n_init=10, random_state=self.seed).fit_predict(x_clust)
                if len(np.unique(lbl)) < 2:
                    continue
                sil = float(silhouette_score(x_clust, lbl))
                if sil > best_sil:
                    best_sil, best_lbl = sil, lbl
            labels = best_lbl

        return _finalize_partition_result(
            x_clust, ref_64, labels, self.min_cluster_size, self.core_top_ratio, self.silhouette_threshold
        )


def _finch_step(feats: np.ndarray) -> tuple[np.ndarray, int]:
    """One FINCH first-integer-neighbor partition step over L2-normalized vectors."""
    n = len(feats)
    if n <= 1:
        return np.zeros(n, dtype=np.int64), 1
    sim = feats @ feats.T
    np.fill_diagonal(sim, -np.inf)
    nn1 = np.argmax(sim, axis=1)

    rows: list[int] = []
    cols: list[int] = []
    # Direct & symmetric 1-NN links
    idx = np.arange(n, dtype=np.int64)
    rows.extend(idx.tolist())
    cols.extend(nn1.tolist())
    # Shared-neighbor links grouped by nn1 target
    buckets: dict[int, list[int]] = {}
    for i, target in enumerate(nn1.tolist()):
        buckets.setdefault(target, []).append(i)
    for group in buckets.values():
        if len(group) > 1:
            anchor = group[0]
            for other in group[1:]:
                rows.append(anchor)
                cols.append(other)

    adj = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n))
    adj = adj + adj.T
    n_comp, comp_labels = connected_components(csgraph=adj, directed=False, return_labels=True)
    return np.asarray(comp_labels, dtype=np.int64), int(n_comp)


class FINCHDiscoverer:
    """Parameter-free FINCH clustering baseline (Sarfraz et al., CVPR 2019)."""

    def __init__(
        self,
        partition_level: int = -1,
        min_cluster_size: int = 10,
        core_top_ratio: float = 0.60,
        silhouette_threshold: float = 0.0,
        feature_mode: str = "stem_preln",
    ) -> None:
        self.partition_level = int(partition_level)
        self.min_cluster_size = int(min_cluster_size)
        self.core_top_ratio = float(core_top_ratio)
        self.silhouette_threshold = float(silhouette_threshold)
        self.feature_mode = feature_mode
        self._space_builder = HDBSCANDiscoverer(feature_mode=feature_mode)

    def discover(self, feats: dict[str, np.ndarray] | np.ndarray) -> DiscoveryResult:
        x_clust, ref_64 = self._space_builder.build_clustering_space(feats)
        n_samples = len(x_clust)
        if n_samples < self.min_cluster_size:
            return _finalize_partition_result(
                x_clust, ref_64, np.full(n_samples, -1), self.min_cluster_size, self.core_top_ratio, self.silhouette_threshold
            )

        partitions: list[np.ndarray] = []
        cur_feats = x_clust
        sample_to_cur = np.arange(n_samples, dtype=np.int64)

        for _ in range(8):
            step_lbl, k_cur = _finch_step(cur_feats)
            if k_cur == len(cur_feats) or k_cur < 1:
                break
            global_lbl = step_lbl[sample_to_cur]
            partitions.append(global_lbl)
            if k_cur <= 2:
                break
            sample_to_cur = global_lbl
            cents = np.stack([x_clust[global_lbl == c].mean(axis=0) for c in range(k_cur)], axis=0)
            cur_feats = _l2_normalize(cents)

        if not partitions:
            chosen = np.zeros(n_samples, dtype=np.int64)
        elif self.partition_level < 0:
            # Default: pick deepest partition with >= 2 non-trivial clusters if available
            candidates = [p for p in partitions if len(np.unique(p)) >= 2]
            chosen = candidates[-1] if candidates else partitions[-1]
        else:
            lvl = min(self.partition_level, len(partitions) - 1)
            chosen = partitions[lvl]

        return _finalize_partition_result(
            x_clust, ref_64, chosen, self.min_cluster_size, self.core_top_ratio, self.silhouette_threshold
        )
