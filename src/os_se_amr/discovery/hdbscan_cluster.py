"""Density-based unknown modulation discovery using HDBSCAN and silhouette gating."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score


def _l2_normalize(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr64 = np.asarray(arr, dtype=np.float64)
    norms = np.maximum(np.linalg.norm(arr64, axis=1, keepdims=True), eps)
    return arr64 / norms


@dataclass
class DiscoveryResult:
    """Output artifact of an HDBSCAN unknown-discovery pass."""

    accepted: bool
    reason: str
    cluster_ids: np.ndarray
    membership_probs: np.ndarray
    k_hat: int
    noise_ratio: float
    silhouette: float
    per_cluster_silhouette: dict[int, float]
    core_mask: np.ndarray
    prototypes_64d: np.ndarray
    cluster_sizes: dict[int, int]
    cluster_ids_unique: list[int]
    cluster_cohesions: dict[int, float] = field(default_factory=dict)
    raw_k_hat: int = 0
    merged_groups: list[list[int]] = field(default_factory=list)


class HDBSCANDiscoverer:
    """Discover novel modulation clusters from buffered embeddings with silhouette quality gating."""

    def __init__(
        self,
        min_cluster_size: int = 50,
        min_samples: int = 20,
        silhouette_threshold: float = 0.45,
        core_prob_threshold: float = 0.75,
        core_top_ratio: float = 0.60,
        feature_mode: str = "stem_preln",
        allow_single_cluster: bool = False,
        merge_cosine_threshold: float | None = None,
        min_cluster_silhouette: float | None = None,
        min_cluster_cohesion: float | None = None,
    ) -> None:
        if feature_mode not in ("stem_preln", "penult"):
            raise ValueError(f"Unsupported feature_mode='{feature_mode}'; use 'stem_preln' or 'penult'")
        self.min_cluster_size = int(min_cluster_size)
        self.min_samples = int(min_samples)
        self.silhouette_threshold = float(silhouette_threshold)
        self.core_prob_threshold = float(core_prob_threshold)
        self.core_top_ratio = float(core_top_ratio)
        self.feature_mode = feature_mode
        self.allow_single_cluster = bool(allow_single_cluster)
        self.merge_cosine_threshold = (
            float(merge_cosine_threshold) if merge_cosine_threshold is not None else None
        )
        self.min_cluster_silhouette = (
            float(min_cluster_silhouette) if min_cluster_silhouette is not None else None
        )
        self.min_cluster_cohesion = float(min_cluster_cohesion) if min_cluster_cohesion is not None else None

    def build_clustering_space(self, feats: dict[str, np.ndarray] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Construct L2-normalized clustering embeddings and 64-d reference features."""
        if isinstance(feats, np.ndarray):
            arr = np.asarray(feats, dtype=np.float32)
            return _l2_normalize(arr), arr

        if self.feature_mode == "stem_preln":
            if "stem" not in feats or "pre_ln" not in feats:
                raise KeyError("feature_mode='stem_preln' requires 'stem' and 'pre_ln' feature keys")
            stem_n = _l2_normalize(feats["stem"])
            pre_n = _l2_normalize(feats["pre_ln"])
            x_clust = _l2_normalize(np.concatenate([stem_n, pre_n], axis=1))
            ref_64 = np.asarray(feats.get("penult", feats["pre_ln"]), dtype=np.float32)
            return x_clust, ref_64

        if "penult" not in feats:
            raise KeyError("feature_mode='penult' requires 'penult' feature key")
        pen = np.asarray(feats["penult"], dtype=np.float32)
        return _l2_normalize(pen), pen

    def _select_subcluster_cores(
        self,
        x_clust: np.ndarray,
        ref_n: np.ndarray,
        labels: np.ndarray,
        probs: np.ndarray,
        raw_ids: list[int],
    ) -> tuple[dict[int, np.ndarray], dict[int, float], dict[int, np.ndarray]]:
        """Select core exemplar indices, x_clust cohesion, and unit reference centroids per sub-cluster."""
        sub_core: dict[int, np.ndarray] = {}
        sub_coh_x: dict[int, float] = {}
        sub_cents_ref: dict[int, np.ndarray] = {}

        for cid in raw_ids:
            c_idx = np.where(labels == cid)[0]
            cent_x = x_clust[c_idx].mean(axis=0, keepdims=True)
            dists = np.linalg.norm(x_clust[c_idx] - cent_x, axis=1)

            target_k = max(1, min(len(c_idx), int(math.ceil(self.core_top_ratio * len(c_idx)))))
            min_core = min(len(c_idx), max(10, self.min_cluster_size // 2))

            order = np.argsort(dists)
            top_dist_local = set(order[:target_k].tolist())
            high_prob_local = set(np.where(probs[c_idx] >= self.core_prob_threshold)[0].tolist())
            selected_local = sorted(top_dist_local & high_prob_local)

            if len(selected_local) < min_core:
                selected_local = order[: max(target_k, min_core)].tolist()

            selected_global = c_idx[np.asarray(selected_local, dtype=np.int64)]
            sub_core[cid] = selected_global

            cx_unit = cent_x[0] / max(float(np.linalg.norm(cent_x[0])), 1e-8)
            sub_coh_x[cid] = float((x_clust[selected_global] @ cx_unit).mean())

            vr = ref_n[c_idx].mean(axis=0)
            sub_cents_ref[cid] = vr / max(float(np.linalg.norm(vr)), 1e-8)

        return sub_core, sub_coh_x, sub_cents_ref

    def _evaluate_silhouette(
        self,
        eval_space: np.ndarray,
        labels: np.ndarray,
        cluster_ids_unique: list[int],
    ) -> tuple[float, dict[int, float], bool, str]:
        """Compute silhouette score, per-cluster scores, and acceptance decision."""
        k_hat = len(cluster_ids_unique)
        per_cluster_sil: dict[int, float] = {}
        if k_hat >= 2:
            valid = labels >= 0
            sil = float(silhouette_score(eval_space[valid], labels[valid]))
            sample_sil = silhouette_samples(eval_space[valid], labels[valid])
            valid_labels = labels[valid]
            for cid in cluster_ids_unique:
                per_cluster_sil[cid] = float(sample_sil[valid_labels == cid].mean())
            accepted = sil > self.silhouette_threshold
            reason = "accepted" if accepted else "silhouette_below_threshold"
            return sil, per_cluster_sil, accepted, reason

        per_cluster_sil[0] = 0.0
        return 0.0, per_cluster_sil, True, "accepted_single_cluster"

    def _merge_subclusters(
        self,
        raw_ids: list[int],
        sub_cents_ref: dict[int, np.ndarray],
    ) -> tuple[list[list[int]], bool]:
        """Group raw sub-clusters whose unit reference centroids exceed merge_cosine_threshold."""
        if self.merge_cosine_threshold is None or len(raw_ids) < 2:
            return [[cid] for cid in raw_ids], False

        parent = {cid: cid for cid in raw_ids}

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(a: int, b: int) -> None:
            ra, rb = _find(a), _find(b)
            if ra != rb:
                parent[rb] = ra

        cents_mat = np.stack([sub_cents_ref[cid] for cid in raw_ids], axis=0)
        sim = cents_mat @ cents_mat.T
        for i in range(len(raw_ids)):
            for j in range(i + 1, len(raw_ids)):
                if sim[i, j] >= self.merge_cosine_threshold:
                    _union(raw_ids[i], raw_ids[j])

        groups_map: dict[int, list[int]] = {}
        for cid in raw_ids:
            groups_map.setdefault(_find(cid), []).append(cid)
        groups = [groups_map[root] for root in sorted(groups_map.keys())]
        return groups, any(len(g) > 1 for g in groups)

    def discover(self, feats: dict[str, np.ndarray] | np.ndarray) -> DiscoveryResult:
        """Cluster buffered samples, enforce silhouette gate, and select core exemplars."""
        x_clust, ref_64 = self.build_clustering_space(feats)
        n_samples = len(x_clust)
        empty_mask = np.zeros(n_samples, dtype=bool)
        empty_proto = np.empty((0, ref_64.shape[1]), dtype=np.float32)

        if n_samples < self.min_cluster_size:
            return DiscoveryResult(
                accepted=False,
                reason="insufficient_samples",
                cluster_ids=np.full(n_samples, -1, dtype=np.int64),
                membership_probs=np.zeros(n_samples, dtype=np.float32),
                k_hat=0,
                noise_ratio=1.0,
                silhouette=-1.0,
                per_cluster_silhouette={},
                core_mask=empty_mask,
                prototypes_64d=empty_proto,
                cluster_sizes={},
                cluster_ids_unique=[],
            )

        clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric="euclidean",
            allow_single_cluster=self.allow_single_cluster,
            copy=True,
        ).fit(x_clust)

        raw_labels = np.asarray(clusterer.labels_, dtype=np.int64)
        probs = np.asarray(clusterer.probabilities_, dtype=np.float32)

        # Enforce minimum valid size per cluster and remap kept raw cluster IDs to 0 .. raw_k_hat - 1
        kept_raw_ids = [
            int(cid)
            for cid in sorted(np.unique(raw_labels).tolist())
            if cid >= 0 and int((raw_labels == cid).sum()) >= self.min_cluster_size
        ]
        raw_remapped = np.full(n_samples, -1, dtype=np.int64)
        for new_id, raw_id in enumerate(kept_raw_ids):
            raw_remapped[raw_labels == raw_id] = new_id

        raw_k_hat = len(kept_raw_ids)
        if raw_k_hat == 0:
            noise_ratio = float(1.0 - (raw_remapped >= 0).mean())
            return DiscoveryResult(
                accepted=False,
                reason="no_valid_clusters",
                cluster_ids=raw_remapped,
                membership_probs=probs,
                k_hat=0,
                noise_ratio=noise_ratio,
                silhouette=-1.0,
                per_cluster_silhouette={},
                core_mask=empty_mask,
                prototypes_64d=empty_proto,
                cluster_sizes={},
                cluster_ids_unique=[],
            )

        raw_ids = list(range(raw_k_hat))
        ref_n = _l2_normalize(ref_64)
        sub_core, sub_coh_x, sub_cents_ref = self._select_subcluster_cores(
            x_clust=x_clust, ref_n=ref_n, labels=raw_remapped, probs=probs, raw_ids=raw_ids
        )

        # Step 1: Merge over-split same-modulation sub-clusters in semantic ref space
        groups, did_merge = self._merge_subclusters(raw_ids=raw_ids, sub_cents_ref=sub_cents_ref)
        merged_labels = np.full(n_samples, -1, dtype=np.int64)
        merged_core = np.zeros(n_samples, dtype=bool)
        merged_coh_x: dict[int, float] = {}
        for gid, grp in enumerate(groups):
            for oid in grp:
                merged_labels[raw_remapped == oid] = gid
                merged_core[sub_core[oid]] = True
            tot_core = sum(len(sub_core[oid]) for oid in grp)
            merged_coh_x[gid] = sum(sub_coh_x[oid] * len(sub_core[oid]) for oid in grp) / max(tot_core, 1)

        # Evaluate post-merge silhouette in semantic ref space when sub-clusters were merged
        eval_space = ref_n if did_merge else x_clust

        # Step 2: Prune diffuse/mixed low-silhouette or low-cohesion clusters
        k_merged = len(groups)
        keep_gids: list[int] = []
        if k_merged >= 2 and (self.min_cluster_silhouette is not None or self.min_cluster_cohesion is not None):
            valid_m = merged_labels >= 0
            ssamp_m = silhouette_samples(eval_space[valid_m], merged_labels[valid_m])
            vl_m = merged_labels[valid_m]
            for gid in range(k_merged):
                s_k = float(ssamp_m[vl_m == gid].mean())
                ok_sil = self.min_cluster_silhouette is None or s_k >= self.min_cluster_silhouette
                ok_coh = self.min_cluster_cohesion is None or merged_coh_x[gid] >= self.min_cluster_cohesion
                if ok_sil and ok_coh:
                    keep_gids.append(gid)
        elif k_merged == 1 and self.min_cluster_cohesion is not None:
            if merged_coh_x[0] >= self.min_cluster_cohesion:
                keep_gids.append(0)
        else:
            keep_gids = list(range(k_merged))

        remapped_labels = np.full(n_samples, -1, dtype=np.int64)
        final_coh: dict[int, float] = {}
        kept_groups: list[list[int]] = []
        for new_id, gid in enumerate(keep_gids):
            remapped_labels[merged_labels == gid] = new_id
            final_coh[new_id] = merged_coh_x[gid]
            kept_groups.append(groups[gid])

        core_mask = merged_core & (remapped_labels >= 0)
        valid = remapped_labels >= 0
        k_hat = len(keep_gids)
        noise_ratio = float(1.0 - valid.mean())

        if k_hat == 0:
            return DiscoveryResult(
                accepted=False,
                reason="clusters_pruned_low_quality",
                cluster_ids=remapped_labels,
                membership_probs=probs,
                k_hat=0,
                noise_ratio=noise_ratio,
                silhouette=-1.0,
                per_cluster_silhouette={},
                core_mask=empty_mask,
                prototypes_64d=empty_proto,
                cluster_sizes={},
                cluster_ids_unique=[],
                cluster_cohesions={},
                raw_k_hat=raw_k_hat,
                merged_groups=[],
            )

        cluster_ids_unique = list(range(k_hat))
        cluster_sizes = {cid: int((remapped_labels == cid).sum()) for cid in cluster_ids_unique}
        sil, per_cluster_sil, accepted, reason = self._evaluate_silhouette(
            eval_space=eval_space, labels=remapped_labels, cluster_ids_unique=cluster_ids_unique
        )

        protos = np.zeros((k_hat, ref_64.shape[1]), dtype=np.float32)
        for cid in cluster_ids_unique:
            protos[cid] = ref_64[(remapped_labels == cid) & core_mask].mean(axis=0)

        return DiscoveryResult(
            accepted=accepted,
            reason=reason,
            cluster_ids=remapped_labels,
            membership_probs=probs,
            k_hat=k_hat,
            noise_ratio=noise_ratio,
            silhouette=sil,
            per_cluster_silhouette=per_cluster_sil,
            core_mask=core_mask,
            prototypes_64d=protos,
            cluster_sizes=cluster_sizes,
            cluster_ids_unique=cluster_ids_unique,
            cluster_cohesions=final_coh,
            raw_k_hat=raw_k_hat,
            merged_groups=kept_groups,
        )
