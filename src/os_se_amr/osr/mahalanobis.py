"""Multi-Scale Covariance-Aware Mahalanobis Prototype scorer (MS-MahaProto)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.covariance import LedoitWolf

from os_se_amr.osr.base import BaseOSRScorer


class MahalanobisPrototypeScorer(BaseOSRScorer):
    """Class-prototype Mahalanobis scorer using shrinkage covariance over multi-scale features.

    Polarity invariant: returns negative weighted standardized minimum Mahalanobis distance,
    so higher scores indicate closer proximity to known class manifolds.
    """

    def __init__(
        self,
        layers: tuple[str, ...] = ("stem", "pre_ln"),
        weights: dict[str, float] | None = None,
    ) -> None:
        if not layers:
            raise ValueError("layers must be a non-empty tuple/list of feature layer names")
        self.layers = tuple(str(layer) for layer in layers)
        if weights is None:
            w = 1.0 / float(len(self.layers))
            self.weights = {layer: w for layer in self.layers}
        else:
            self.weights = {layer: float(weights.get(layer, 1.0)) for layer in self.layers}

        self.classes: list[int] = []
        self.means: dict[str, np.ndarray] = {}
        self.precisions: dict[str, np.ndarray] = {}
        self.dist_stats: dict[str, tuple[float, float]] = {}
        self._fitted = False

    def _layer_min_dist(self, layer: str, x: np.ndarray) -> np.ndarray:
        mus = self.means[layer]
        prec = self.precisions[layer]
        x64 = np.asarray(x, dtype=np.float64)
        dists = []
        for c_idx in range(len(self.classes)):
            diff = x64 - mus[c_idx]
            sq = np.einsum("ij,jk,ik->i", diff, prec, diff)
            dists.append(np.sqrt(np.clip(sq, 0.0, None)))
        return np.min(np.stack(dists, axis=1), axis=1)

    def fit_multiscale(
        self,
        train_feats: dict[str, np.ndarray],
        train_labels: np.ndarray,
        val_feats: dict[str, np.ndarray] | None = None,
    ) -> None:
        """Fit class centroids and shrinkage precision matrices strictly on known Train (+ Val stats)."""
        y = np.asarray(train_labels, dtype=np.int64).reshape(-1)
        self.classes = sorted(np.unique(y).tolist())
        if not self.classes:
            raise ValueError("train_labels must contain at least one class")

        self.means = {}
        self.precisions = {}
        self.dist_stats = {}

        for layer in self.layers:
            if layer not in train_feats:
                raise KeyError(f"Layer '{layer}' missing in train_feats (available: {list(train_feats.keys())})")
            x_tr = np.asarray(train_feats[layer], dtype=np.float64)
            mus = np.stack([x_tr[y == c].mean(axis=0) for c in self.classes], axis=0)
            centered = np.concatenate([x_tr[y == c] - mus[i] for i, c in enumerate(self.classes)], axis=0)
            lw = LedoitWolf(assume_centered=True).fit(centered)

            self.means[layer] = mus
            self.precisions[layer] = np.asarray(lw.precision_, dtype=np.float64)

            ref_x = (
                np.asarray(val_feats[layer], dtype=np.float64)
                if (val_feats is not None and layer in val_feats)
                else x_tr
            )
            ref_dists = self._layer_min_dist(layer, ref_x)
            d_mean = float(ref_dists.mean())
            d_std = float(max(ref_dists.std(), 1e-8))
            self.dist_stats[layer] = (d_mean, d_std)

        self._fitted = True

    def fit(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        """Single-layer BaseOSRScorer convenience calibration."""
        feats_np = features.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        self.fit_multiscale({self.layers[0]: feats_np}, labels_np)

    def score_multiscale(self, feats: dict[str, np.ndarray]) -> np.ndarray:
        """Return normality scores (higher = closer to known class prototypes)."""
        if not self._fitted:
            raise RuntimeError("MahalanobisPrototypeScorer must be fitted before scoring")

        total_z: np.ndarray | None = None
        for layer in self.layers:
            if layer not in feats:
                raise KeyError(f"Layer '{layer}' missing in input feats")
            d = self._layer_min_dist(layer, feats[layer])
            d_mean, d_std = self.dist_stats[layer]
            z = (d - d_mean) / max(d_std, 1e-8)
            contrib = self.weights[layer] * z
            total_z = contrib if total_z is None else (total_z + contrib)

        assert total_z is not None
        return -total_z

    def score(self, logits: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Single-layer BaseOSRScorer fallback using the first configured layer."""
        feats_np = features.detach().cpu().numpy()
        scores_np = self.score_multiscale({self.layers[0]: feats_np})
        return torch.from_numpy(scores_np).to(device=features.device, dtype=features.dtype)

    def state_dict(self) -> dict[str, Any]:
        """Export fitted prototypes, precision matrices, and distance normalizers."""
        return {
            "layers": self.layers,
            "weights": self.weights,
            "classes": self.classes,
            "means": self.means,
            "precisions": self.precisions,
            "dist_stats": self.dist_stats,
            "fitted": self._fitted,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore fitted scorer state from checkpoint dict."""
        self.layers = tuple(state["layers"])
        self.weights = dict(state["weights"])
        self.classes = list(state["classes"])
        self.means = {k: np.asarray(v, dtype=np.float64) for k, v in state["means"].items()}
        self.precisions = {k: np.asarray(v, dtype=np.float64) for k, v in state["precisions"].items()}
        self.dist_stats = {k: (float(v[0]), float(v[1])) for k, v in state["dist_stats"].items()}
        self._fitted = bool(state.get("fitted", True))
