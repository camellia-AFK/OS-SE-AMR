"""Additional static OSR baselines: Energy Score (NeurIPS 2020) and OpenMax (CVPR 2016)."""

from __future__ import annotations

import numpy as np
from scipy.stats import weibull_min
import torch

from os_se_amr.osr.base import BaseOSRScorer


class EnergyScorer(BaseOSRScorer):
    """Free-energy OSR scorer (Liu et al., NeurIPS 2020).

    Higher free-energy negative score ``T * logsumexp(logits / T)`` indicates Known.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = float(temperature)

    def score(self, logits: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        t = self.temperature
        return t * torch.logsumexp(logits / t, dim=-1)


class OpenMaxScorer(BaseOSRScorer):
    """Weibull-calibrated OpenMax scorer (Bendale & Boult, CVPR 2016).

    Polarity invariant: returns ``1 - P(unknown)`` so higher scores indicate Known.
    """

    def __init__(
        self,
        tailsize: int = 20,
        alpha: int = 3,
        euclid_weight: float = 0.5,
    ) -> None:
        self.tailsize = int(tailsize)
        self.alpha = int(alpha)
        self.euclid_weight = float(euclid_weight)
        self.classes: list[int] = []
        self.mavs: np.ndarray | None = None
        self.weibull_params: dict[int, tuple[float, float, float]] = {}

    def _compute_distance(self, vecs: np.ndarray, mav: np.ndarray) -> np.ndarray:
        v64 = np.asarray(vecs, dtype=np.float64)
        m64 = np.asarray(mav, dtype=np.float64)
        euc = np.linalg.norm(v64 - m64[None, :], axis=1)
        v_norm = np.maximum(np.linalg.norm(v64, axis=1), 1e-8)
        m_norm = max(float(np.linalg.norm(m64)), 1e-8)
        cos_d = 1.0 - np.clip((v64 @ m64) / (v_norm * m_norm), -1.0, 1.0)
        return self.euclid_weight * euc + (1.0 - self.euclid_weight) * cos_d

    def fit_logits(
        self,
        train_logits: np.ndarray | torch.Tensor,
        train_labels: np.ndarray | torch.Tensor,
    ) -> None:
        """Fit class MAVs and extreme-value Weibull tails strictly on known Train activations."""
        act = (
            train_logits.detach().cpu().numpy()
            if torch.is_tensor(train_logits)
            else np.asarray(train_logits, dtype=np.float64)
        ).astype(np.float64)
        y = (
            train_labels.detach().cpu().numpy()
            if torch.is_tensor(train_labels)
            else np.asarray(train_labels, dtype=np.int64)
        ).reshape(-1)

        self.classes = sorted(np.unique(y).tolist())
        if not self.classes:
            raise ValueError("train_labels must not be empty")

        preds = act.argmax(axis=1)
        mavs = []
        self.weibull_params = {}
        for pos, cid in enumerate(self.classes):
            mask = y == cid
            corr = mask & (preds == cid)
            cls_vecs = act[corr] if corr.any() else act[mask]
            mav = cls_vecs.mean(axis=0)
            mavs.append(mav)

            dists = np.sort(self._compute_distance(cls_vecs, mav))
            t_len = max(3, min(self.tailsize, len(dists)))
            tail = np.maximum(dists[-t_len:], 1e-6)
            if float(np.std(tail)) < 1e-7:
                # Guard against degenerate zero-variance synthetic inputs
                self.weibull_params[pos] = (2.0, 0.0, float(np.mean(tail) * 1.5))
            else:
                try:
                    c_shape, loc, scale = weibull_min.fit(tail, floc=0.0)
                    self.weibull_params[pos] = (float(max(c_shape, 0.1)), float(loc), float(max(scale, 1e-4)))
                except Exception:
                    self.weibull_params[pos] = (2.0, 0.0, float(max(np.mean(tail), 1e-3)))

        self.mavs = np.stack(mavs, axis=0)

    def fit(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        """Fallback fitting when called through generic BaseOSRScorer.fit(logits_or_feats, y)."""
        self.fit_logits(features, labels)

    def score(self, logits: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        if self.mavs is None:
            raise RuntimeError("OpenMaxScorer must be fitted with fit_logits() before scoring")

        act = logits.detach().cpu().numpy().astype(np.float64)
        num_cls = act.shape[1]
        if num_cls != len(self.classes):
            # Fallback if fit() was inadvertently called on 64-d features instead of logits
            act = features.detach().cpu().numpy().astype(np.float64)
            num_cls = act.shape[1]

        top_a = max(1, min(self.alpha, num_cls))
        dists = np.stack([self._compute_distance(act, self.mavs[c]) for c in range(len(self.classes))], axis=1)

        order = np.argsort(-act, axis=1)
        v_hat = act.copy()
        v_unk = np.zeros(len(act), dtype=np.float64)

        for r_idx in range(top_a):
            weight_mult = float(top_a - r_idx) / float(top_a)
            cls_idx = order[:, r_idx]
            w_rev = np.zeros(len(act), dtype=np.float64)
            for pos in np.unique(cls_idx).tolist():
                sel = cls_idx == pos
                c_shape, loc, scale = self.weibull_params[int(pos)]
                cdf = weibull_min.cdf(dists[sel, int(pos)], c_shape, loc=loc, scale=scale)
                w_rev[sel] = weight_mult * np.clip(cdf, 0.0, 1.0)
            row_idx = np.arange(len(act))
            raw_top = act[row_idx, cls_idx]
            v_unk += raw_top * w_rev
            v_hat[row_idx, cls_idx] = raw_top * (1.0 - w_rev)

        openmax_logits = np.concatenate([v_hat, v_unk[:, None]], axis=1)
        openmax_logits -= openmax_logits.max(axis=1, keepdims=True)
        exp_l = np.exp(openmax_logits)
        probs = exp_l / np.maximum(exp_l.sum(axis=1, keepdims=True), 1e-12)
        known_score = 1.0 - probs[:, -1]
        return torch.from_numpy(known_score).to(device=logits.device, dtype=logits.dtype)
