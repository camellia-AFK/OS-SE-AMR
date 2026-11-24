"""SSD-AMC-Open-set baseline: RF contrastive augmentation, SupCon projection head & Mahalanobis scorer."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.covariance import LedoitWolf

from os_se_amr.osr.base import BaseOSRScorer


def ssd_rf_augment(
    x: torch.Tensor,
    gen: torch.Generator | None = None,
    noise_std: float = 0.03,
    max_shift: int = 4,
) -> torch.Tensor:
    """Apply modulation-preserving RF augmentations for self/supervised contrastive views."""
    out = x.clone()
    bsz = len(out)
    if bsz == 0:
        return out

    # 1. Carrier phase rotation: discrete 90-deg jumps + small continuous jitter U[-0.08, 0.08] rad
    rot_k = torch.randint(0, 4, (bsz,), device=out.device, generator=gen).to(dtype=out.dtype)
    jitter = (torch.rand(bsz, device=out.device, dtype=out.dtype, generator=gen) * 2.0 - 1.0) * 0.08
    theta = rot_k * (math.pi * 0.5) + jitter
    cos_t = torch.cos(theta).unsqueeze(-1)
    sin_t = torch.sin(theta).unsqueeze(-1)

    i_ch = out[:, 0].clone()
    q_ch = out[:, 1].clone()
    out[:, 0] = cos_t * i_ch - sin_t * q_ch
    out[:, 1] = sin_t * i_ch + cos_t * q_ch

    # 2. Mild RMS-scaled AWGN perturbation
    if noise_std > 0.0:
        rms = out.pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
        noise = torch.randn(out.shape, device=out.device, dtype=out.dtype, generator=gen)
        out = out + float(noise_std) * rms * noise

    # 3. Small random circular time shift along frame length
    if max_shift > 0:
        shifts = torch.randint(-int(max_shift), int(max_shift) + 1, (bsz,), device=out.device, generator=gen)
        for s in range(-int(max_shift), int(max_shift) + 1):
            if s == 0:
                continue
            mask = shifts == s
            if mask.any():
                out[mask] = torch.roll(out[mask], shifts=s, dims=-1)

    return out


class SSDProjectionHead(nn.Module):
    """2-layer contrastive projection MLP returning L2-normalized embeddings."""

    def __init__(self, in_dim: int = 64, hidden_dim: int = 64, proj_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.BatchNorm1d(int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), int(proj_dim)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(features), p=2, dim=-1)


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al.) over normalized multi-view projections."""

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        z = F.normalize(features, p=2, dim=-1)
        y = labels.reshape(-1)
        n = int(z.shape[0])
        if n < 2:
            return z.sum() * 0.0

        sim = torch.matmul(z, z.T) / max(self.temperature, 1e-6)
        logits_mask = ~torch.eye(n, dtype=torch.bool, device=z.device)

        # Row-wise max excluding self-diagonal for numerical stability
        sim_off = sim.masked_fill(~logits_mask, -1e9)
        sim_max = sim_off.max(dim=1, keepdim=True).values.detach()
        logits = sim - sim_max
        exp_logits = torch.exp(logits) * logits_mask.to(dtype=logits.dtype)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))

        pos_mask = (y.unsqueeze(0) == y.unsqueeze(1)) & logits_mask
        pos_count = pos_mask.sum(dim=1)
        valid = pos_count > 0
        if not valid.any():
            return z.sum() * 0.0

        mean_log_prob_pos = (pos_mask.to(dtype=logits.dtype) * log_prob).sum(dim=1) / pos_count.clamp_min(1).to(
            dtype=logits.dtype
        )
        return -mean_log_prob_pos[valid].mean()


class SSDMahalanobisScorer(BaseOSRScorer):
    """Mahalanobis detector over L2-normalized penultimate features (SSD-AMC-Open-set)."""

    def __init__(self) -> None:
        self.classes: list[int] = []
        self.means: np.ndarray | None = None
        self.precision: np.ndarray | None = None
        self._fitted = False

    @staticmethod
    def _l2_normalize_np(x: np.ndarray) -> np.ndarray:
        x64 = np.asarray(x, dtype=np.float64)
        norms = np.maximum(np.linalg.norm(x64, axis=1, keepdims=True), 1e-8)
        return x64 / norms

    def fit_numpy(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Fit class centroids and Ledoit-Wolf precision matrix in L2-normalized space."""
        x_unit = self._l2_normalize_np(features)
        y = np.asarray(labels, dtype=np.int64).reshape(-1)
        self.classes = sorted(np.unique(y).tolist())
        if not self.classes:
            raise ValueError("labels must contain at least one class")

        mus = np.stack([x_unit[y == c].mean(axis=0) for c in self.classes], axis=0)
        centered = np.concatenate([x_unit[y == c] - mus[i] for i, c in enumerate(self.classes)], axis=0)
        lw = LedoitWolf(assume_centered=True).fit(centered)
        self.means = mus
        self.precision = np.asarray(lw.precision_, dtype=np.float64)
        self._fitted = True

    def fit(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        self.fit_numpy(features.detach().cpu().numpy(), labels.detach().cpu().numpy())

    def score_numpy(self, features: np.ndarray) -> np.ndarray:
        """Return negative minimum Mahalanobis distance (higher = more Known)."""
        if not self._fitted or self.means is None or self.precision is None:
            raise RuntimeError("SSDMahalanobisScorer must be fitted before scoring")

        x_unit = self._l2_normalize_np(features)
        dists = []
        for c_idx in range(len(self.classes)):
            diff = x_unit - self.means[c_idx]
            sq = np.einsum("ij,jk,ik->i", diff, self.precision, diff)
            dists.append(np.sqrt(np.clip(sq, 0.0, None)))
        return -np.min(np.stack(dists, axis=1), axis=1)

    def score(self, logits: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        scores_np = self.score_numpy(features.detach().cpu().numpy())
        return torch.from_numpy(scores_np).to(device=features.device, dtype=features.dtype)

    def state_dict(self) -> dict[str, Any]:
        return {
            "classes": self.classes,
            "means": self.means,
            "precision": self.precision,
            "fitted": self._fitted,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.classes = list(state["classes"])
        self.means = None if state["means"] is None else np.asarray(state["means"], dtype=np.float64)
        self.precision = None if state["precision"] is None else np.asarray(state["precision"], dtype=np.float64)
        self._fitted = bool(state.get("fitted", True))
