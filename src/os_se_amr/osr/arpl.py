"""Adversarial Reciprocal Point Learning (ARPL, TPAMI 2022) loss and OSR scorer."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from os_se_amr.osr.base import BaseOSRScorer


class ARPLLoss(nn.Module):
    """Canonical Reciprocal Point learning loss with learnable open-space radius.

    Reciprocal point P_c models the extra-class (non-c) space. Known samples of class c
    are pushed away from P_c via distance-dot logits while bounded inside radius R.
    """

    def __init__(
        self,
        num_classes: int,
        feat_dim: int = 64,
        weight_pl: float = 0.1,
        temp: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.feat_dim = int(feat_dim)
        self.weight_pl = float(weight_pl)
        self.temp = float(temp)

        self.points = nn.Parameter(0.1 * torch.randn(self.num_classes, self.feat_dim))
        self.radius = nn.Parameter(torch.zeros(1))

    def dist_l2(self, features: torch.Tensor) -> torch.Tensor:
        f_sq = features.pow(2).sum(dim=1, keepdim=True)
        p_sq = self.points.pow(2).sum(dim=1, keepdim=True).t()
        dist = f_sq - 2.0 * torch.matmul(features, self.points.t()) + p_sq
        return dist.clamp_min(0.0) / float(self.feat_dim)

    def dist_dot(self, features: torch.Tensor) -> torch.Tensor:
        return torch.matmul(features, self.points.t())

    def forward(
        self, features: torch.Tensor, labels: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        d_l2 = self.dist_l2(features)
        d_dot = self.dist_dot(features)
        logits = d_l2 - d_dot

        if labels is None:
            return logits, None

        loss_ce = F.cross_entropy(logits / self.temp, labels)
        p_batch = self.points[labels]
        dis_known = (features - p_batch).pow(2).mean(dim=1)
        target = torch.ones_like(dis_known)
        loss_r = F.margin_ranking_loss(
            self.radius.expand_as(dis_known),
            dis_known,
            target,
            margin=1.0,
        )
        loss = loss_ce + self.weight_pl * loss_r
        return logits, loss


class ARPLScorer(BaseOSRScorer):
    """OSR scorer wrapping ARPLLoss; uses max reciprocal-point logit as known score."""

    def __init__(self, arpl_loss: ARPLLoss) -> None:
        self.arpl_loss = arpl_loss

    def logits_and_score(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, _ = self.arpl_loss(features, labels=None)
        return logits, logits.max(dim=-1).values

    def score(self, logits: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        _, scores = self.logits_and_score(features)
        return scores
