"""Intra-class Distance-Prototype loss and OSR scorer (reusable in Stages 2 & 4)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from os_se_amr.osr.base import BaseOSRScorer


class DistancePrototypeLoss(nn.Module):
    """Learnable intra-class prototype centroids with compactness regularization."""

    def __init__(
        self,
        num_classes: int,
        feat_dim: int = 64,
        lambda_proto: float = 0.1,
        temp: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.feat_dim = int(feat_dim)
        self.lambda_proto = float(lambda_proto)
        self.temp = float(temp)
        self.prototypes = nn.Parameter(0.1 * torch.randn(self.num_classes, self.feat_dim))

    def dist_sq(self, features: torch.Tensor) -> torch.Tensor:
        """Compute per-dimension mean squared Euclidean distance (B, C) to all prototypes."""
        f_sq = features.pow(2).sum(dim=1, keepdim=True)
        p_sq = self.prototypes.pow(2).sum(dim=1, keepdim=True).t()
        dist = f_sq - 2.0 * torch.matmul(features, self.prototypes.t()) + p_sq
        return dist.clamp_min(0.0) / float(self.feat_dim)

    def forward(
        self, features: torch.Tensor, labels: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        d2 = self.dist_sq(features)
        logits = -d2 / self.temp

        if labels is None:
            return logits, None

        loss_ce = F.cross_entropy(logits, labels)
        d2_true = d2.gather(1, labels.unsqueeze(1)).squeeze(1)
        loss = loss_ce + self.lambda_proto * d2_true.mean()
        return logits, loss

    @torch.no_grad()
    def expand(
        self,
        num_new_classes: int,
        init_prototypes: torch.Tensor | None = None,
    ) -> None:
        """Append new prototype vectors while preserving existing prototypes 0..C_old-1."""
        if num_new_classes <= 0:
            raise ValueError(f"num_new_classes must be positive, got {num_new_classes}")

        old_c = self.num_classes
        new_c = old_c + int(num_new_classes)
        device = self.prototypes.device
        dtype = self.prototypes.dtype

        new_proto = torch.empty((new_c, self.feat_dim), device=device, dtype=dtype)
        new_proto[:old_c].copy_(self.prototypes)
        if init_prototypes is not None:
            expected_shape = (num_new_classes, self.feat_dim)
            if tuple(init_prototypes.shape) != expected_shape:
                raise ValueError(f"init_prototypes shape {tuple(init_prototypes.shape)} != {expected_shape}")
            new_proto[old_c:].copy_(init_prototypes.to(device=device, dtype=dtype))
        else:
            new_proto[old_c:].normal_(mean=0.0, std=0.1)

        self.prototypes = nn.Parameter(new_proto)
        self.num_classes = new_c

    @torch.no_grad()
    def refresh_prototypes(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        """Recompute prototypes for classes present in (features, labels) to remove drift."""
        device = self.prototypes.device
        dtype = self.prototypes.dtype
        feats = features.to(device=device, dtype=dtype)
        lbls = labels.to(device=device)
        for c in range(self.num_classes):
            mask = lbls == c
            if mask.any():
                self.prototypes[c].copy_(feats[mask].mean(dim=0))


class DistancePrototypeScorer(BaseOSRScorer):
    """OSR scorer using negative minimum Euclidean distance to learned class prototypes."""

    def __init__(self, proto_loss: DistancePrototypeLoss) -> None:
        self.proto_loss = proto_loss

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        return self.proto_loss.dist_sq(features).argmin(dim=-1)

    def logits_and_score(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        d2 = self.proto_loss.dist_sq(features)
        logits = -d2 / self.proto_loss.temp
        min_dist = torch.sqrt(d2.min(dim=-1).values.clamp_min(1e-12))
        return logits, -min_dist

    def score(self, logits: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        _, scores = self.logits_and_score(features)
        return scores
