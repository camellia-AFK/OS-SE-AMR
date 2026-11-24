"""Expandable classification heads supporting incremental class additions."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpandableLinearHead(nn.Module):
    """Linear classifier supporting weight-preserving class-head expansion."""

    def __init__(self, in_features: int, num_classes: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.num_classes = int(num_classes)
        self.use_bias = bool(bias)
        self.fc = nn.Linear(self.in_features, self.num_classes, bias=self.use_bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)

    @torch.no_grad()
    def expand(
        self,
        num_new_classes: int,
        init_weights: torch.Tensor | None = None,
    ) -> None:
        """Add `num_new_classes` output neurons while preserving existing weights verbatim."""
        if num_new_classes <= 0:
            raise ValueError(f"num_new_classes must be positive, got {num_new_classes}")

        old_fc = self.fc
        old_c = self.num_classes
        new_c = old_c + int(num_new_classes)
        device = old_fc.weight.device
        dtype = old_fc.weight.dtype

        new_fc = nn.Linear(self.in_features, new_c, bias=self.use_bias, device=device, dtype=dtype)
        new_fc.weight[:old_c].copy_(old_fc.weight)

        if init_weights is not None:
            expected_shape = (num_new_classes, self.in_features)
            if tuple(init_weights.shape) != expected_shape:
                raise ValueError(f"init_weights shape {tuple(init_weights.shape)} != {expected_shape}")
            new_fc.weight[old_c:].copy_(init_weights.to(device=device, dtype=dtype))

        if self.use_bias and old_fc.bias is not None and new_fc.bias is not None:
            new_fc.bias[:old_c].copy_(old_fc.bias)
            if init_weights is not None:
                new_fc.bias[old_c:].zero_()

        self.fc = new_fc
        self.num_classes = new_c


class CosinePrototypeHead(nn.Module):
    """L2-normalized cosine similarity head with learnable temperature scale."""

    def __init__(self, in_features: int, num_classes: int, init_scale: float = 16.0) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.num_classes = int(num_classes)
        self.weight = nn.Parameter(torch.empty(self.num_classes, self.in_features))
        nn.init.xavier_uniform_(self.weight)
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        z_norm = F.normalize(features, p=2, dim=-1)
        w_norm = F.normalize(self.weight, p=2, dim=-1)
        return self.scale * F.linear(z_norm, w_norm)

    @torch.no_grad()
    def expand(
        self,
        num_new_classes: int,
        init_weights: torch.Tensor | None = None,
    ) -> None:
        """Expand cosine prototype matrix without altering existing class prototypes."""
        if num_new_classes <= 0:
            raise ValueError(f"num_new_classes must be positive, got {num_new_classes}")

        old_c = self.num_classes
        new_c = old_c + int(num_new_classes)
        device = self.weight.device
        dtype = self.weight.dtype

        new_weight = torch.empty((new_c, self.in_features), device=device, dtype=dtype)
        nn.init.xavier_uniform_(new_weight)
        new_weight[:old_c].copy_(self.weight)

        if init_weights is not None:
            expected_shape = (num_new_classes, self.in_features)
            if tuple(init_weights.shape) != expected_shape:
                raise ValueError(f"init_weights shape {tuple(init_weights.shape)} != {expected_shape}")
            new_weight[old_c:].copy_(init_weights.to(device=device, dtype=dtype))

        self.weight = nn.Parameter(new_weight)
        self.num_classes = new_c
