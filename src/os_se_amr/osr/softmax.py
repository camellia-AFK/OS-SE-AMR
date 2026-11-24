"""Softmax Maximum Probability (MSP) OSR baseline."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from os_se_amr.osr.base import BaseOSRScorer


class SoftmaxScorer(BaseOSRScorer):
    """Maximum Softmax Probability score: max_c softmax(logits)_c."""

    def score(self, logits: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits, dim=-1).max(dim=-1).values
