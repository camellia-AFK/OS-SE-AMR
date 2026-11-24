"""Maximum Logit Score (MLS, ICLR 2022) OSR baseline."""

from __future__ import annotations

import torch

from os_se_amr.osr.base import BaseOSRScorer


class MLSScorer(BaseOSRScorer):
    """Maximum unnormalized Logit Score: max_c g_c(x)."""

    def score(self, logits: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return logits.max(dim=-1).values
