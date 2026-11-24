"""Abstract base class for Open-Set Recognition (OSR) scorers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BaseOSRScorer(ABC):
    """Unified OSR scorer interface.

    Polarity invariant: higher score means higher confidence that the sample belongs
    to a Known class; lower score means more likely Unknown.
    """

    def fit(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        """Optional calibration or prototype fitting on training features."""
        return None

    @abstractmethod
    def score(self, logits: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Return normality score of shape (B,) from classifier logits and latent features."""
