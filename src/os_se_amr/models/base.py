"""Abstract base interfaces for OS-SE-AMR backbones and blind SNR estimators."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseBackbone(nn.Module, ABC):
    """Unified backbone contract exposing logits and 64-d standardized embeddings."""

    feature_dim: int = 64

    def __init__(self, num_classes: int, feature_dim: int = 64) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)

    @abstractmethod
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract latent feature representation z of shape (B, feature_dim) from raw I/Q (B, 2, 128)."""

    def extract_multiscale_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract intermediate and penultimate representations for multi-scale OSR."""
        return {"penult": self.extract_features(x)}

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (logits[B, C_current], features[B, feature_dim])."""
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits, features

    def expand_classifier(
        self,
        num_new_classes: int,
        init_weights: torch.Tensor | None = None,
    ) -> None:
        """Atomically expand classifier head without altering weights for classes 0..C_old-1."""
        if not hasattr(self, "classifier") or not hasattr(self.classifier, "expand"):
            raise AttributeError("Backbone must define an expandable self.classifier head")
        self.classifier.expand(num_new_classes=num_new_classes, init_weights=init_weights)
        self.num_classes += int(num_new_classes)


class BaseSNREstimator(ABC):
    """Unified interface for blind SNR estimators (Stage 4 dual-gating contract)."""

    @abstractmethod
    def predict_snr(self, x_iq: torch.Tensor) -> torch.Tensor:
        """Estimate per-sample SNR in dB from raw I/Q frames (B, 2, 128) -> (B,)."""
