"""ResNet-1D backbone family with Squeeze-and-Excitation and length-agnostic pooling."""

from __future__ import annotations

import torch
import torch.nn as nn

from os_se_amr.data.transforms import iq_to_iq_ap
from os_se_amr.models.base import BaseBackbone
from os_se_amr.models.heads import ExpandableLinearHead
from os_se_amr.models.tldnn import SEBlock1d


class ResidualBlock1d(nn.Module):
    """Post-BN 1D residual block with channel Squeeze-and-Excitation."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5, stride: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, stride=1, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.se = SEBlock1d(out_ch, reduction=4)
        self.shortcut = (
            nn.Identity()
            if (in_ch == out_ch and stride == 1)
            else nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop(self.act(self.bn1(self.conv1(x))))
        out = self.se(self.bn2(self.conv2(out)))
        return self.act(out + self.shortcut(x))


class ResNet1D(BaseBackbone):
    """Multi-scale 1D Residual SE-CNN AMR backbone compatible with L=128 and L=1024."""

    def __init__(
        self,
        num_classes: int,
        feature_dim: int = 64,
        base_channels: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(num_classes=num_classes, feature_dim=feature_dim)
        d = int(base_channels)
        self.stem = nn.Sequential(
            nn.Conv1d(4, d, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(d),
            nn.GELU(),
            SEBlock1d(d, reduction=4),
            nn.Conv1d(d, d, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(d),
            nn.GELU(),
            SEBlock1d(d, reduction=4),
        )
        self.stage1 = ResidualBlock1d(d, d, kernel_size=5, stride=1, dropout=dropout)
        self.stage2 = ResidualBlock1d(d, d * 2, kernel_size=5, stride=2, dropout=dropout)
        self.stage3 = ResidualBlock1d(d * 2, d * 2, kernel_size=3, stride=2, dropout=dropout)
        self.feat_proj = nn.Linear(d * 4, feature_dim)
        self.feat_norm = nn.LayerNorm(feature_dim)
        self.classifier = ExpandableLinearHead(feature_dim, num_classes)

    def extract_multiscale_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return multi-scale features: stem (B, 2*d), pre_ln (B, 64), and penult (B, 64)."""
        x_in = iq_to_iq_ap(x)
        st = self.stem(x_in)
        stem_stats = torch.cat([st.mean(dim=-1), st.std(dim=-1, unbiased=False)], dim=-1)

        h = self.stage3(self.stage2(self.stage1(st)))
        pooled = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=-1)
        pre_ln = self.feat_proj(pooled)
        penult = self.feat_norm(pre_ln)
        return {"stem": stem_stats, "pre_ln": pre_ln, "penult": penult}

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.extract_multiscale_features(x)["penult"]
