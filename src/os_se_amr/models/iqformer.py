"""IQFormer backbone: Multi-modality I/Q + STFT Dynamic Fusion Embedding Transformer."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from os_se_amr.data.transforms import iq_to_iq_ap, iq_to_stft_mag
from os_se_amr.models.base import BaseBackbone
from os_se_amr.models.heads import ExpandableLinearHead


class DynamicFusionEmbedding(nn.Module):
    """Dynamically fuse temporal I/Q-A/P tokens with STFT spectral tokens."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Conv1d(dim * 2, dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.fuse_proj = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(dim),
            nn.GELU(),
        )

    def forward(self, time_feat: torch.Tensor, freq_feat: torch.Tensor) -> torch.Tensor:
        # Both inputs: (B, dim, L)
        gate = self.gate_net(torch.cat((time_feat, freq_feat), dim=1))
        blended = gate * time_feat + (1.0 - gate) * freq_feat
        return blended + self.fuse_proj(blended)


class ConvTransformerBlock(nn.Module):
    """Hybrid local depthwise convolution + Pre-LN self-attention + FFN block."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.local_conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=1, bias=False),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, dim)
        x = x + self.local_conv(x.transpose(1, 2)).transpose(1, 2)
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop1(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x


class IQFormer(BaseBackbone):
    """Dual-branch Time + STFT Transformer backbone exposing 64-d global-pooled features."""

    def __init__(
        self,
        num_classes: int,
        feature_dim: int = 64,
        embed_dim: int = 64,
        depth: int = 2,
        num_heads: int = 4,
        stft_n_fft: int = 32,
        stft_hop: int = 8,
        dropout: float = 0.1,
        seq_len: int = 32,
    ) -> None:
        super().__init__(num_classes=num_classes, feature_dim=feature_dim)
        self.stft_n_fft = int(stft_n_fft)
        self.stft_hop = int(stft_hop)
        self.seq_len = int(seq_len)

        self.time_stem = nn.Sequential(
            nn.Conv1d(4, embed_dim, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
        )

        self.freq_stem = nn.Sequential(
            nn.Conv2d(1, embed_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=(2, 1), padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, self.seq_len)),
        )

        self.dfe = DynamicFusionEmbedding(embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [ConvTransformerBlock(embed_dim, num_heads, dropout=dropout) for _ in range(depth)]
        )
        self.feat_proj = (
            nn.Identity() if embed_dim == feature_dim else nn.Linear(embed_dim, feature_dim)
        )
        self.feat_norm = nn.LayerNorm(feature_dim)
        self.classifier = ExpandableLinearHead(feature_dim, num_classes)

    def extract_multiscale_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return multi-scale features: stem (B, 128), pre_ln (B, 64), and penult (B, 64)."""
        time_in = iq_to_iq_ap(x)  # (B, 4, L)
        time_tokens = self.time_stem(time_in)  # (B, embed_dim, T_t)
        if time_tokens.shape[-1] != self.seq_len:
            time_tokens = F.adaptive_avg_pool1d(time_tokens, self.seq_len)

        spec = iq_to_stft_mag(x, n_fft=self.stft_n_fft, hop_length=self.stft_hop)  # (B, 1, F, T)
        freq_tokens = self.freq_stem(spec).squeeze(2)  # (B, embed_dim, 32)

        fused_c = self.dfe(time_tokens, freq_tokens)  # (B, embed_dim, 32)
        stem_stats = torch.cat([fused_c.mean(dim=-1), fused_c.std(dim=-1, unbiased=False)], dim=-1)

        tokens = fused_c.transpose(1, 2) + self.pos_embed  # (B, 32, embed_dim)
        for blk in self.blocks:
            tokens = blk(tokens)

        pre_ln = self.feat_proj(tokens.mean(dim=1))
        penult = self.feat_norm(pre_ln)
        return {"stem": stem_stats, "pre_ln": pre_ln, "penult": penult}

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.extract_multiscale_features(x)["penult"]
