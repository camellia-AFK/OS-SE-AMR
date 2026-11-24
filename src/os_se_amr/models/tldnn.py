"""TLDNN backbone: Conv-SE stem + Transformer encoder + LSTM temporal head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from os_se_amr.data.transforms import iq_to_ap, iq_to_iq_ap, normalize_iq_rms
from os_se_amr.models.base import BaseBackbone
from os_se_amr.models.heads import ExpandableLinearHead


class SEBlock1d(nn.Module):
    """1D Squeeze-and-Excitation channel attention block."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(4, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=True),
            nn.GELU(),
            nn.Linear(hidden, channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        scale = self.pool(x).squeeze(-1)
        scale = self.fc(scale).unsqueeze(-1)
        return x * scale


class TLDNN(BaseBackbone):
    """Hybrid Conv-SE + Transformer + LSTM backbone tailored for AMR.

    Accepts raw I/Q waveforms (B, 2, 128) and produces (logits[B, C], features[B, 64]).
    """

    def __init__(
        self,
        num_classes: int,
        feature_dim: int = 64,
        input_mode: str = "iq_ap",
        d_model: int = 64,
        nhead: int = 4,
        num_transformer_layers: int = 2,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        seq_len: int = 32,
    ) -> None:
        super().__init__(num_classes=num_classes, feature_dim=feature_dim)

        if input_mode not in ("iq_ap", "ap", "iq"):
            raise ValueError(f"Unsupported input_mode='{input_mode}'; choose from 'iq_ap', 'ap', 'iq'")
        self.input_mode = input_mode
        in_channels = 4 if input_mode == "iq_ap" else 2

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            SEBlock1d(d_model, reduction=4),
            nn.Conv1d(d_model, d_model, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            SEBlock1d(d_model, reduction=4),
        )

        # Length downsamples 128 -> 64 -> 32 (or adaptive pool to seq_len for long frames e.g. L=1024)
        self.seq_len = int(seq_len)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers,
            enable_nested_tensor=False,
        )

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.feat_proj = (
            nn.Identity() if lstm_hidden == feature_dim else nn.Linear(lstm_hidden, feature_dim)
        )
        self.feat_norm = nn.LayerNorm(feature_dim)
        self.classifier = ExpandableLinearHead(feature_dim, num_classes)

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_mode == "iq_ap":
            return iq_to_iq_ap(x)
        if self.input_mode == "ap":
            return iq_to_ap(x)
        return normalize_iq_rms(x)

    def extract_multiscale_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return multi-scale features: stem (B, 128), pre_ln (B, 64), and penult (B, 64)."""
        x_in = self._prepare_input(x)  # (B, C_in, 128)
        st = self.stem(x_in)  # (B, d_model, L/4)
        stem_stats = torch.cat([st.mean(dim=-1), st.std(dim=-1, unbiased=False)], dim=-1)
        if st.shape[-1] != self.seq_len:
            st = F.adaptive_avg_pool1d(st, self.seq_len)

        tokens = st.transpose(1, 2) + self.pos_embed  # (B, seq_len, d_model)
        tokens = self.transformer(tokens)
        lstm_out, _ = self.lstm(tokens)
        pre_ln = self.feat_proj(lstm_out[:, -1, :])
        penult = self.feat_norm(pre_ln)
        return {"stem": stem_stats, "pre_ln": pre_ln, "penult": penult}

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.extract_multiscale_features(x)["penult"]
