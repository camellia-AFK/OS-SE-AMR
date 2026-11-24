"""Backbone factory and model interfaces."""

from __future__ import annotations

from typing import Any

from os_se_amr.models.base import BaseBackbone, BaseSNREstimator
from os_se_amr.models.heads import CosinePrototypeHead, ExpandableLinearHead
from os_se_amr.models.iqformer import IQFormer
from os_se_amr.models.resnet1d import ResNet1D
from os_se_amr.models.snr_estimator import (
    AuxSNRRegressor,
    M2M4SNREstimator,
    evaluate_snr_estimator,
    train_aux_snr_regressor,
)
from os_se_amr.models.tldnn import TLDNN


def build_backbone(model_cfg: dict[str, Any], num_classes: int) -> BaseBackbone:
    """Instantiate TLDNN, IQFormer, or ResNet1D backbone from configuration dict."""
    name = str(model_cfg.get("name", "tldnn")).lower()
    feature_dim = int(model_cfg.get("feature_dim", 64))
    dropout = float(model_cfg.get("dropout", 0.1))

    if name == "tldnn":
        sub = model_cfg.get("tldnn", {})
        return TLDNN(
            num_classes=num_classes,
            feature_dim=feature_dim,
            input_mode=str(sub.get("input_mode", "iq_ap")),
            d_model=int(sub.get("d_model", 64)),
            nhead=int(sub.get("nhead", 4)),
            num_transformer_layers=int(sub.get("num_transformer_layers", 2)),
            lstm_hidden=int(sub.get("lstm_hidden", 64)),
            lstm_layers=int(sub.get("lstm_layers", 2)),
            dropout=dropout,
            seq_len=int(sub.get("seq_len", 32)),
        )

    if name == "iqformer":
        sub = model_cfg.get("iqformer", {})
        return IQFormer(
            num_classes=num_classes,
            feature_dim=feature_dim,
            embed_dim=int(sub.get("embed_dim", 64)),
            depth=int(sub.get("depth", 2)),
            num_heads=int(sub.get("num_heads", 4)),
            stft_n_fft=int(sub.get("stft_n_fft", 32)),
            stft_hop=int(sub.get("stft_hop", 8)),
            dropout=dropout,
            seq_len=int(sub.get("seq_len", 32)),
        )

    if name == "resnet1d":
        sub = model_cfg.get("resnet1d", {})
        return ResNet1D(
            num_classes=num_classes,
            feature_dim=feature_dim,
            base_channels=int(sub.get("base_channels", 64)),
            dropout=dropout,
        )

    raise ValueError(f"Unknown backbone name '{name}'; expected 'tldnn', 'iqformer', or 'resnet1d'")


__all__ = [
    "AuxSNRRegressor",
    "BaseBackbone",
    "BaseSNREstimator",
    "CosinePrototypeHead",
    "ExpandableLinearHead",
    "IQFormer",
    "M2M4SNREstimator",
    "ResNet1D",
    "TLDNN",
    "build_backbone",
    "evaluate_snr_estimator",
    "train_aux_snr_regressor",
]
