"""Unit tests for signal transforms, TLDNN, IQFormer, and classifier expansion."""

from __future__ import annotations

import pytest
import torch

from os_se_amr.data.transforms import iq_to_ap, iq_to_iq_ap, iq_to_stft_mag, normalize_iq_rms
from os_se_amr.models import IQFormer, TLDNN, build_backbone


def test_signal_transforms() -> None:
    x = torch.randn(8, 2, 128) * 5.0
    x_norm = normalize_iq_rms(x)
    power = (x_norm[:, 0] ** 2 + x_norm[:, 1] ** 2).mean(dim=-1)
    assert torch.allclose(power, torch.ones_like(power), atol=1e-5)

    ap = iq_to_ap(x)
    assert ap.shape == (8, 2, 128)
    assert torch.all(ap[:, 1] >= -1.0 - 1e-6)
    assert torch.all(ap[:, 1] <= 1.0 + 1e-6)

    iq_ap = iq_to_iq_ap(x)
    assert iq_ap.shape == (8, 4, 128)

    spec = iq_to_stft_mag(x, n_fft=32, hop_length=8)
    assert spec.ndim == 4
    assert spec.shape[:3] == (8, 1, 32)
    assert torch.isfinite(spec).all()


@pytest.mark.parametrize("model_name", ["tldnn", "iqformer"])
def test_backbone_forward_backward_and_expansion(model_name: str) -> None:
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    cfg = {
        "name": model_name,
        "feature_dim": 64,
        "dropout": 0.1,
        "tldnn": {"input_mode": "iq_ap", "d_model": 64, "nhead": 4, "num_transformer_layers": 2, "lstm_hidden": 64, "lstm_layers": 2},
        "iqformer": {"embed_dim": 64, "depth": 2, "num_heads": 4, "stft_n_fft": 32, "stft_hop": 8},
    }

    for dev in devices:
        model = build_backbone(cfg, num_classes=7).to(dev)
        x = torch.randn(6, 2, 128, device=dev)

        # Forward + backward pass
        model.train()
        logits, feats = model(x)
        assert logits.shape == (6, 7)
        assert feats.shape == (6, 64)
        assert torch.isfinite(logits).all()
        assert torch.isfinite(feats).all()

        ms_feats = model.extract_multiscale_features(x)
        assert set(ms_feats.keys()) == {"stem", "pre_ln", "penult"}
        assert ms_feats["stem"].shape == (6, 128)
        assert ms_feats["pre_ln"].shape == (6, 64)
        assert ms_feats["penult"].shape == (6, 64)
        assert all(torch.isfinite(v).all() for v in ms_feats.values())
        model.eval()
        with torch.no_grad():
            assert torch.allclose(
                model.extract_multiscale_features(x)["penult"],
                model.extract_features(x),
                atol=1e-6,
            )
        model.train()

        loss = logits.sum() + feats.pow(2).mean()
        loss.backward()
        assert any(p.grad is not None for p in model.parameters())

        # Weight-preserving head expansion 7 -> 9
        model.eval()
        with torch.no_grad():
            old_logits, old_feats = model(x)
            model.expand_classifier(2)
            assert model.num_classes == 9
            new_logits, new_feats = model(x)

        assert new_logits.shape == (6, 9)
        assert new_logits.device == old_logits.device
        assert torch.allclose(new_feats, old_feats, atol=1e-6)
        assert torch.allclose(new_logits[:, :7], old_logits, atol=1e-6)
