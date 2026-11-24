"""Unit tests for Stage 2 Hungarian matching, CIL metrics, ReplayBuffer, HDBSCAN, RF-Aug, and BiC."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from os_se_amr.continual import (
    DualLwFLearner,
    ReplayBuffer,
    calibrate_per_head_bias,
    rf_augment_batch,
)
from os_se_amr.discovery import HDBSCANDiscoverer, RejectionBuffer
from os_se_amr.eval import (
    apply_locked_mapping,
    evaluate_cil_session,
    evaluate_clustering_diagnostics,
    locked_known_hungarian_match,
)
from os_se_amr.models import TLDNN


def test_locked_known_hungarian_and_cil_metrics() -> None:
    y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4])
    y_pred = np.array([0, 1, 1, 1, 2, 2, 4, 4, 4, 3, 3, 4])

    match = locked_known_hungarian_match(y_true, y_pred, num_known=3)
    assert match["head_to_true"] == {3: 4, 4: 3}
    assert match["delta_k"] == 0

    mapped = apply_locked_mapping(y_pred, num_known=3, head_to_true=match["head_to_true"])
    assert np.array_equal(mapped[:6], y_pred[:6])

    cil = evaluate_cil_session(
        y_true=y_true,
        y_pred=y_pred,
        num_base_classes=3,
        current_true_classes=[0, 1, 2, 3, 4],
        base_acc_0=1.0,
        prev_per_class_accs=[{0: 1.0, 1: 1.0, 2: 1.0}],
    )
    assert cil["a_old"] == pytest.approx(5.0 / 6.0)
    assert cil["a_new"] == pytest.approx(5.0 / 6.0)
    assert cil["delta_a_old"] == pytest.approx(1.0 / 6.0)
    assert cil["fm_cls"] == pytest.approx((0.5 + 0.0 + 0.0) / 3.0)

    # Under-clustering: 2 novel classes, only 1 predicted novel head (head 3)
    y_under_pred = np.array([0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3])
    cil_under = evaluate_cil_session(y_true, y_under_pred, num_base_classes=3)
    assert cil_under["a_new"] == pytest.approx(0.5)

    # Over-clustering: 3 novel heads predicted for 2 true classes (head 5 unmatched -> -1)
    y_over_pred = np.array([0, 0, 1, 1, 2, 2, 3, 3, 5, 4, 4, 4])
    cil_over = evaluate_cil_session(y_true, y_over_pred, num_base_classes=3)
    assert len(cil_over["head_to_true"]) == 2
    assert cil_over["a_new"] == pytest.approx(5.0 / 6.0)


def test_hdbscan_discoverer_and_silhouette_gate() -> None:
    rng = np.random.default_rng(42)
    c1_stem = rng.normal(3.0, 0.05, size=(70, 128)).astype(np.float32)
    c1_pre = rng.normal(3.0, 0.05, size=(70, 64)).astype(np.float32)
    c2_stem = rng.normal(-3.0, 0.05, size=(70, 128)).astype(np.float32)
    c2_pre = rng.normal(-3.0, 0.05, size=(70, 64)).astype(np.float32)
    c2_stem[:, 64:] *= -1.0

    feats = {
        "stem": np.vstack([c1_stem, c2_stem]),
        "pre_ln": np.vstack([c1_pre, c2_pre]),
        "penult": np.vstack([c1_pre, c2_pre]),
    }
    y_true = np.array([0] * 70 + [1] * 70)

    discoverer = HDBSCANDiscoverer(min_cluster_size=30, min_samples=10, silhouette_threshold=0.45)
    res = discoverer.discover(feats)
    assert res.accepted is True
    assert res.k_hat == 2
    assert res.silhouette > 0.45
    assert res.prototypes_64d.shape == (2, 64)
    assert res.core_mask.any()

    diag = evaluate_clustering_diagnostics(y_true, res.cluster_ids)
    assert diag["strict_1to1_acc"] == pytest.approx(1.0)

    buf = RejectionBuffer()
    buf.add(np.zeros((10, 2, 128), dtype=np.float32))
    assert len(buf) == 10
    assert buf.get_waveforms().shape == (10, 2, 128)
    buf.clear()
    assert len(buf) == 0


def test_replay_herding_snr_stratified_rf_aug_and_bic() -> None:
    rng = np.random.default_rng(7)
    x0 = rng.normal(0.0, 1.0, size=(40, 2, 128)).astype(np.float32)
    # Tag first 20 with SNR=0 and next 20 with SNR=10 in waveform marker
    x0[:20, 0, 0] = 100.0
    x0[20:, 0, 0] = -100.0
    f0 = rng.normal(1.0, 0.1, size=(40, 64)).astype(np.float32)
    snrs0 = np.array([0] * 20 + [10] * 20, dtype=np.int64)

    rb = ReplayBuffer(exemplars_per_class=10, selection_strategy="herding")
    rb.add_class_exemplars(0, x0, f0, snrs=snrs0)
    tx, ty = rb.as_tensors()
    assert tx.shape == (10, 2, 128) and ty.shape == (10,)
    # Must select exactly 5 from SNR=0 (+100 marker) and 5 from SNR=10 (-100 marker)
    assert int((tx[:, 0, 0] > 0).sum().item()) == 5
    assert int((tx[:, 0, 0] < 0).sum().item()) == 5

    # RF augmentation shape and finiteness
    aug = rf_augment_batch(tx, ty)
    assert aug.shape == tx.shape and torch.isfinite(aug).all()
    assert not torch.allclose(aug, tx)

    # Per-head BiC calibration check: head 2 collides with base head 0, head 3 is orthogonal
    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = torch.nn.Linear(4, 4, bias=True)
            with torch.no_grad():
                self.fc.weight.zero_()
                self.fc.weight[0, 0] = 1.0
                self.fc.weight[1, 1] = 1.0
                self.fc.weight[2, 0] = 1.0  # collides with base head 0
                self.fc.weight[3, 3] = 1.0  # orthogonal to base features
                self.fc.bias.copy_(torch.tensor([0.0, 0.0, 2.0, -5.0]))

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return self.fc(x), x

    dummy = DummyModel()
    val_x = torch.cat([torch.tensor([[2.0, 0.0, 0.0, 0.0]]).repeat(50, 1), torch.tensor([[0.0, 2.0, 0.0, 0.0]]).repeat(50, 1)], dim=0)
    val_loader = DataLoader(TensorDataset(val_x, torch.zeros(100, dtype=torch.long)), batch_size=50)
    gammas = calibrate_per_head_bias(dummy, val_loader, num_base_classes=2, max_steal_per_head=0.05, device="cpu")
    assert gammas.shape == (2,)
    assert gammas[0] < -1.5  # colliding head 2 suppressed
    assert gammas[1] == pytest.approx(0.0)  # orthogonal head 3 untouched

    # 1-epoch DualLwFLearner smoke test expanding 2 -> 4 classes
    x1 = rng.normal(0.0, 1.0, size=(40, 2, 128)).astype(np.float32)
    f1 = rng.normal(-1.0, 0.1, size=(40, 64)).astype(np.float32)
    rb.add_class_exemplars(1, x1, f1)

    model = TLDNN(num_classes=2)
    learner = DualLwFLearner(
        epochs=1,
        steps_per_epoch=2,
        samples_per_class_per_step=8,
        use_replay=True,
        use_rf_aug=True,
    )
    new_x = rng.normal(0.0, 1.0, size=(32, 2, 128)).astype(np.float32)
    new_y = np.array([0] * 16 + [1] * 16, dtype=np.int64)
    protos = rng.normal(0.0, 0.5, size=(2, 64)).astype(np.float32)

    val_dummy = DataLoader(TensorDataset(torch.from_numpy(x0[:16]), torch.zeros(16, dtype=torch.long)), batch_size=16)
    report = learner.evolve(
        backbone=model,
        replay_buffer=rb,
        new_samples=new_x,
        new_pseudo_labels=new_y,
        prototypes_64d=protos,
        val_known_loader=val_dummy,
        num_base_classes=2,
        device="cpu",
    )
    assert model.num_classes == 4
    assert len(report.bic_gammas) == 2
    assert np.isfinite(report.final_loss)
    out_logits, out_feats = model(torch.from_numpy(new_x[:4]))
    assert out_logits.shape == (4, 4) and out_feats.shape == (4, 64)
