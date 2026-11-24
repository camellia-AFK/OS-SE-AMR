"""Unit tests for Stage 3 prototype-sync fix, CIL baselines, SSD-AMC OSR, and SNR drift evaluators."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from os_se_amr.continual import (
    NaiveFineTuneLearner,
    ReplayBuffer,
    SimpleCILLearner,
    iCaRLLearner,
    sync_prototypes_and_mahalanobis,
)
from os_se_amr.data import SplitBundle, build_arrival_sessions
from os_se_amr.eval import evaluate_snr_cil_drift, evaluate_snr_osr_drift
from os_se_amr.models import TLDNN
from os_se_amr.osr import (
    MahalanobisPrototypeScorer,
    SSDMahalanobisScorer,
    SSDProjectionHead,
    SupConLoss,
    ssd_rf_augment,
)


def test_prototype_sync_no_duplicate_classes() -> None:
    rng = np.random.default_rng(42)
    backbone = TLDNN(num_classes=3, feature_dim=64)
    old_x = rng.normal(0.0, 1.0, size=(30, 2, 128)).astype(np.float32)
    old_feats = rng.normal(0.0, 1.0, size=(30, 64)).astype(np.float32)

    rb = ReplayBuffer(exemplars_per_class=5, selection_strategy="random")
    for cid in range(3):
        rb.add_class_exemplars(cid, old_x[cid * 10 : (cid + 1) * 10], seed=42)

    prev_scorer = MahalanobisPrototypeScorer(layers=("stem", "pre_ln"), weights={"stem": 0.7, "pre_ln": 0.3})
    prev_scorer.fit_multiscale(
        {"stem": rng.normal(size=(30, 128)).astype(np.float32), "pre_ln": old_feats},
        np.repeat([0, 1, 2], 10),
    )

    # Simulate expanding backbone from 3 -> 5 classes
    backbone.expand_classifier(2)
    new_x = rng.normal(0.5, 1.0, size=(20, 2, 128)).astype(np.float32)
    new_local_y = np.array([0] * 10 + [1] * 10, dtype=np.int64)

    # Case 1: ReplayBuffer ALREADY contains classes 3 & 4 before sync (the Stage 2 regression bug case)
    for k in range(2):
        rb.add_class_exemplars(3 + k, new_x[new_local_y == k], seed=42)
    res_preadded = sync_prototypes_and_mahalanobis(
        backbone=backbone,
        replay_buffer=rb,
        new_core_x=new_x,
        new_core_labels=new_local_y,
        prev_scorer=prev_scorer,
        device="cpu",
    )
    assert res_preadded["num_synced_classes"] == 5
    assert res_preadded["scorer"].classes == [0, 1, 2, 3, 4]
    assert np.isfinite(res_preadded["old_proto_shift_pre_ln"])
    assert np.isfinite(res_preadded["old_proto_shift_stem"])

    # Case 2: ReplayBuffer contains only classes 0, 1, 2
    rb_old = ReplayBuffer(exemplars_per_class=5, selection_strategy="random")
    for cid in range(3):
        rb_old.add_class_exemplars(cid, old_x[cid * 10 : (cid + 1) * 10], seed=42)
    res_old_only = sync_prototypes_and_mahalanobis(
        backbone=backbone,
        replay_buffer=rb_old,
        new_core_x=new_x,
        new_core_labels=new_local_y,
        prev_scorer=prev_scorer,
        device="cpu",
        num_old_classes=3,
    )
    assert res_old_only["num_synced_classes"] == 5
    assert res_old_only["scorer"].classes == [0, 1, 2, 3, 4]


def test_simplecil_and_icarl_and_naive_smoke() -> None:
    rng = np.random.default_rng(11)
    base_x = rng.normal(0.0, 1.0, size=(18, 2, 128)).astype(np.float32)
    base_y = np.repeat([0, 1, 2], 6).astype(np.int64)
    new_x = rng.normal(1.0, 1.0, size=(12, 2, 128)).astype(np.float32)
    new_local_y = np.array([0] * 6 + [1] * 6, dtype=np.int64)
    test_ds = TensorDataset(
        torch.from_numpy(np.concatenate([base_x[:6], new_x[:6]], axis=0)),
        torch.tensor([0] * 6 + [3] * 6, dtype=torch.long),
    )

    # 1. SimpleCIL must freeze all backbone feature weights bit-identically
    m_sc = TLDNN(num_classes=3, feature_dim=64)
    snap = {k: v.detach().clone() for k, v in m_sc.named_parameters() if not k.startswith("classifier.")}
    sc = SimpleCILLearner(feature_layer="penult")
    sc.fit_base(m_sc, base_x, base_y, device="cpu")
    rep_sc = sc.evolve(m_sc, new_x, new_local_y, device="cpu")
    assert rep_sc.epochs == 0
    assert m_sc.num_classes == 5
    for k, v in m_sc.named_parameters():
        if not k.startswith("classifier."):
            assert torch.equal(v, snap[k])
    yt_sc, yp_sc = sc.predict_dataset(m_sc, test_ds, device="cpu")
    assert yt_sc.shape == (12,) and yp_sc.shape == (12,)

    # 2. iCaRLLearner: evolve + update_class_means + predict_nme
    m_ic = TLDNN(num_classes=3, feature_dim=64)
    rb = ReplayBuffer(exemplars_per_class=4, selection_strategy="random")
    for c in range(3):
        rb.add_class_exemplars(c, base_x[base_y == c], seed=1)
    ic = iCaRLLearner(epochs=2, batch_size=8, lr=1e-3)
    rep_ic = ic.evolve(m_ic, rb, new_x, new_local_y, device="cpu", seed=1)
    assert rep_ic.num_old_classes == 3 and rep_ic.num_new_classes == 2
    assert m_ic.num_classes == 5
    assert np.isfinite(rep_ic.final_loss)
    for k in range(2):
        rb.add_class_exemplars(3 + k, new_x[new_local_y == k], seed=1)
    ic.update_class_means(m_ic, rb, device="cpu")
    assert sorted(ic.class_means.keys()) == [0, 1, 2, 3, 4]
    yt_ic, yp_ic = ic.predict_nme(m_ic, test_ds, device="cpu")
    assert yt_ic.shape == (12,) and yp_ic.shape == (12,)

    # 3. NaiveFineTuneLearner
    m_nv = TLDNN(num_classes=3, feature_dim=64)
    nv = NaiveFineTuneLearner(epochs=2, batch_size_new=6, lr=1e-3)
    rep_nv = nv.evolve(m_nv, ReplayBuffer(4, "random"), new_x, new_local_y, device="cpu", seed=1)
    assert rep_nv.num_new_classes == 2
    assert rep_nv.final_feat_kd == 0.0 and rep_nv.final_logit_kd == 0.0
    assert m_nv.num_classes == 5


def test_ssd_amc_loss_and_scorer() -> None:
    torch.manual_seed(42)
    x = torch.randn(16, 2, 128)
    y = torch.arange(4).repeat(4)

    x_aug = ssd_rf_augment(x)
    assert x_aug.shape == x.shape
    assert torch.isfinite(x_aug).all()

    proj = SSDProjectionHead(in_dim=64, hidden_dim=64, proj_dim=32)
    f1 = torch.randn(16, 64, requires_grad=True)
    f2 = torch.randn(16, 64, requires_grad=True)
    p1, p2 = proj(f1), proj(f2)
    assert p1.shape == (16, 32)
    assert torch.allclose(p1.norm(dim=1), torch.ones(16), atol=1e-5)

    loss_fn = SupConLoss(temperature=0.1)
    loss = loss_fn(torch.cat([p1, p2], dim=0), torch.cat([y, y], dim=0))
    assert torch.isfinite(loss) and float(loss.item()) > 0.0
    loss.backward()

    # SSDMahalanobisScorer separability + state_dict round-trip
    rng = np.random.default_rng(7)
    k_feats = np.vstack(
        [
            rng.normal([3.0, 0.0] + [0.0] * 62, 0.08, size=(40, 64)),
            rng.normal([0.0, 3.0] + [0.0] * 62, 0.08, size=(40, 64)),
        ]
    ).astype(np.float32)
    k_labels = np.array([0] * 40 + [1] * 40, dtype=np.int64)
    u_feats = rng.normal([0.0, 0.0, 3.0] + [0.0] * 61, 0.08, size=(20, 64)).astype(np.float32)

    scorer = SSDMahalanobisScorer()
    scorer.fit_numpy(k_feats, k_labels)
    k_scores = scorer.score_numpy(k_feats)
    u_scores = scorer.score_numpy(u_feats)
    assert float(k_scores.mean()) > float(u_scores.mean())

    restored = SSDMahalanobisScorer()
    restored.load_state_dict(scorer.state_dict())
    assert np.allclose(scorer.score_numpy(u_feats), restored.score_numpy(u_feats))


def test_snr_drift_evaluators_and_stream_snrs() -> None:
    # 1. evaluate_snr_osr_drift: lower SNR degraded scores increase Known FRR
    regimes = {"high_snr": [10, 12], "mid_low_snr": [0, 2], "neg_snr": [-10, -8]}
    k_snrs = np.array([10] * 10 + [12] * 10 + [0] * 10 + [2] * 10 + [-10] * 10 + [-8] * 10)
    k_scores = np.array([2.0] * 20 + [0.2] * 20 + [-1.5] * 20, dtype=np.float64)
    k_labels = np.tile([0, 1], 30)
    k_preds = k_labels.copy()

    u_snrs = k_snrs.copy()
    u_scores = np.full(60, -2.5, dtype=np.float64)
    u_labels = np.full(60, 2, dtype=np.int64)

    osr_drift = evaluate_snr_osr_drift(
        known_scores=k_scores,
        known_preds=k_preds,
        known_labels=k_labels,
        known_snrs=k_snrs,
        unknown_scores=u_scores,
        unknown_labels=u_labels,
        unknown_snrs=u_snrs,
        calibrated_threshold=0.5,
        regimes=regimes,
    )
    assert osr_drift["regimes"]["high_snr"]["known_frr_pct"] == pytest.approx(0.0)
    assert osr_drift["regimes"]["mid_low_snr"]["known_frr_pct"] == pytest.approx(100.0)
    assert osr_drift["drift_summary"]["frr_drift_neg_pct"] == pytest.approx(100.0)
    assert set(osr_drift["per_snr"].keys()) == {"-10", "-8", "0", "2", "10", "12"}

    # 2. evaluate_snr_cil_drift with fixed head_to_true mapping
    yt = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int64)
    yp = np.array([0, 1, 3, 0, 1, 3, 0, 0, 1], dtype=np.int64)
    snrs = np.array([10, 10, 10, 0, 0, 0, -10, -10, -10], dtype=np.int64)
    cil_drift = evaluate_snr_cil_drift(
        y_true=yt,
        y_pred=yp,
        snrs=snrs,
        num_base_classes=2,
        current_true_classes=[0, 1, 2],
        fixed_head_to_true={3: 2},
        regimes={"high_snr": [10], "mid_low_snr": [0], "neg_snr": [-10]},
    )
    assert cil_drift["regimes"]["high_snr"]["a_all_pct"] == pytest.approx(100.0)
    assert cil_drift["regimes"]["neg_snr"]["a_all_pct"] < 50.0
    assert cil_drift["drift_summary"]["a_all_drop_neg_pct"] > 50.0

    # 3. build_arrival_sessions(..., snrs=[10, 12]) filters SNRs strictly
    rng = np.random.default_rng(9)
    mods = ["M0", "M1", "M2", "M3"]
    eval_snrs = [-2, 0, 6, 10, 12]
    raw_dict = {(m, s): rng.normal(size=(20, 2, 128)).astype(np.float32) for m in mods for s in eval_snrs}
    split_cfg = {
        "name": "tiny_snr",
        "seed": 42,
        "snr_min": 0,
        "snr_max": 12,
        "stress_snrs": [-2],
        "ratios": {"train": 0.6, "val": 0.2, "test": 0.2},
        "known_classes": ["M0", "M1"],
        "unknown_classes": ["M2", "M3"],
        "arrival_orders": {"order_a": [["M2", "M3"]]},
    }
    bundle = SplitBundle(raw_dict=raw_dict, split_cfg=split_cfg)
    batches = build_arrival_sessions(bundle, order_name="order_a", samples_per_class=10, snrs=[10, 12], seed=42)
    assert len(batches) == 1
    assert set(np.unique(batches[0].snrs).tolist()) == {10, 12}
