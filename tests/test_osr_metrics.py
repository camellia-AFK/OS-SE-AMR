"""Unit tests for Stage 1 OSR metrics, scorers, ARPL, and DistancePrototype modules."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from os_se_amr.eval.osr_metrics import compute_auroc, compute_fpr_at_tpr, compute_oscr
from os_se_amr.osr import (
    ARPLLoss,
    ARPLScorer,
    DistancePrototypeLoss,
    DistancePrototypeScorer,
    MLSScorer,
    MahalanobisPrototypeScorer,
    SoftmaxScorer,
    calibrate_threshold,
)


def _reference_bruteforce_oscr(
    known_scores: np.ndarray,
    known_preds: np.ndarray,
    known_labels: np.ndarray,
    unknown_scores: np.ndarray,
) -> float:
    """Reference O(M * N) OSCR integration over exact unique score thresholds."""
    thresholds = np.unique(np.concatenate([known_scores, unknown_scores]))[::-1]
    correct = known_preds == known_labels
    fprs = [0.0]
    ccrs = [0.0]
    for thr in thresholds:
        fprs.append(float((unknown_scores >= thr).mean()))
        ccrs.append(float((correct & (known_scores >= thr)).mean()))
    return float(np.trapz(ccrs, fprs))


def test_osr_metrics_synthetic_boundaries() -> None:
    # 1. Perfect separation + 100% known accuracy
    k_scores = np.array([5.0, 4.0, 3.0, 2.0])
    u_scores = np.array([-1.0, -2.0, -3.0, -4.0])
    k_labels = np.array([0, 1, 2, 3])
    k_preds = np.array([0, 1, 2, 3])

    assert compute_auroc(k_scores, u_scores) == pytest.approx(1.0)
    fpr95, _ = compute_fpr_at_tpr(k_scores, u_scores, target_tpr=0.95)
    assert fpr95 == pytest.approx(0.0)
    oscr, _, _ = compute_oscr(k_scores, k_preds, k_labels, u_scores)
    assert oscr == pytest.approx(1.0)

    # 2. Perfect separation + 75% known accuracy -> OSCR == 0.75
    k_preds_75 = np.array([0, 1, 2, 0])
    oscr_75, _, _ = compute_oscr(k_scores, k_preds_75, k_labels, u_scores)
    assert oscr_75 == pytest.approx(0.75)

    # 3. Reversed scores -> AUROC == 0.0, FPR@TPR95 == 1.0, OSCR == 0.0
    assert compute_auroc(u_scores, k_scores) == pytest.approx(0.0)
    fpr_rev, _ = compute_fpr_at_tpr(u_scores, k_scores, target_tpr=0.95)
    assert fpr_rev == pytest.approx(1.0)
    oscr_rev, _, _ = compute_oscr(u_scores, k_preds, k_labels, k_scores)
    assert oscr_rev == pytest.approx(0.0)

    # 4. Tied scores + reference cross-check
    rng = np.random.default_rng(7)
    k_mix = np.round(rng.normal(1.0, 1.0, size=60), decimals=1)
    u_mix = np.round(rng.normal(0.0, 1.0, size=50), decimals=1)
    y_mix = rng.integers(0, 4, size=60)
    p_mix = y_mix.copy()
    p_mix[:12] = (p_mix[:12] + 1) % 4

    oscr_vec, fprs, ccrs = compute_oscr(k_mix, p_mix, y_mix, u_mix)
    oscr_ref = _reference_bruteforce_oscr(k_mix, p_mix, y_mix, u_mix)
    assert np.isfinite(oscr_vec) and 0.0 <= oscr_vec <= 1.0
    assert fprs[0] == 0.0 and ccrs[0] == 0.0
    assert fprs[-1] == pytest.approx(1.0)
    assert oscr_vec == pytest.approx(oscr_ref, abs=1e-4)

    # 5. Threshold calibration
    val_scores = np.linspace(0.0, 10.0, 1001)
    thr95 = calibrate_threshold(val_scores, target_tpr=0.95)
    assert (val_scores >= thr95).mean() == pytest.approx(0.95, abs=0.005)


def test_osr_scorers_and_losses() -> None:
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    # Softmax & MLS polarity check
    logits = torch.tensor([[5.0, 1.0, 0.0], [0.5, 0.4, 0.3]])
    feats = torch.randn(2, 64)
    s_soft = SoftmaxScorer().score(logits, feats)
    s_mls = MLSScorer().score(logits, feats)
    assert s_soft[0] > s_soft[1]
    assert s_mls[0] > s_mls[1]

    for dev in devices:
        f = torch.randn(12, 64, device=dev, requires_grad=True)
        y = torch.arange(12, device=dev) % 7

        # ARPL gradient & scorer check
        arpl = ARPLLoss(num_classes=7, feat_dim=64, weight_pl=0.1, temp=1.0).to(dev)
        logits_a, loss_a = arpl(f, y)
        assert logits_a.shape == (12, 7)
        assert loss_a is not None and torch.isfinite(loss_a)
        loss_a.backward()
        assert arpl.points.grad is not None and arpl.radius.grad is not None

        arpl_scorer = ARPLScorer(arpl)
        scores_a = arpl_scorer.score(logits_a.detach(), f.detach())
        assert scores_a.shape == (12,) and torch.isfinite(scores_a).all()

        # DistancePrototype gradient, polarity & expansion check
        f2 = torch.randn(12, 64, device=dev, requires_grad=True)
        proto = DistancePrototypeLoss(num_classes=7, feat_dim=64, lambda_proto=0.1, temp=0.1).to(dev)
        logits_p, loss_p = proto(f2, y)
        assert logits_p.shape == (12, 7)
        assert loss_p is not None and torch.isfinite(loss_p)
        loss_p.backward()
        assert proto.prototypes.grad is not None

        proto_scorer = DistancePrototypeScorer(proto)
        # Feature identical to prototype 0 must score higher than distant feature
        near_f = proto.prototypes[0:1].detach()
        far_f = near_f + 5.0
        s_near = proto_scorer.score(torch.empty(1, 7, device=dev), near_f)
        s_far = proto_scorer.score(torch.empty(1, 7, device=dev), far_f)
        assert s_near.item() > s_far.item()

        # Expand 7 -> 9 preserving old prototypes
        old_p = proto.prototypes.detach().clone()
        proto.expand(2)
        assert proto.num_classes == 9
        assert proto.prototypes.shape == (9, 64)
        assert proto.prototypes.device == old_p.device
        assert torch.allclose(proto.prototypes[:7], old_p, atol=1e-6)


def test_mahalanobis_prototype_scorer() -> None:
    rng = np.random.default_rng(42)
    # Create 2 anisotropic clusters stretched along dim 0, tightly constrained along dims 1..3
    n_per_cls = 200
    c0 = np.column_stack(
        [
            rng.normal(-2.0, 1.5, size=n_per_cls),
            rng.normal(0.0, 0.05, size=n_per_cls),
            rng.normal(0.0, 0.05, size=n_per_cls),
            rng.normal(0.0, 0.05, size=n_per_cls),
        ]
    )
    c1 = np.column_stack(
        [
            rng.normal(2.0, 1.5, size=n_per_cls),
            rng.normal(0.0, 0.05, size=n_per_cls),
            rng.normal(0.0, 0.05, size=n_per_cls),
            rng.normal(0.0, 0.05, size=n_per_cls),
        ]
    )
    train_x = np.vstack([c0, c1])
    train_y = np.array([0] * n_per_cls + [1] * n_per_cls)

    stem_x = train_x * 2.0 + rng.normal(0.0, 0.02, size=train_x.shape)
    scorer = MahalanobisPrototypeScorer(layers=("stem", "pre_ln"), weights={"stem": 0.7, "pre_ln": 0.3})
    scorer.fit_multiscale({"stem": stem_x, "pre_ln": train_x}, train_y)

    # In-distribution sample near cluster 0 vs OOD sample shifted along tight orthogonal dim 1
    id_sample = {"stem": np.array([[-4.0, 0.0, 0.0, 0.0]]), "pre_ln": np.array([[-2.0, 0.0, 0.0, 0.0]])}
    ood_sample = {"stem": np.array([[-4.0, 1.0, 0.0, 0.0]]), "pre_ln": np.array([[-2.0, 0.5, 0.0, 0.0]])}
    s_id = scorer.score_multiscale(id_sample)
    s_ood = scorer.score_multiscale(ood_sample)
    assert s_id[0] > s_ood[0]

    # Verify state_dict / load_state_dict round-trip
    restored = MahalanobisPrototypeScorer()
    restored.load_state_dict(scorer.state_dict())
    assert np.allclose(restored.score_multiscale(id_sample), s_id)
    assert np.allclose(restored.score_multiscale(ood_sample), s_ood)

    # Single-layer Tensor fallback check
    single_scorer = MahalanobisPrototypeScorer(layers=("pre_ln",))
    single_scorer.fit(torch.from_numpy(train_x).float(), torch.from_numpy(train_y))
    t_out = single_scorer.score(torch.empty(1, 2), torch.from_numpy(ood_sample["pre_ln"]).float())
    assert t_out.shape == (1,) and torch.isfinite(t_out).all()
