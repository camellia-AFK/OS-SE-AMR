"""Unit tests for Stage 4 blind SNR estimators, TLDNN-SS, DualAwareGate, and ClosedLoopEvolver."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from os_se_amr.continual import rf_augment_batch
from os_se_amr.data import SplitBundle, compute_openness
from os_se_amr.discovery import (
    DualAwareGate,
    HDBSCANDiscoverer,
    augment_core_clusters,
    ss_iq_pair,
    ss_phase_aligned_pair,
)
from os_se_amr.engine import (
    ClosedLoopConfig,
    ClosedLoopEvolver,
    build_closed_loop_mixed_sessions,
)
from os_se_amr.eval import aggregate_multiseed_results
from os_se_amr.models import (
    AuxSNRRegressor,
    M2M4SNREstimator,
    TLDNN,
    evaluate_snr_estimator,
)
from os_se_amr.osr import MahalanobisPrototypeScorer
from os_se_amr.utils import load_yaml, resolve_path, save_checkpoint


def test_m2m4_and_aux_snr_estimator_interface() -> None:
    # Clean constant-envelope QPSK-like wave vs heavy AWGN wave
    rng = np.random.default_rng(42)
    t = np.arange(128, dtype=np.float32)
    clean_wave = np.stack([np.cos(0.3 * t), np.sin(0.3 * t)], axis=0)
    hi_iq = np.repeat(clean_wave[None, :, :], 32, axis=0) + 0.02 * rng.normal(size=(32, 2, 128)).astype(np.float32)
    lo_iq = np.repeat(clean_wave[None, :, :], 32, axis=0) + 1.50 * rng.normal(size=(32, 2, 128)).astype(np.float32)

    m2m4 = M2M4SNREstimator()
    hi_est = m2m4.predict_snr(torch.from_numpy(hi_iq)).numpy()
    lo_est = m2m4.predict_snr(torch.from_numpy(lo_iq)).numpy()
    assert hi_est.shape == (32,) and lo_est.shape == (32,)
    assert np.all(np.isfinite(hi_est)) and np.all(np.isfinite(lo_est))
    assert float(hi_est.mean()) > float(lo_est.mean()) + 5.0

    # Calibration & state_dict round-trip
    cal_x = np.concatenate([lo_iq, hi_iq], axis=0)
    cal_y = np.array([-4.0] * 32 + [14.0] * 32, dtype=np.float32)
    m2m4.fit_calibration(cal_x, cal_y)
    restored = M2M4SNREstimator()
    restored.load_state_dict(m2m4.state_dict())
    assert np.allclose(
        m2m4.predict_snr(torch.from_numpy(cal_x)).numpy(),
        restored.predict_snr(torch.from_numpy(cal_x)).numpy(),
    )

    # AuxSNRRegressor forward/backward & evaluate_snr_estimator schema
    aux = AuxSNRRegressor()
    xb = torch.from_numpy(cal_x)
    yb = torch.from_numpy(cal_y)
    pred = aux(xb)
    assert pred.shape == (64,)
    loss = torch.nn.functional.smooth_l1_loss(pred, yb)
    assert torch.isfinite(loss)
    loss.backward()

    snr_p = aux.predict_snr(xb)
    assert snr_p.shape == (64,) and torch.isfinite(snr_p).all()

    report = evaluate_snr_estimator(
        aux,
        cal_x,
        cal_y,
        labels=np.array([0] * 32 + [1] * 32),
        idx_to_name={0: "A", 1: "B"},
        gate_thresholds=(0.0, 4.0),
        device="cpu",
    )
    assert "mae_db" in report and "rmse_db" in report
    assert "ge_4db" in report["gates"]
    assert set(report["per_class"].keys()) == {"A", "B"}


def test_ss_augment_intra_cluster_and_phase_continuity() -> None:
    t = np.arange(128, dtype=np.float32)
    xa = np.stack([2.0 * np.cos(0.25 * t + 0.3), 2.0 * np.sin(0.25 * t + 0.3)], axis=0)
    xb = np.stack([3.0 * np.cos(0.25 * t + 2.1), 3.0 * np.sin(0.25 * t + 2.1)], axis=0)

    iq_stitched = ss_iq_pair(xa, xb, cut=64)
    assert np.allclose(iq_stitched[:, :64], xa[:, :64])
    assert np.allclose(iq_stitched[:, 64:], xb[:, 64:])

    pa_stitched = ss_phase_aligned_pair(xa, xb, cut=64)
    assert np.allclose(pa_stitched[:, :64], xa[:, :64])
    # Exact instantaneous envelope preservation on second half
    amp_orig = np.hypot(xb[0, 64:], xb[1, 64:])
    amp_aligned = np.hypot(pa_stitched[0, 64:], pa_stitched[1, 64:])
    assert np.allclose(amp_aligned, amp_orig, atol=1e-5)

    # Zero-order seam phase jump across 63 -> 64 must be zero (mod 2*pi)
    th63 = float(np.arctan2(pa_stitched[1, 63], pa_stitched[0, 63]))
    th64 = float(np.arctan2(pa_stitched[1, 64], pa_stitched[0, 64]))
    diff_wrapped = math.atan2(math.sin(th64 - th63), math.cos(th64 - th63))
    assert abs(diff_wrapped) < 1e-4

    # Cluster isolation test: cluster 0 has value +10, cluster 1 has value -10 -> zero cross-mixing
    c0 = np.full((12, 2, 128), 10.0, dtype=np.float32)
    c1 = np.full((12, 2, 128), -10.0, dtype=np.float32)
    x_core = np.concatenate([c0, c1], axis=0)
    y_core = np.array([0] * 12 + [1] * 12, dtype=np.int64)

    x_none, y_none = augment_core_clusters(x_core, y_core, mode="none", expand_factor=4, seed=42)
    assert len(x_none) == 24 and np.allclose(x_none, x_core)

    x_aug, y_aug = augment_core_clusters(
        x_core, y_core, mode="ss_phase_aligned", expand_factor=3, max_samples_per_cluster=30, seed=42
    )
    # Each cluster expands from 12 -> min(30, 12*3) = 30 => total 60
    assert len(x_aug) == 60
    assert np.allclose(x_aug[y_aug == 0], 10.0, atol=1e-4)
    assert np.allclose(x_aug[y_aug == 1], -10.0, atol=1e-4)


def test_dual_aware_gate_modes() -> None:
    osr_scores = np.array([-3.0, -3.0, 2.0, -2.5, -2.5, 1.5], dtype=np.float64)
    snr_est = np.array([10.0, -5.0, 12.0, 8.0, -6.0, 14.0], dtype=np.float64)
    true_snrs = np.array([10, -4, 12, 8, -6, 14], dtype=np.int64)
    true_labels = np.array([2, 2, 0, 1, 0, 1], dtype=np.int64)

    g_osr = DualAwareGate(gate_mode="osr_only", snr_threshold=4.0).filter(
        osr_scores, osr_threshold=0.0, true_labels=true_labels, true_snrs=true_snrs, num_known_classes=2
    )
    assert np.array_equal(g_osr.passed_mask, np.array([True, True, False, True, True, False]))
    assert g_osr.neg_snr_leakage_pct == pytest.approx(50.0)

    g_aux = DualAwareGate(gate_mode="aux_snr", snr_threshold=4.0).filter(
        osr_scores, osr_threshold=0.0, snr_est=snr_est, true_labels=true_labels, true_snrs=true_snrs, num_known_classes=2
    )
    assert np.array_equal(g_aux.passed_mask, np.array([True, False, False, True, False, False]))
    assert g_aux.neg_snr_leakage_pct == pytest.approx(0.0)
    assert g_aux.unknown_purity_pct == pytest.approx(50.0)

    g_oracle = DualAwareGate(gate_mode="oracle_snr", snr_threshold=4.0).filter(
        osr_scores, osr_threshold=0.0, true_labels=true_labels, true_snrs=true_snrs, num_known_classes=2
    )
    assert np.array_equal(g_oracle.passed_mask, g_aux.passed_mask)


def test_lwf_use_ss_aug_flag_and_closed_loop_smoke(tmp_path: Path) -> None:
    # 1. rf_augment_batch(..., use_ss_aug=False) must preserve per-sample power envelope
    torch.manual_seed(0)
    x = torch.randn(16, 2, 128)
    y = torch.zeros(16, dtype=torch.long)
    out_no_ss = rf_augment_batch(x, y, use_ss_aug=False)
    p_in = x[:, 0] ** 2 + x[:, 1] ** 2
    p_out = out_no_ss[:, 0] ** 2 + out_no_ss[:, 1] ** 2
    assert torch.allclose(p_in, p_out, atol=1e-5)

    # 2. Tiny end-to-end ClosedLoopEvolver smoke test
    rng = np.random.default_rng(7)
    mods = ["K0", "K1", "K2", "U0", "U1"]
    snrs = [-4, 0, 10, 14]
    raw_dict: dict[tuple[str, int], np.ndarray] = {}
    t = np.arange(128, dtype=np.float32)
    freqs = {"K0": 0.10, "K1": 0.25, "K2": 0.42, "U0": 0.65, "U1": 0.85}
    for m in mods:
        base_wave = np.stack([np.cos(freqs[m] * t), np.sin(freqs[m] * t)], axis=0)
        for s in snrs:
            noise_scale = 0.05 if s >= 10 else (0.15 if s >= 0 else 1.4)
            raw_dict[(m, s)] = (
                np.repeat(base_wave[None, :, :], 40, axis=0)
                + noise_scale * rng.normal(size=(40, 2, 128)).astype(np.float32)
            )

    split_cfg = {
        "name": "tiny_cl",
        "seed": 42,
        "snr_min": 0,
        "snr_max": 14,
        "stress_snrs": [-4],
        "ratios": {"train": 0.6, "val": 0.2, "test": 0.2},
        "known_classes": ["K0", "K1", "K2"],
        "unknown_classes": ["U0", "U1"],
        "arrival_orders": {"order_a": [["U0", "U1"]]},
    }
    bundle = SplitBundle(raw_dict=raw_dict, split_cfg=split_cfg)
    model_cfg = {
        "name": "tldnn",
        "feature_dim": 64,
        "dropout": 0.0,
        "tldnn": {
            "input_mode": "iq_ap",
            "d_model": 64,
            "nhead": 4,
            "num_transformer_layers": 1,
            "lstm_hidden": 64,
            "lstm_layers": 1,
        },
    }
    cil_cfg = {
        "exemplars_per_class": 6,
        "selection_strategy": "herding",
        "epochs": 2,
        "steps_per_epoch": 2,
        "samples_per_class_per_step": 4,
        "batch_size_new": 8,
        "batch_size_old": 8,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "beta_feat": 0.05,
        "beta_logit": 0.5,
        "distill_temp": 2.0,
        "use_rf_aug": True,
        "bic_max_steal": 0.05,
    }

    base_model = TLDNN(num_classes=3, feature_dim=64, num_transformer_layers=1, lstm_layers=1)
    ckpt_path = tmp_path / "base.pt"
    save_checkpoint(ckpt_path, base_model, None, epoch=1, metrics={})

    tr_k, va_k, _ = bundle.get_closed_set_datasets()
    with torch.no_grad():
        ms_tr = base_model.eval().extract_multiscale_features(torch.from_numpy(tr_k.x))
        ms_va = base_model.extract_multiscale_features(torch.from_numpy(va_k.x))
    scorer = MahalanobisPrototypeScorer(layers=("stem", "pre_ln"), weights={"stem": 0.7, "pre_ln": 0.3})
    scorer.fit_multiscale(
        {k: v.numpy() for k, v in ms_tr.items()},
        tr_k.y,
        val_feats={k: v.numpy() for k, v in ms_va.items()},
    )

    sessions = build_closed_loop_mixed_sessions(
        bundle, order_name="order_a", samples_per_class_main=20, samples_per_class_neg=8, seed=42
    )
    assert len(sessions) == 1 and sessions[0].mode == "mixed"

    m2m4 = M2M4SNREstimator()
    cl_cfg = ClosedLoopConfig(
        method_key="ours",
        gate_mode="m2m4",
        snr_threshold=-20.0,
        target_tpr=0.90,
        ss_mode="ss_phase_aligned",
        ss_expand_factor=2,
        exemplars_per_class=6,
        min_cluster_size=5,
        min_samples=3,
        silhouette_threshold=0.0,
        merge_cosine_threshold=None,
        min_cluster_silhouette=None,
        min_cluster_cohesion=None,
        ss_min_cluster_silhouette=None,
        ss_min_cluster_cohesion=None,
        ss_max_core_size=None,
        seed=42,
    )
    res = ClosedLoopEvolver().run_trajectory(
        method_label="Ours Smoke",
        order_name="order_a",
        cl_cfg=cl_cfg,
        sessions=sessions,
        base_ckpt_path=ckpt_path,
        base_scorer=scorer,
        snr_estimator=m2m4,
        bundle=bundle,
        model_cfg=model_cfg,
        cil_cfg=cil_cfg,
        device=torch.device("cpu"),
        logger=logging.getLogger("test_stage4"),
    )
    assert len(res["sessions"]) == 2
    assert "gating" in res["sessions"][1]
    assert np.isfinite(res["sessions"][1]["a_all_pct"])

    # 3. Verify allow_single_cluster=True accepts a single clean cluster
    single_feats = {
        "stem": np.random.default_rng(1).normal(2.0, 0.05, size=(40, 128)).astype(np.float32),
        "pre_ln": np.random.default_rng(2).normal(2.0, 0.05, size=(40, 64)).astype(np.float32),
    }
    disc_single = HDBSCANDiscoverer(
        min_cluster_size=15, min_samples=5, allow_single_cluster=True
    ).discover(single_feats)
    assert disc_single.accepted is True and disc_single.k_hat == 1


def test_hdbscan_merge_and_prune_subclusters() -> None:
    rng = np.random.default_rng(123)

    def _basis(dim: int, idx: int) -> np.ndarray:
        v = np.zeros(dim, dtype=np.float32)
        v[idx] = 1.0
        return v

    # C0 & C1 share identical pre_ln direction e_0 (cos > 0.95) with orthogonal stems s_0 vs s_1 -> should merge
    c0_stem = 4.0 * _basis(128, 0)[None, :] + 0.05 * rng.normal(size=(40, 128)).astype(np.float32)
    c0_pre = 4.0 * _basis(64, 0)[None, :] + 0.05 * rng.normal(size=(40, 64)).astype(np.float32)

    c1_stem = 4.0 * _basis(128, 1)[None, :] + 0.05 * rng.normal(size=(40, 128)).astype(np.float32)
    c1_pre = 4.0 * _basis(64, 0)[None, :] + 0.05 * rng.normal(size=(40, 64)).astype(np.float32)

    # C2 is orthogonal in both stem (s_2) and pre_ln (e_1) -> stays a separate clean cluster
    c2_stem = 4.0 * _basis(128, 2)[None, :] + 0.05 * rng.normal(size=(40, 128)).astype(np.float32)
    c2_pre = 4.0 * _basis(64, 1)[None, :] + 0.05 * rng.normal(size=(40, 64)).astype(np.float32)

    # C3 is a diffuse low-cohesion cluster isolated in dims [10:25] -> forms 4th raw cluster then gets pruned
    c3_stem = np.repeat(0.8 * _basis(128, 10)[None, :], 40, axis=0)
    c3_pre = np.repeat(0.8 * _basis(64, 10)[None, :], 40, axis=0)
    c3_stem[:, 11:25] += 0.35 * rng.normal(size=(40, 14)).astype(np.float32)
    c3_pre[:, 11:25] += 0.35 * rng.normal(size=(40, 14)).astype(np.float32)

    feats = {
        "stem": np.concatenate([c0_stem, c1_stem, c2_stem, c3_stem], axis=0),
        "pre_ln": np.concatenate([c0_pre, c1_pre, c2_pre, c3_pre], axis=0),
    }

    disc_raw = HDBSCANDiscoverer(
        min_cluster_size=15,
        min_samples=5,
        silhouette_threshold=0.40,
        allow_single_cluster=True,
    ).discover(feats)
    assert disc_raw.k_hat == 4

    disc_refined = HDBSCANDiscoverer(
        min_cluster_size=15,
        min_samples=5,
        silhouette_threshold=0.40,
        allow_single_cluster=True,
        merge_cosine_threshold=0.80,
        min_cluster_silhouette=0.40,
        min_cluster_cohesion=0.85,
    ).discover(feats)
    assert disc_refined.accepted is True
    assert disc_refined.raw_k_hat == 4
    assert disc_refined.k_hat == 2
    assert set(np.unique(disc_refined.cluster_ids[disc_refined.cluster_ids >= 0]).tolist()) == {0, 1}
    # C0 (0:40) and C1 (40:80) must be merged into the same cluster, distinct from C2 (80:120)
    lbl_c0 = int(disc_refined.cluster_ids[0])
    lbl_c1 = int(disc_refined.cluster_ids[40])
    lbl_c2 = int(disc_refined.cluster_ids[80])
    assert lbl_c0 >= 0 and lbl_c0 == lbl_c1 and lbl_c2 >= 0 and lbl_c2 != lbl_c0
    assert np.all(disc_refined.cluster_ids[120:160] == -1)


def test_adaptive_ss_gating() -> None:
    # Build 4 distinct-amplitude clusters:
    # cid=0: scarce tight (n=20, s_k=0.60, coh=0.96) -> eligible -> 20 * 4 = 80
    # cid=1: sample-rich (n=500, s_k=0.60, coh=0.96) -> skipped (> 450)
    # cid=2: low silhouette (n=20, s_k=0.30, coh=0.96) -> skipped (< 0.45)
    # cid=3: low cohesion (n=20, s_k=0.60, coh=0.80) -> skipped (< 0.90)
    x0 = np.full((20, 2, 128), 1.0, dtype=np.float32)
    x1 = np.full((500, 2, 128), 2.0, dtype=np.float32)
    x2 = np.full((20, 2, 128), 3.0, dtype=np.float32)
    x3 = np.full((20, 2, 128), 4.0, dtype=np.float32)
    x_core = np.concatenate([x0, x1, x2, x3], axis=0)
    y_core = np.concatenate(
        [
            np.full(20, 0, dtype=np.int64),
            np.full(500, 1, dtype=np.int64),
            np.full(20, 2, dtype=np.int64),
            np.full(20, 3, dtype=np.int64),
        ]
    )

    x_aug, y_aug = augment_core_clusters(
        x_core=x_core,
        cluster_labels=y_core,
        mode="ss_phase_aligned",
        expand_factor=4,
        max_samples_per_cluster=2000,
        seed=42,
        per_cluster_silhouette={0: 0.60, 1: 0.60, 2: 0.30, 3: 0.60},
        per_cluster_cohesion={0: 0.96, 1: 0.96, 2: 0.96, 3: 0.80},
        min_cluster_silhouette=0.45,
        min_cluster_cohesion=0.90,
        max_core_size_for_ss=450,
    )
    counts = {cid: int((y_aug == cid).sum()) for cid in (0, 1, 2, 3)}
    assert counts == {0: 80, 1: 500, 2: 20, 3: 20}
    assert len(x_aug) == 620


def test_clean10_split_config_openness_and_hygiene() -> None:
    cfg_11 = load_yaml(resolve_path("configs/splits/config_a.yaml"))
    cfg_10 = load_yaml(resolve_path("configs/splits/config_a_clean10.yaml"))

    assert cfg_10["known_classes"] == cfg_11["known_classes"]
    assert "AM-SSB" not in cfg_10["unknown_classes"]
    assert cfg_10["unknown_classes"] == ["QAM64", "WBFM", "AM-DSB"]
    assert cfg_10["arrival_orders"]["order_clean"] == [["QAM64", "WBFM"], ["AM-DSB"]]

    expected_openness = 1.0 - math.sqrt(14.0 / 17.0)
    actual_openness = compute_openness(len(cfg_10["known_classes"]), len(cfg_10["known_classes"]) + len(cfg_10["unknown_classes"]))
    assert actual_openness == pytest.approx(expected_openness, abs=1e-6)
    assert actual_openness * 100.0 == pytest.approx(9.25, abs=0.01)


def test_multiseed_aggregation_and_split_independence() -> None:
    def _make_seed_payload(seed: int, acc0: float, auroc: float, mae: float, s2_all: float) -> dict:
        sessions_b = [
            {"session": 0, "a_all_pct": acc0, "a_old_pct": acc0, "a_new_pct": None, "delta_a_old_pct": 0.0, "fm_cls_pct": 0.0},
            {"session": 1, "a_all_pct": 86.0, "a_old_pct": 93.5, "a_new_pct": 60.0, "delta_a_old_pct": 5.0, "fm_cls_pct": 5.0, "discovered_clusters": 2, "core_purity_pct": 88.0},
            {"session": 2, "a_all_pct": s2_all, "a_old_pct": 93.0, "a_new_pct": 47.0, "delta_a_old_pct": 5.5, "fm_cls_pct": 5.5, "discovered_clusters": 1, "core_purity_pct": 94.0, "gating": {"known_frr_on_stream_pct": 20.0, "unknown_purity_pct": 65.0}},
        ]
        sessions_wo = [
            sessions_b[0],
            sessions_b[1],
            {"session": 2, "a_all_pct": 53.0, "a_old_pct": 65.0, "a_new_pct": 31.0, "delta_a_old_pct": 33.5, "fm_cls_pct": 33.5, "gating": {"known_frr_on_stream_pct": 98.5, "unknown_purity_pct": 34.0}},
        ]
        return {
            "seed": seed,
            "stage0": {"tldnn_test_acc_pct": acc0},
            "stage1_osr": {
                "tldnn_ms_maha": {
                    "known_acc_pct": acc0,
                    "auroc_pct": auroc,
                    "fpr_at_tpr95_pct": 26.5,
                    "oscr_pct": auroc - 1.0,
                }
            },
            "blind_snr": {
                "m2m4": {
                    "known_test": {"mae_db": 5.5, "rmse_db": 6.8, "gates": {"ge_4db": {"f1_pct": 81.5}}},
                    "valid_unseen_unknown_test": {"mae_db": 5.3, "rmse_db": 6.3, "gates": {"ge_4db": {"f1_pct": 81.2}}},
                },
                "aux_snr": {
                    "known_test": {"mae_db": 0.9, "rmse_db": 1.8, "gates": {"ge_4db": {"f1_pct": 96.2}}},
                    "valid_unseen_unknown_test": {"mae_db": mae, "rmse_db": 2.5, "gates": {"ge_4db": {"f1_pct": 96.5}}},
                },
            },
            "table2_closed_loop": [
                {"order": "order_b", "method": "Ours (Full Model + SS)", "sessions": sessions_b},
            ],
            "clean10_closed_loop": {
                "openness_pct": 9.25,
                "runs": [
                    {"order": "order_clean", "method": "Ours (Full Model + SS)", "sessions": sessions_b},
                ],
            },
            "sync_ablation_order_b": [
                {"method": "W/O Prototype Sync", "sessions": sessions_wo},
                {"method": "With Prototype Sync (Ours)", "sessions": sessions_b},
            ],
        }

    seeds_data = [
        _make_seed_payload(42, 98.0, 91.0, 1.5, 79.0),
        _make_seed_payload(1024, 99.0, 92.0, 1.8, 80.0),
        _make_seed_payload(2025, 100.0, 93.0, 2.1, 81.0),
    ]
    agg = aggregate_multiseed_results(seeds_data)
    assert agg["seeds"] == [42, 1024, 2025]
    assert agg["n_seeds"] == 3
    assert agg["stage0"]["tldnn_test_acc_pct"]["mean"] == pytest.approx(99.0)
    assert agg["stage0"]["tldnn_test_acc_pct"]["std"] == pytest.approx(1.0)
    assert agg["stage1_osr"]["tldnn_ms_maha"]["auroc_pct"]["mean"] == pytest.approx(92.0)
    assert agg["stage1_osr"]["tldnn_ms_maha"]["auroc_pct"]["std"] == pytest.approx(1.0)
    assert agg["blind_snr"]["aux_snr"]["valid_unseen_unknown_test"]["mae_db"]["mean"] == pytest.approx(1.8)
    assert agg["blind_snr"]["aux_snr"]["valid_unseen_unknown_test"]["mae_db"]["std"] == pytest.approx(0.3)
    assert agg["table2_closed_loop"]["order_b"]["Ours (Full Model + SS)"]["s2_a_all_pct"]["mean"] == pytest.approx(80.0)
    assert agg["clean10_closed_loop"]["methods"]["Ours (Full Model + SS)"]["s2_core_purity_pct"]["mean"] == pytest.approx(94.0)
    assert agg["sync_ablation_order_b"]["With Prototype Sync (Ours)"]["s2_known_frr_pct"]["mean"] == pytest.approx(20.0)

    # Verify different seeds produce distinct, leakage-free train/val/test partitions
    rng = np.random.default_rng(0)
    raw = {
        (m, 10): rng.normal(size=(30, 2, 128)).astype(np.float32)
        for m in ("K0", "K1", "U0")
    }
    base_cfg = {
        "name": "tiny_ms",
        "snr_min": 10,
        "snr_max": 10,
        "ratios": {"train": 0.6, "val": 0.2, "test": 0.2},
        "known_classes": ["K0", "K1"],
        "unknown_classes": ["U0"],
    }
    b42 = SplitBundle(raw, {**base_cfg, "seed": 42})
    b1024 = SplitBundle(raw, {**base_cfg, "seed": 1024})
    assert b42.verify_disjoint() and b1024.verify_disjoint()
    assert not np.array_equal(b42.indices[("K0", 10)]["test"], b1024.indices[("K0", 10)]["test"])
