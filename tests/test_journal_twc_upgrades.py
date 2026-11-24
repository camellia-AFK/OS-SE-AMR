"""Unit and smoke tests for IEEE TWC / TCOM top-journal upgrades."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
import torch

from os_se_amr.continual import (
    DERLearner,
    DERWrapperBackbone,
    DualLwFLearner,
    EWCLearner,
    ReplayBuffer,
    WALearner,
    apply_weight_alignment,
    sync_prototypes_and_mahalanobis,
)
from os_se_amr.data import (
    RML2018_CLASSES,
    SplitBundle,
    apply_rf_channel_impairments,
    extract_rf_physics_features,
    load_dataset_dict,
    load_rml2018_hdf5,
    save_rml2018_cache,
)
from os_se_amr.discovery import FINCHDiscoverer, KMeansDiscoverer
from os_se_amr.engine import ClosedLoopConfig, ClosedLoopEvolver, build_zero_novelty_session
from os_se_amr.eval import (
    paired_significance_test,
    plot_covariance_anisotropy_spectrum,
    plot_phase_continuity_zoom,
    plot_sensitivity_grid,
    profile_system_overhead,
    render_booktabs_table,
    save_latex_bundle,
)
from os_se_amr.models import AuxSNRRegressor, TLDNN, build_backbone
from os_se_amr.osr import EnergyScorer, MahalanobisPrototypeScorer, OpenMaxScorer
from os_se_amr.utils import save_checkpoint


def _make_synth_mod_iq(length: int = 128, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    t = np.arange(length, dtype=np.float32)

    # Constant-envelope PSK
    bpsk = np.zeros((16, 2, length), dtype=np.float32)
    for i in range(16):
        ph = (rng.integers(0, 2, size=length // 8).repeat(8) * np.pi) + 0.05 * t
        bpsk[i, 0] = np.cos(ph)
        bpsk[i, 1] = np.sin(ph)

    # Multi-amplitude 16QAM
    levels = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float32) / np.sqrt(10.0)
    qam16 = np.zeros((16, 2, length), dtype=np.float32)
    for i in range(16):
        sym_i = rng.choice(levels, size=length // 8).repeat(8)
        sym_q = rng.choice(levels, size=length // 8).repeat(8)
        qam16[i, 0] = sym_i
        qam16[i, 1] = sym_q

    # Pure AWGN
    awgn = rng.normal(0.0, 0.5, size=(16, 2, length)).astype(np.float32)
    return torch.from_numpy(bpsk), torch.from_numpy(qam16), torch.from_numpy(awgn)


def test_rf_physics_features_and_impairments() -> None:
    for flen in (128, 1024):
        bpsk, qam16, awgn = _make_synth_mod_iq(length=flen, seed=10 + flen)
        f_psk = extract_rf_physics_features(bpsk)
        f_qam = extract_rf_physics_features(qam16)
        f_awgn = extract_rf_physics_features(awgn)

        assert f_psk.shape == (16, 10)
        assert f_qam.shape == (16, 10)
        assert torch.isfinite(f_psk).all()
        assert torch.isfinite(f_qam).all()
        assert torch.isfinite(f_awgn).all()

        d_psk_qam = float(torch.norm(f_psk.mean(dim=0) - f_qam.mean(dim=0)).item())
        d_psk_awgn = float(torch.norm(f_psk.mean(dim=0) - f_awgn.mean(dim=0)).item())
        assert d_psk_qam > 0.3
        assert d_psk_awgn > 0.3

    x_np = bpsk.numpy()
    x_id = apply_rf_channel_impairments(x_np, cfo_norm=0.0, phase_noise_std=0.0, seed=42)
    assert np.allclose(x_id, x_np)

    x_imp = apply_rf_channel_impairments(x_np, cfo_norm=0.005, phase_noise_std=0.05, seed=42)
    assert x_imp.shape == x_np.shape
    assert not np.allclose(x_imp, x_np)
    env_orig = np.sqrt(x_np[:, 0] ** 2 + x_np[:, 1] ** 2)
    env_imp = np.sqrt(x_imp[:, 0] ** 2 + x_imp[:, 1] ** 2)
    assert np.allclose(env_orig, env_imp, atol=1e-5)


def test_rml2018_hdf5_and_cache_roundtrip(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    raw_h5 = tmp_path / "synth_2018.hdf5"
    cache_h5 = tmp_path / "cache_2018.h5"

    classes_used = ["OOK", "BPSK", "QPSK", "16QAM", "FM", "GMSK"]
    snrs_used = [0, 10]
    per_cell = 20
    n_total = len(classes_used) * len(snrs_used) * per_cell

    rng = np.random.default_rng(42)
    x_arr = np.zeros((n_total, 1024, 2), dtype=np.float32)
    y_arr = np.zeros((n_total, len(RML2018_CLASSES)), dtype=np.float32)
    z_arr = np.zeros((n_total, 1), dtype=np.int16)

    idx = 0
    for c_name in classes_used:
        cid = RML2018_CLASSES.index(c_name)
        for snr in snrs_used:
            x_arr[idx : idx + per_cell] = rng.normal(float(cid) * 0.1, 0.2, size=(per_cell, 1024, 2)).astype(np.float32)
            y_arr[idx : idx + per_cell, cid] = 1.0
            z_arr[idx : idx + per_cell, 0] = snr
            idx += per_cell

    with h5py.File(raw_h5, "w") as f:
        f.create_dataset("X", data=x_arr)
        f.create_dataset("Y", data=y_arr)
        f.create_dataset("Z", data=z_arr)

    loaded_raw = load_rml2018_hdf5(raw_h5, max_per_cell=12, seed=42)
    assert len(loaded_raw) == len(classes_used) * len(snrs_used)
    for c_name in classes_used:
        for snr in snrs_used:
            assert loaded_raw[(c_name, snr)].shape == (12, 2, 1024)

    save_rml2018_cache(loaded_raw, cache_h5, dtype="float16")
    loaded_cache = load_dataset_dict(cache_h5)
    assert loaded_cache[("BPSK", 0)].dtype == np.float32
    assert loaded_cache[("BPSK", 0)].shape == (12, 2, 1024)

    split_cfg = {
        "name": "test_2018",
        "dataset_path": str(cache_h5),
        "seed": 42,
        "snr_min": 0,
        "snr_max": 10,
        "ratios": {"train": 0.6, "val": 0.2, "test": 0.2},
        "known_classes": ["BPSK", "QPSK", "16QAM", "FM"],
        "unknown_classes": ["OOK", "GMSK"],
    }
    bundle = SplitBundle.from_config(split_cfg)
    bundle.verify_disjoint()
    tr_ds, _, _ = bundle.get_closed_set_datasets()
    assert tr_ds.x.shape[1:] == (2, 1024)


@pytest.mark.parametrize("frame_len", [128, 1024])
def test_variable_length_backbones_and_resnet1d(frame_len: int) -> None:
    tldnn_cfg = {"name": "tldnn", "feature_dim": 64, "tldnn": {"seq_len": 128 if frame_len == 1024 else 32}}
    iqformer_cfg = {
        "name": "iqformer",
        "feature_dim": 64,
        "iqformer": {
            "stft_n_fft": 64 if frame_len == 1024 else 32,
            "stft_hop": 16 if frame_len == 1024 else 8,
            "seq_len": 128 if frame_len == 1024 else 32,
        },
    }
    resnet_cfg = {"name": "resnet1d", "feature_dim": 64, "resnet1d": {"base_channels": 32}}

    for cfg in (tldnn_cfg, iqformer_cfg, resnet_cfg):
        model = build_backbone(cfg, num_classes=7)
        x = torch.randn(4, 2, frame_len)
        logits, feat = model(x)
        assert logits.shape == (4, 7)
        assert feat.shape == (4, 64)

        ms = model.extract_multiscale_features(x)
        assert set(ms.keys()) >= {"stem", "pre_ln", "penult"}
        assert ms["penult"].shape == (4, 64)

        loss = logits.sum()
        loss.backward()

        model.expand_classifier(3)
        assert model.num_classes == 10
        logits2, _ = model(x)
        assert logits2.shape == (4, 10)


def test_physics_ms_mahaproto_and_sync() -> None:
    rng = np.random.default_rng(7)
    n_k = 60
    y_k = np.repeat(np.arange(3), n_k // 3)
    tr_feats = {
        "stem": rng.normal(0.0, 0.3, size=(n_k, 128)).astype(np.float32) + y_k[:, None] * 2.5,
        "pre_ln": rng.normal(0.0, 0.3, size=(n_k, 64)).astype(np.float32) + y_k[:, None] * 2.0,
        "physics": rng.normal(0.0, 0.2, size=(n_k, 10)).astype(np.float32) + y_k[:, None] * 1.5,
    }
    va_feats = {k: v + rng.normal(0.0, 0.05, size=v.shape).astype(np.float32) for k, v in tr_feats.items()}

    scorer = MahalanobisPrototypeScorer(
        layers=("stem", "pre_ln", "physics"),
        weights={"stem": 0.55, "pre_ln": 0.25, "physics": 0.20},
    )
    scorer.fit_multiscale(tr_feats, y_k, val_feats=va_feats)
    known_scores = scorer.score_multiscale(va_feats)
    unk_feats = {k: rng.normal(12.0, 0.3, size=(20, v.shape[1])).astype(np.float32) for k, v in tr_feats.items()}
    unk_scores = scorer.score_multiscale(unk_feats)
    assert float(known_scores.mean()) > float(unk_scores.mean())

    # End-to-end sync after expansion with include_physics=True
    backbone = build_backbone({"name": "resnet1d", "feature_dim": 32, "resnet1d": {"base_channels": 16}}, num_classes=4)
    x_old = rng.normal(0.0, 0.5, size=(40, 2, 128)).astype(np.float32)
    y_old = np.repeat(np.arange(4), 10)
    with torch.no_grad():
        x_old_t = torch.from_numpy(x_old)
        ms_old = {k: v.numpy() for k, v in backbone.extract_multiscale_features(x_old_t).items()}
        ms_old["physics"] = extract_rf_physics_features(x_old_t).numpy()
    scorer_bb = MahalanobisPrototypeScorer(
        layers=("stem", "pre_ln", "physics"),
        weights={"stem": 0.55, "pre_ln": 0.25, "physics": 0.20},
    )
    scorer_bb.fit_multiscale(ms_old, y_old)

    rb = ReplayBuffer(exemplars_per_class=10, selection_strategy="herding")
    for cid in range(4):
        rb.add_class_exemplars(cid, x_old[y_old == cid], features=ms_old["penult"][y_old == cid], seed=42)

    backbone.expand_classifier(2)
    x_new = rng.normal(0.5, 0.5, size=(20, 2, 128)).astype(np.float32)
    y_new = np.repeat(np.arange(2), 10)
    sync_res = sync_prototypes_and_mahalanobis(
        backbone=backbone,
        replay_buffer=rb,
        new_core_x=x_new,
        new_core_labels=y_new,
        prev_scorer=scorer_bb,
        device="cpu",
        include_physics=True,
    )
    synced_scorer = sync_res["scorer"]
    assert synced_scorer.classes == [0, 1, 2, 3, 4, 5]
    assert "physics" in synced_scorer.means and "physics" in synced_scorer.precisions


def test_weight_alignment_and_dual_lwf_upgrades() -> None:
    model = build_backbone({"name": "resnet1d", "feature_dim": 32, "resnet1d": {"base_channels": 16}}, num_classes=5)
    model.expand_classifier(2)
    with torch.no_grad():
        model.classifier.fc.weight.data[5:] *= 5.0

    gamma = apply_weight_alignment(model, num_old_classes=5)
    assert 0.05 < gamma < 0.5
    norms = torch.norm(model.classifier.fc.weight.data, p=2, dim=1)
    assert abs(float(norms[:5].mean().item()) - float(norms[5:].mean().item())) < 1e-4

    rng = np.random.default_rng(11)
    rb = ReplayBuffer(exemplars_per_class=8, selection_strategy="random")
    for cid in range(5):
        rb.add_class_exemplars(cid, rng.normal(float(cid) * 0.1, 0.3, size=(8, 2, 128)).astype(np.float32), seed=42)

    x_new = rng.normal(0.8, 0.3, size=(16, 2, 128)).astype(np.float32)
    y_new = np.repeat(np.arange(2), 8)
    learner = DualLwFLearner(
        epochs=2,
        steps_per_epoch=2,
        samples_per_class_per_step=4,
        use_rf_aug=False,
        use_weight_align=True,
        replay_ss_discovered=True,
        bic_max_steal=None,
    )
    # Re-create clean 5-class backbone before evolve
    base_m = build_backbone({"name": "resnet1d", "feature_dim": 32, "resnet1d": {"base_channels": 16}}, num_classes=5)
    rep = learner.evolve(
        backbone=base_m,
        replay_buffer=rb,
        new_samples=x_new,
        new_pseudo_labels=y_new,
        num_base_classes=4,
        device="cpu",
        seed=42,
    )
    assert base_m.num_classes == 7
    assert rep.num_old_classes == 5
    assert rep.num_new_classes == 2
    assert rep.wa_gamma > 0.0
    assert np.isfinite(rep.final_loss)


def test_sota_osr_cl_discovery_baselines() -> None:
    rng = np.random.default_rng(21)
    tr_lg = rng.normal(0.0, 1.0, size=(80, 4)).astype(np.float32)
    tr_y = np.repeat(np.arange(4), 20)
    for i in range(80):
        tr_lg[i, tr_y[i]] += 4.0

    te_lg = rng.normal(0.0, 1.0, size=(20, 4)).astype(np.float32)
    te_feat = rng.normal(0.0, 1.0, size=(20, 16)).astype(np.float32)
    e_sc = EnergyScorer(temperature=1.0).score(torch.from_numpy(te_lg), torch.from_numpy(te_feat))
    assert e_sc.shape == (20,) and torch.isfinite(e_sc).all()

    om = OpenMaxScorer(tailsize=10, alpha=3)
    om.fit_logits(tr_lg, tr_y)
    om_sc = om.score(torch.from_numpy(te_lg), torch.from_numpy(te_feat))
    assert om_sc.shape == (20,) and torch.isfinite(om_sc).all()

    # Discovery baselines on 2 well-separated clusters
    c1 = rng.normal(0.0, 0.05, size=(20, 64)).astype(np.float32)
    c1[:, 0] += 5.0
    c2 = rng.normal(0.0, 0.05, size=(20, 64)).astype(np.float32)
    c2[:, 1] += 5.0
    feats = {
        "stem": np.concatenate([c1, c2], axis=0),
        "pre_ln": np.concatenate([c1, c2], axis=0),
        "penult": np.concatenate([c1, c2], axis=0),
    }
    km = KMeansDiscoverer(k_range=(2, 4), min_cluster_size=8).discover(feats)
    fi = FINCHDiscoverer(min_cluster_size=8).discover(feats)
    assert km.k_hat == 2 and km.accepted
    assert fi.k_hat >= 2 and fi.accepted

    # CL baselines: EWC, WA, DER
    cfg = {"name": "resnet1d", "feature_dim": 16, "resnet1d": {"base_channels": 16}}
    rb = ReplayBuffer(exemplars_per_class=6, selection_strategy="random")
    for cid in range(4):
        rb.add_class_exemplars(cid, rng.normal(0.0, 0.3, size=(6, 2, 128)).astype(np.float32), seed=42)
    x_new = rng.normal(0.3, 0.3, size=(12, 2, 128)).astype(np.float32)
    y_new = np.repeat(np.arange(2), 6)

    m_ewc = build_backbone(cfg, num_classes=4)
    EWCLearner(epochs=2, batch_size=8, lambda_ewc=50.0).evolve(m_ewc, rb, x_new, y_new, device="cpu", seed=42)
    assert m_ewc.num_classes == 6

    m_wa = build_backbone(cfg, num_classes=4)
    WALearner(epochs=2, batch_size=8).evolve(m_wa, rb, x_new, y_new, device="cpu", seed=42)
    assert m_wa.num_classes == 6

    m_der = build_backbone(cfg, num_classes=4)
    wrapper, der_rep = DERLearner(epochs=2, batch_size=8).evolve(m_der, rb, x_new, y_new, cfg, device="cpu", seed=42)
    assert isinstance(wrapper, DERWrapperBackbone)
    assert wrapper.num_classes == 6
    assert der_rep.num_new_classes == 2
    ms_der = wrapper.extract_multiscale_features(torch.randn(2, 2, 128))
    assert set(ms_der.keys()) == {"stem", "pre_ln", "penult"}


def test_zero_novelty_session_and_evolver_hold(tmp_path: Path) -> None:
    rng = np.random.default_rng(99)
    raw_dict: dict[tuple[str, int], np.ndarray] = {}
    mods = ["K0", "K1", "K2", "K3", "U0", "U1"]
    for idx, mod in enumerate(mods):
        for snr in (6, 10):
            base = np.zeros((40, 2, 128), dtype=np.float32) + float(idx) * 0.25
            base += rng.normal(0.0, 0.1, size=base.shape).astype(np.float32)
            raw_dict[(mod, snr)] = base

    split_cfg = {
        "name": "zn_test",
        "seed": 42,
        "snr_min": 6,
        "snr_max": 10,
        "ratios": {"train": 0.6, "val": 0.2, "test": 0.2},
        "known_classes": ["K0", "K1", "K2", "K3"],
        "unknown_classes": ["U0", "U1"],
    }
    bundle = SplitBundle(raw_dict=raw_dict, split_cfg=split_cfg)
    model_cfg = {"name": "resnet1d", "feature_dim": 16, "resnet1d": {"base_channels": 16}}
    model = build_backbone(model_cfg, num_classes=4)
    ckpt = tmp_path / "base.pt"
    save_checkpoint(ckpt, model, optimizer=None, epoch=1, metrics={"val_acc": 0.9})

    tr_ds, va_ds, _ = bundle.get_closed_set_datasets()
    with torch.no_grad():
        tr_ms = {k: v.numpy() for k, v in model.extract_multiscale_features(torch.from_numpy(tr_ds.x)).items()}
        va_ms = {k: v.numpy() for k, v in model.extract_multiscale_features(torch.from_numpy(va_ds.x)).items()}
    scorer = MahalanobisPrototypeScorer(layers=("stem", "pre_ln"), weights={"stem": 0.7, "pre_ln": 0.3})
    scorer.fit_multiscale(tr_ms, tr_ds.y, val_feats=va_ms)

    aux_snr = AuxSNRRegressor()
    sess = build_zero_novelty_session(bundle, bundle.known_classes, samples_per_class_main=15, samples_per_class_neg=0, session_id=1, seed=42)
    assert sess.arriving_classes == []

    cl_cfg = ClosedLoopConfig(
        method_key="ours",
        gate_mode="m2m4",
        snr_threshold=25.0,
        min_cluster_size=20,
        min_samples=5,
        exemplars_per_class=8,
        seed=42,
    )
    evo = ClosedLoopEvolver().run_trajectory(
        method_label="ZeroNoveltyCheck",
        order_name="zero_novelty",
        cl_cfg=cl_cfg,
        sessions=[sess],
        base_ckpt_path=ckpt,
        base_scorer=scorer,
        snr_estimator=aux_snr,
        bundle=bundle,
        model_cfg=model_cfg,
        cil_cfg={"epochs": 1, "steps_per_epoch": 1, "samples_per_class_per_step": 4, "selection_strategy": "random"},
        device=torch.device("cpu"),
        logger=logging.getLogger("test_zn"),
    )
    s1 = evo["sessions"][1]
    assert s1["arriving_classes"] == []
    assert s1["false_discovery_triggered"] is False
    assert s1["false_heads_added"] == 0


def test_significance_profiler_and_latex_exports(tmp_path: Path) -> None:
    sig = paired_significance_test([92.1, 92.4, 92.0, 92.3], [90.0, 90.2, 89.9, 90.1])
    assert sig["mean_diff"] > 2.0
    assert sig["p_value"] < 0.01
    assert sig["t_stat"] > 5.0
    assert sig["significant_01"] is True

    for cfg, flen, nc in [
        ({"name": "resnet1d", "feature_dim": 64, "resnet1d": {"base_channels": 32}}, 128, 7),
        ({"name": "tldnn", "feature_dim": 64, "tldnn": {"seq_len": 128}}, 1024, 12),
    ]:
        bb = build_backbone(cfg, num_classes=nc)
        prof = profile_system_overhead(bb, None, AuxSNRRegressor(), frame_len=flen, num_classes=nc, batch_size=8, device="cpu")
        assert prof["params_total_k"] > 0.0
        assert prof["flops_per_frame_m"] > 0.0
        assert prof["full_online_gate_per_frame_ms"] > 0.0
        assert prof["replay_memory_mb"] > 0.0

    tex = render_booktabs_table("Demo", "tab:demo", ["Method", "Acc"], [["Ours", "95.0"]])
    written = save_latex_bundle({"demo": tex}, tmp_path / "latex")
    assert (tmp_path / "latex" / "demo.tex").exists()
    assert (tmp_path / "latex" / "all_tables.tex").exists()
    text_all = (tmp_path / "latex" / "all_tables.tex").read_text(encoding="utf-8")
    assert "\\toprule" in text_all and "\\bottomrule" in text_all
    assert len(written) == 2

    rng = np.random.default_rng(5)
    p1 = plot_covariance_anisotropy_spectrum({"stem": np.geomspace(1.0, 1e-4, 32)}, tmp_path / "figs" / "spec")
    t = np.arange(128, dtype=np.float32)
    iq = np.stack([np.cos(0.3 * t), np.sin(0.3 * t)], axis=0)
    p2 = plot_phase_continuity_zoom(iq, iq.copy(), 64, tmp_path / "figs" / "phase")
    p3 = plot_sensitivity_grid(
        [{"title": "A", "xlabel": "x", "ylabel": "y", "xs": [1, 2], "series": {"s": [80.0, 82.0]}}],
        tmp_path / "figs" / "sens",
    )
    for paths in (p1, p2, p3):
        assert len(paths) == 3
        for p in paths:
            assert p.exists() and p.stat().st_size > 0
