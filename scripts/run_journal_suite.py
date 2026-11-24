"""IEEE TWC / TCOM Top-Journal Benchmark Runner (Colab L4 24 GB + Local Ready, Atomic --resume)."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (SRC_DIR, SCRIPTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from os_se_amr.data import (
    RMLDataset,
    SplitBundle,
    apply_rf_channel_impairments,
    extract_rf_physics_features,
    load_dataset_dict,
    load_rml2018_hdf5,
    save_rml2018_cache,
)
from os_se_amr.discovery import FINCHDiscoverer, HDBSCANDiscoverer, KMeansDiscoverer
from os_se_amr.discovery.ss_augment import ss_iq_pair, ss_phase_aligned_pair
from os_se_amr.engine import (
    ClosedLoopConfig,
    ClosedLoopEvolver,
    ClosedSetTrainer,
    build_closed_loop_mixed_sessions,
    build_zero_novelty_session,
)
from os_se_amr.eval import (
    aggregate_multiseed_results,
    evaluate_clustering_diagnostics,
    evaluate_osr,
    format_mean_std,
    paired_significance_test,
    plot_covariance_anisotropy_spectrum,
    plot_phase_continuity_zoom,
    plot_sensitivity_grid,
    profile_system_overhead,
    render_booktabs_table,
    save_latex_bundle,
    summarize_scalar,
)
from os_se_amr.models import AuxSNRRegressor, BaseBackbone, build_backbone, train_aux_snr_regressor
from os_se_amr.osr import (
    EnergyScorer,
    MLSScorer,
    MahalanobisPrototypeScorer,
    OpenMaxScorer,
    SoftmaxScorer,
    calibrate_threshold,
)
from os_se_amr.utils import (
    load_checkpoint,
    load_yaml,
    resolve_path,
    save_json,
    seed_everything,
    setup_logger,
)
from run_stage1 import extract_multiscale_all, extract_predictions_and_scores
from run_stage4 import run_snr_estimator_section


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IEEE TWC/TCOM Top-Journal Benchmark Suite")
    parser.add_argument("--config", type=str, default="configs/journal_twc.yaml", help="Master YAML path")
    parser.add_argument("--batch", type=str, default="all", choices=["1", "2", "3", "4", "all"], help="Batch to run")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory (e.g. Drive path)")
    parser.add_argument("--device", type=str, default=None, help="Override device (cuda or cpu)")
    parser.add_argument("--amp", action="store_true", help="Force BF16/FP16 AMP on CUDA")
    parser.add_argument("--resume", action="store_true", help="Resume from saved per-step JSON & checkpoints")
    return parser.parse_args()


def _load_or_run(
    cache_path: Path,
    resume: bool,
    fn: Any,
) -> Any:
    if resume and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    res = fn()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(res, cache_path)
    return res


def _train_or_load_stage0(
    bundle: SplitBundle,
    classes: list[str],
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    out_dir: Path,
    device: torch.device,
    resume: bool,
    seed: int,
    logger: Any,
) -> tuple[Path, float]:
    """Train or reuse closed-set backbone strictly on `classes` Train/Val."""
    ckpt_path = out_dir / "checkpoints" / "best.pt"
    metrics_path = out_dir / "metrics.json"
    if resume and ckpt_path.exists() and metrics_path.exists():
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        return ckpt_path, round(float(m["test_metrics"]["overall_acc"]) * 100.0, 2)

    seed_everything(seed, deterministic=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    label_map = {c: i for i, c in enumerate(classes)}
    tr_ds = bundle.get_dataset("train", classes, bundle.main_snrs, label_map)
    va_ds = bundle.get_dataset("val", classes, bundle.main_snrs, label_map)
    te_ds = bundle.get_dataset("test", classes, bundle.main_snrs, label_map)

    bs = int(train_cfg.get("batch_size", 256))
    nw = int(train_cfg.get("num_workers", 0)) if device.type == "cuda" else 0
    pin = device.type == "cuda"
    tr_loader = DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=pin, persistent_workers=(nw > 0))
    va_loader = DataLoader(va_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin)
    te_loader = DataLoader(te_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin)

    model = build_backbone(copy.deepcopy(model_cfg), num_classes=len(classes))
    trainer = ClosedSetTrainer(model=model, device=device, train_cfg=train_cfg, output_dir=out_dir, logger=logger)
    train_sum = trainer.fit(tr_loader, va_loader, classes)
    test_m = trainer.evaluate(te_loader, classes)
    acc_pct = round(float(test_m["overall_acc"]) * 100.0, 2)
    save_json({"train_summary": train_sum, "test_metrics": test_m}, metrics_path)
    return ckpt_path, acc_pct


def _extract_ms_with_physics(
    model: BaseBackbone,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Extract multi-scale deep features + 10-d RF physics features + logits/preds/labels."""
    feats, preds, labels = extract_multiscale_all(model, loader, device)
    model.eval()
    logits_parts, phys_parts = [], []
    with torch.no_grad():
        for x, _, _ in loader:
            xb = x.to(device, non_blocking=True)
            lg, _ = model(xb)
            logits_parts.append(lg.cpu().numpy())
            phys_parts.append(extract_rf_physics_features(xb).cpu().numpy())
    feats["physics"] = np.concatenate(phys_parts, axis=0)
    return feats, np.concatenate(logits_parts, axis=0), preds, labels


def _evaluate_osr_suite(
    bundle: SplitBundle,
    model_cfg: dict[str, Any],
    ckpt_path: Path,
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[list[dict[str, Any]], MahalanobisPrototypeScorer, MahalanobisPrototypeScorer]:
    """Evaluate 6 zero-retrain static OSR scorers strictly on `bundle`."""
    num_k = len(bundle.known_classes)
    model = build_backbone(copy.deepcopy(model_cfg), num_classes=num_k).to(device)
    load_checkpoint(ckpt_path, model, device=device)

    tr_ds, va_ds, _ = bundle.get_closed_set_datasets()
    k_te_ds, u_te_ds = bundle.get_open_set_test_datasets()
    idx_to_unk = {idx: cls for cls, idx in bundle.all_to_idx.items() if cls in bundle.unknown_classes}

    bs = int(cfg["train"].get("batch_size", 256))
    tr_l = DataLoader(tr_ds, batch_size=bs, shuffle=False)
    va_l = DataLoader(va_ds, batch_size=bs, shuffle=False)
    k_l = DataLoader(k_te_ds, batch_size=bs, shuffle=False)
    u_l = DataLoader(u_te_ds, batch_size=bs, shuffle=False)

    tr_f, tr_lg, _, tr_y = _extract_ms_with_physics(model, tr_l, device)
    va_f, va_lg, _, _ = _extract_ms_with_physics(model, va_l, device)
    k_f, k_lg, k_p, k_y = _extract_ms_with_physics(model, k_l, device)
    u_f, u_lg, _, u_y = _extract_ms_with_physics(model, u_l, device)

    rows: list[dict[str, Any]] = []
    for name, scorer in [("Softmax", SoftmaxScorer()), ("MLS", MLSScorer()), ("Energy", EnergyScorer(1.0))]:
        va_s = scorer.score(torch.from_numpy(va_lg), torch.from_numpy(va_f["penult"])).numpy()
        thr = calibrate_threshold(va_s, 0.95)
        k_s = scorer.score(torch.from_numpy(k_lg), torch.from_numpy(k_f["penult"])).numpy()
        u_s = scorer.score(torch.from_numpy(u_lg), torch.from_numpy(u_f["penult"])).numpy()
        m = evaluate_osr(k_s, k_p, k_y, u_s, u_y, idx_to_unk, thr, 0.95)
        rows.append(_osr_row(name, m))

    openmax = OpenMaxScorer(tailsize=20, alpha=min(3, num_k))
    openmax.fit_logits(tr_lg, tr_y)
    va_om = openmax.score(torch.from_numpy(va_lg), torch.from_numpy(va_f["penult"])).numpy()
    thr_om = calibrate_threshold(va_om, 0.95)
    k_om = openmax.score(torch.from_numpy(k_lg), torch.from_numpy(k_f["penult"])).numpy()
    u_om = openmax.score(torch.from_numpy(u_lg), torch.from_numpy(u_f["penult"])).numpy()
    rows.append(_osr_row("OpenMax", evaluate_osr(k_om, k_p, k_y, u_om, u_y, idx_to_unk, thr_om, 0.95)))

    maha_std = MahalanobisPrototypeScorer(layers=("stem", "pre_ln"), weights={"stem": 0.7, "pre_ln": 0.3})
    maha_std.fit_multiscale(tr_f, tr_y, val_feats=va_f)
    thr_std = calibrate_threshold(maha_std.score_multiscale(va_f), 0.95)
    rows.append(
        _osr_row(
            "MS-MahaProto",
            evaluate_osr(maha_std.score_multiscale(k_f), k_p, k_y, maha_std.score_multiscale(u_f), u_y, idx_to_unk, thr_std, 0.95),
        )
    )

    pm_cfg = cfg["physics_maha"]
    maha_phys = MahalanobisPrototypeScorer(layers=tuple(pm_cfg["layers"]), weights=dict(pm_cfg["weights"]))
    maha_phys.fit_multiscale(tr_f, tr_y, val_feats=va_f)
    thr_ph = calibrate_threshold(maha_phys.score_multiscale(va_f), 0.95)
    rows.append(
        _osr_row(
            "Physics-MS-MahaProto (Ours)",
            evaluate_osr(maha_phys.score_multiscale(k_f), k_p, k_y, maha_phys.score_multiscale(u_f), u_y, idx_to_unk, thr_ph, 0.95),
        )
    )
    return rows, maha_std, maha_phys


def _osr_row(name: str, m: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": name,
        "known_acc_pct": round(float(m["known_acc"]) * 100.0, 2),
        "auroc_pct": round(float(m["auroc"]) * 100.0, 2),
        "fpr_at_tpr95_pct": round(float(m["fpr_at_tpr95"]) * 100.0, 2),
        "oscr_pct": round(float(m["oscr"]) * 100.0, 2),
        "per_unknown_auroc_pct": {k: round(float(v) * 100.0, 2) for k, v in m.get("per_unknown_class_auroc", {}).items()},
    }


def _train_or_load_aux_snr(
    bundle: SplitBundle,
    snrs: list[int],
    ckpt_path: Path,
    snr_cfg: dict[str, Any],
    device: torch.device,
    resume: bool,
    logger: Any,
) -> AuxSNRRegressor:
    aux = AuxSNRRegressor().to(device)
    if resume and ckpt_path.exists():
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        aux.load_state_dict(payload["model_state_dict"])
        return aux

    tr = bundle.get_dataset("train", bundle.known_classes, snrs, bundle.known_to_idx)
    va = bundle.get_dataset("val", bundle.known_classes, snrs, bundle.known_to_idx)
    bs = int(snr_cfg.get("batch_size", 512))
    pin = device.type == "cuda"
    train_aux_snr_regressor(
        model=aux,
        train_loader=DataLoader(tr, batch_size=bs, shuffle=True, pin_memory=pin),
        val_loader=DataLoader(va, batch_size=bs, shuffle=False, pin_memory=pin),
        device=device,
        epochs=int(snr_cfg.get("epochs", 30)),
        lr=float(snr_cfg.get("lr", 2e-3)),
        weight_decay=float(snr_cfg.get("weight_decay", 1e-4)),
        patience=int(snr_cfg.get("patience", 8)),
        ckpt_path=ckpt_path,
        logger=logger,
    )
    return aux


def _make_journal_cl_cfg(
    cfg: dict[str, Any],
    method_key: str,
    include_physics: bool = False,
    use_weight_align: bool = False,
    replay_ss_discovered: bool = False,
    ss_mode: str = "ss_phase_aligned",
    snr_threshold: float | None = None,
    merge_cosine_threshold: float | None = None,
    exemplars_per_class: int | None = None,
    seed: int = 42,
) -> ClosedLoopConfig:
    cl = cfg["closed_loop"]
    cil = cfg["continual"]
    return ClosedLoopConfig(
        method_key=method_key,
        gate_mode="aux_snr",
        snr_threshold=float(cl.get("snr_threshold", 4.0) if snr_threshold is None else snr_threshold),
        target_tpr=float(cl.get("target_tpr", 0.90)),
        ss_mode=ss_mode,
        ss_expand_factor=int(cl.get("ss_expand_factor", 4)),
        ss_max_per_cluster=int(cl.get("ss_max_per_cluster", 2000)),
        sync_prototypes=True,
        exemplars_per_class=int(cil.get("exemplars_per_class", 100) if exemplars_per_class is None else exemplars_per_class),
        min_cluster_size=int(cl.get("min_cluster_size", 100)),
        min_samples=int(cl.get("min_samples", 20)),
        silhouette_threshold=float(cl.get("silhouette_threshold", 0.40)),
        core_prob_threshold=float(cl.get("core_prob_threshold", 0.75)),
        core_top_ratio=float(cl.get("core_top_ratio", 0.60)),
        allow_single_cluster=bool(cl.get("allow_single_cluster", True)),
        merge_cosine_threshold=float(cl.get("merge_cosine_threshold", 0.80) if merge_cosine_threshold is None else merge_cosine_threshold),
        min_cluster_silhouette=float(cl.get("min_cluster_silhouette", 0.40)),
        min_cluster_cohesion=float(cl.get("min_cluster_cohesion", 0.85)),
        ss_min_cluster_silhouette=float(cl.get("ss_min_cluster_silhouette", 0.45)),
        ss_min_cluster_cohesion=float(cl.get("ss_min_cluster_cohesion", 0.90)),
        ss_max_core_size=int(cl.get("ss_max_core_size", 450)),
        include_physics=include_physics,
        use_weight_align=use_weight_align,
        replay_ss_discovered=replay_ss_discovered,
        use_amp=bool(cfg.get("use_amp", False)),
        seed=seed,
    )


def _run_oracle_joint_upper_bound(
    bundle: SplitBundle,
    order_name: str,
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    out_dir: Path,
    device: torch.device,
    seed: int,
    logger: Any,
) -> dict[str, Any]:
    """Train joint full-label supervised models sequentially across S0 -> S1 -> ... as upper bound."""
    classes = list(bundle.known_classes)
    sessions_out = []
    _, acc0 = _train_or_load_stage0(
        bundle, classes, model_cfg, train_cfg, out_dir / "s0", device, resume=True, seed=seed, logger=logger
    )
    sessions_out.append({"session": 0, "num_classes": len(classes), "a_all_pct": acc0})
    for idx, arriving in enumerate(bundle.split_cfg["arrival_orders"][order_name], start=1):
        classes = classes + [str(c) for c in arriving]
        _, acc_s = _train_or_load_stage0(
            bundle, classes, model_cfg, train_cfg, out_dir / f"s{idx}", device, resume=True, seed=seed + idx, logger=logger
        )
        sessions_out.append({"session": idx, "num_classes": len(classes), "a_all_pct": acc_s})
    return {"order": order_name, "method": "Oracle Joint (Upper Bound)", "sessions": sessions_out}


def run_batch_1(cfg: dict[str, Any], out_root: Path, device: torch.device, resume: bool, logger: Any) -> dict[str, Any]:
    """Batch 1: Method upgrades (Physics-Maha + WA/Replay-SS), SOTA baselines, 3 backbones, Zero-Novelty."""
    b1_dir = out_root / "batch1"
    seed = int(cfg.get("seed", 42))
    bundle_a = SplitBundle.from_config(load_yaml(resolve_path(cfg["splits"]["rml2016a"])))
    bundle_c10 = SplitBundle(raw_dict=bundle_a.raw_dict, split_cfg=load_yaml(resolve_path(cfg["splits"]["rml2016a_clean10"])))
    bundle_b = SplitBundle(raw_dict=bundle_a.raw_dict, split_cfg=load_yaml(resolve_path(cfg["splits"]["rml2016a_config_b"])))

    # 1. Stage 0 backbones on Config A & Config B
    ckpts_a: dict[str, Path] = {}
    for bname, cfg_key in [("tldnn", "tldnn_128"), ("iqformer", "iqformer_128"), ("resnet1d", "resnet1d_128")]:
        ckpt, acc = _train_or_load_stage0(
            bundle_a, bundle_a.known_classes, cfg["models"][cfg_key], cfg["train"],
            b1_dir / "stage0" / f"{bname}_config_a", device, resume=resume, seed=seed, logger=logger,
        )
        ckpts_a[bname] = ckpt
        logger.info("[Batch 1 | Stage 0 Config A | %-8s] Test Acc=%.2f%%", bname, acc)

    ckpt_b, acc_b = _train_or_load_stage0(
        bundle_b, bundle_b.known_classes, cfg["models"]["tldnn_128"], cfg["train"],
        b1_dir / "stage0" / "tldnn_config_b", device, resume=resume, seed=seed, logger=logger,
    )
    logger.info("[Batch 1 | Stage 0 Config B | tldnn   ] Test Acc=%.2f%%", acc_b)

    # 2. Static OSR 6+ scorers (Config A & Config B)
    osr_a, scorer_std_a, scorer_phys_a = _evaluate_osr_suite(bundle_a, cfg["models"]["tldnn_128"], ckpts_a["tldnn"], cfg, device)
    osr_b, _, _ = _evaluate_osr_suite(bundle_b, cfg["models"]["tldnn_128"], ckpt_b, cfg, device)

    # 3. Blind AuxSNR on Session 0 Known
    aux_snr = _train_or_load_aux_snr(
        bundle_a, [int(s) for s in cfg["snr_estimator"]["all_train_snrs"]],
        b1_dir / "checkpoints" / f"aux_snr_seed{seed}.pt", cfg["snr_estimator"], device, resume, logger,
    )

    # 4. 8-method Closed-Loop CIL Benchmark on Clean10 & Order B
    st_cfg = cfg["stream"]
    sess_clean = build_closed_loop_mixed_sessions(
        bundle_c10, "order_clean", int(st_cfg["samples_per_class_main"]), int(st_cfg["samples_per_class_neg"]),
        float(st_cfg["known_ratio"]), seed,
    )
    evolver = ClosedLoopEvolver()

    def _run_c10_cl() -> list[dict[str, Any]]:
        specs = [
            ("Static Baseline", "static", False, False, False, "none", scorer_std_a),
            ("Naive Fine-Tune", "naive", False, False, False, "none", scorer_std_a),
            ("EWC (PNAS 2017)", "ewc", False, False, False, "none", scorer_std_a),
            ("iCaRL (CVPR 2017)", "icarl", False, False, False, "none", scorer_std_a),
            ("WA (CVPR 2020)", "wa", False, False, False, "none", scorer_std_a),
            ("DER (CVPR 2021)", "der", False, False, False, "none", scorer_std_a),
            ("Ours (Base Stage 4)", "ours", False, False, False, "ss_phase_aligned", scorer_std_a),
            ("Ours+WA+ReplaySS+Phys", "ours", True, True, True, "ss_phase_aligned", scorer_phys_a),
        ]
        out = []
        for lbl, mk, ip, uwa, rss, ssm, sc in specs:
            cl_cfg = _make_journal_cl_cfg(cfg, mk, ip, uwa, rss, ssm, seed=seed)
            out.append(
                evolver.run_trajectory(
                    lbl, "order_clean", cl_cfg, sess_clean, ckpts_a["tldnn"], sc, aux_snr,
                    bundle_c10, cfg["models"]["tldnn_128"], cfg["continual"], device, logger,
                )
            )
        out.append(
            _run_oracle_joint_upper_bound(
                bundle_c10, "order_clean", cfg["models"]["tldnn_128"], cfg["train"], b1_dir / "oracle_clean10", device, seed, logger
            )
        )
        return out

    cl_clean10 = _load_or_run(b1_dir / "clean10_8methods.json", resume, _run_c10_cl)

    # 5. Discovery clustering baseline matrix on Clean10 S1 & S2 gated B_unk
    def _run_discovery_cmp() -> list[dict[str, Any]]:
        m = build_backbone(copy.deepcopy(cfg["models"]["tldnn_128"]), len(bundle_c10.known_classes)).to(device)
        load_checkpoint(ckpts_a["tldnn"], m, device=device)
        m.eval()
        out_rows = []
        for s_idx, sess in enumerate(sess_clean[:1], start=1):
            with torch.no_grad():
                x_t = torch.from_numpy(sess.x_iq).to(device)
                ms_l, _, _ = extract_multiscale_all(m, DataLoader(RMLDataset(sess.x_iq, sess.true_labels, sess.snrs), batch_size=256), device)
                snr_e = aux_snr.predict_snr(x_t).cpu().numpy()
            sc = scorer_std_a.score_multiscale(ms_l)
            thr = calibrate_threshold(scorer_std_a.score_multiscale(extract_multiscale_all(m, DataLoader(bundle_c10.get_closed_set_datasets()[1], batch_size=256), device)[0]), 0.90)
            mask = (sc < thr) & (snr_e >= float(cfg["closed_loop"]["snr_threshold"]))
            ms_b = {k: v[mask] for k, v in ms_l.items()}
            y_b = sess.true_labels[mask]
            for name, disc_obj in [
                ("K-Means (Auto-K)", KMeansDiscoverer(k_range=(2, 6), min_cluster_size=50)),
                ("FINCH (CVPR 2019)", FINCHDiscoverer(min_cluster_size=50)),
                ("Raw HDBSCAN", HDBSCANDiscoverer(min_cluster_size=100, min_samples=20, silhouette_threshold=0.40, allow_single_cluster=True)),
                ("HDBSCAN+MergePrune (Ours)", HDBSCANDiscoverer(min_cluster_size=100, min_samples=20, silhouette_threshold=0.40, allow_single_cluster=True, merge_cosine_threshold=0.80, min_cluster_silhouette=0.40, min_cluster_cohesion=0.85)),
            ]:
                dres = disc_obj.discover(ms_b)
                v = dres.cluster_ids >= 0
                diag = evaluate_clustering_diagnostics(y_b[v], dres.cluster_ids[v]) if v.any() else {"nmi": 0.0, "strict_1to1_acc": 0.0, "many_to_1_purity": 0.0}
                out_rows.append(
                    {
                        "session": s_idx,
                        "method": name,
                        "k_hat": dres.k_hat,
                        "raw_k_hat": dres.raw_k_hat,
                        "nmi": round(float(diag["nmi"]), 4),
                        "strict_1to1_pct": round(float(diag["strict_1to1_acc"]) * 100.0, 2),
                        "purity_pct": round(float(diag["many_to_1_purity"]) * 100.0, 2),
                    }
                )
        return out_rows

    disc_cmp = _load_or_run(b1_dir / "discovery_baselines.json", resume, _run_discovery_cmp)

    # 6. Cross-backbone closed loop (TLDNN, IQFormer, ResNet1D on Clean10)
    def _run_cross_backbone() -> list[dict[str, Any]]:
        rows = []
        for bname, cfg_key in [("TLDNN", "tldnn_128"), ("IQFormer", "iqformer_128"), ("ResNet1D", "resnet1d_128")]:
            key = bname.lower()
            ckpt = ckpts_a[key]
            m = build_backbone(copy.deepcopy(cfg["models"][cfg_key]), len(bundle_c10.known_classes)).to(device)
            load_checkpoint(ckpt, m, device=device)
            tr_d, va_d, _ = bundle_c10.get_closed_set_datasets()
            tr_f, _, _, tr_y = _extract_ms_with_physics(m, DataLoader(tr_d, batch_size=256), device)
            va_f, _, _, _ = _extract_ms_with_physics(m, DataLoader(va_d, batch_size=256), device)
            sc = MahalanobisPrototypeScorer(layers=tuple(cfg["physics_maha"]["layers"]), weights=dict(cfg["physics_maha"]["weights"]))
            sc.fit_multiscale(tr_f, tr_y, val_feats=va_f)
            cl_cfg = _make_journal_cl_cfg(cfg, "ours", True, True, True, "ss_phase_aligned", seed=seed)
            rows.append(
                evolver.run_trajectory(
                    f"{bname} (Ours Full)", "order_clean", cl_cfg, sess_clean, ckpt, sc, aux_snr,
                    bundle_c10, cfg["models"][cfg_key], cfg["continual"], device, logger,
                )
            )
        return rows

    cross_bb = _load_or_run(b1_dir / "cross_backbone_clean10.json", resume, _run_cross_backbone)

    # 7. Zero-Novelty Anti-Hallucination Stress Test
    def _run_zero_nov() -> dict[str, Any]:
        zn_sess = [build_zero_novelty_session(bundle_c10, bundle_c10.known_classes, session_id=1, seed=seed)]
        cl_cfg = _make_journal_cl_cfg(cfg, "ours", True, True, True, "ss_phase_aligned", seed=seed)
        return evolver.run_trajectory(
            "Zero-Novelty Hold", "zero_novelty", cl_cfg, zn_sess, ckpts_a["tldnn"], scorer_phys_a, aux_snr,
            bundle_c10, cfg["models"]["tldnn_128"], cfg["continual"], device, logger,
        )

    zero_nov = _load_or_run(b1_dir / "zero_novelty.json", resume, _run_zero_nov)
    out = {
        "osr_config_a": osr_a,
        "osr_config_b": osr_b,
        "clean10_closed_loop": cl_clean10,
        "discovery_comparison": disc_cmp,
        "cross_backbone_clean10": cross_bb,
        "zero_novelty": zero_nov,
    }
    save_json(out, b1_dir / "batch1_summary.json")
    return out


def _run_dataset_suite(
    tag: str,
    split_cfg_path: str,
    order_name: str,
    model_cfg: dict[str, Any],
    snr_list: list[int],
    n_main: int,
    n_neg: int,
    cfg: dict[str, Any],
    out_dir: Path,
    device: torch.device,
    resume: bool,
    logger: Any,
) -> dict[str, Any]:
    seed = int(cfg.get("seed", 42))
    bundle = SplitBundle.from_config(load_yaml(resolve_path(split_cfg_path)))
    bundle.verify_disjoint()
    ckpt, acc0 = _train_or_load_stage0(
        bundle, bundle.known_classes, model_cfg, cfg["train"], out_dir / "stage0", device, resume, seed, logger
    )
    osr_rows, _, sc_phys = _evaluate_osr_suite(bundle, model_cfg, ckpt, cfg, device)
    aux_snr = _train_or_load_aux_snr(
        bundle, snr_list, out_dir / f"aux_snr_{tag}.pt", cfg["snr_estimator"], device, resume, logger
    )
    sessions = build_closed_loop_mixed_sessions(
        bundle, order_name, n_main, n_neg, float(cfg["stream"]["known_ratio"]), seed
    )
    evolver = ClosedLoopEvolver()
    runs = []
    for lbl, mk, uwa, rss in [
        ("Static Baseline", "static", False, False),
        ("iCaRL (CVPR 2017)", "icarl", False, False),
        ("WA (CVPR 2020)", "wa", False, False),
        ("DER (CVPR 2021)", "der", False, False),
        ("Ours (Full TWC)", "ours", True, True),
    ]:
        cl_cfg = _make_journal_cl_cfg(cfg, mk, True, uwa, rss, "ss_phase_aligned", seed=seed)
        runs.append(
            evolver.run_trajectory(
                lbl, order_name, cl_cfg, sessions, ckpt, sc_phys, aux_snr, bundle, model_cfg, cfg["continual"], device, logger
            )
        )
    runs.append(
        _run_oracle_joint_upper_bound(bundle, order_name, model_cfg, cfg["train"], out_dir / "oracle", device, seed, logger)
    )
    res = {"tag": tag, "openness_pct": round(bundle.openness * 100.0, 2), "stage0_acc_pct": acc0, "osr": osr_rows, "closed_loop": runs}
    save_json(res, out_dir / f"{tag}_summary.json")
    return res


def run_batch_2(cfg: dict[str, Any], out_root: Path, device: torch.device, resume: bool, logger: Any) -> dict[str, Any]:
    """Batch 2: Cross-dataset validation on RML2016.10b and RML2018.01a (1024-pt, 5-step long horizon)."""
    b2_dir = out_root / "batch2"
    out: dict[str, Any] = {}

    cfg_10b = load_yaml(resolve_path(cfg["splits"]["rml2016b"]))
    if resolve_path(cfg_10b["dataset_path"]).exists():
        out["rml2016b"] = _load_or_run(
            b2_dir / "rml2016b_summary.json",
            resume,
            lambda: _run_dataset_suite(
                "rml2016b", cfg["splits"]["rml2016b"], "order_clean", cfg["models"]["tldnn_128"],
                [int(s) for s in cfg["snr_estimator"]["all_train_snrs"]],
                int(cfg["stream"]["samples_per_class_main"]), int(cfg["stream"]["samples_per_class_neg"]),
                cfg, b2_dir / "rml2016b", device, resume, logger,
            ),
        )
    else:
        logger.warning("[Batch 2] Skipping RML2016.10b (file not found: %s)", cfg_10b["dataset_path"])

    cfg_2018 = load_yaml(resolve_path(cfg["splits"]["rml2018"]))
    raw_2018 = resolve_path("data/GOLD_XYZ_OSC.0001_1024.hdf5")
    cache_2018 = resolve_path(cfg_2018["dataset_path"])
    if cache_2018.exists() or raw_2018.exists():
        if not cache_2018.exists() and raw_2018.exists():
            logger.info("[Batch 2] Building compact RML2018.01a cache at %s ...", cache_2018)
            raw_d = load_rml2018_hdf5(
                raw_2018,
                max_per_cell=int(cfg_2018.get("max_per_cell", 1000)),
                seed=int(cfg_2018.get("seed", 42)),
            )
            save_rml2018_cache(raw_d, cache_2018)
        out["rml2018"] = _load_or_run(
            b2_dir / "rml2018_summary.json",
            resume,
            lambda: _run_dataset_suite(
                "rml2018", cfg["splits"]["rml2018"], "order_5step", cfg["models"]["tldnn_1024"],
                [int(s) for s in cfg["snr_estimator"]["rml2018_train_snrs"]],
                int(cfg["stream"]["rml2018_samples_main"]), int(cfg["stream"]["rml2018_samples_neg"]),
                cfg, b2_dir / "rml2018", device, resume, logger,
            ),
        )
    else:
        logger.warning("[Batch 2] Skipping RML2018.01a (cache/HDF5 not found: %s)", cfg_2018["dataset_path"])

    save_json(out, b2_dir / "batch2_summary.json")
    return out


def run_batch_3(cfg: dict[str, Any], out_root: Path, device: torch.device, resume: bool, logger: Any) -> dict[str, Any]:
    """Batch 3: 5-seed statistical significance, 4-panel sensitivity sweep, & CFO/phase-noise DOCO++."""
    b3_dir = out_root / "batch3"
    base_seed = int(cfg.get("seed", 42))
    raw_a = load_dataset_dict(load_yaml(resolve_path(cfg["splits"]["rml2016a"]))["dataset_path"])
    c10_cfg = load_yaml(resolve_path(cfg["splits"]["rml2016a_clean10"]))

    # 1. 5-Seed statistical significance (Clean10 Physics+WA+ReplaySS vs Base Stage4 & Static)
    def _run_5seed() -> dict[str, Any]:
        seeds = [int(s) for s in cfg["multiseed"]["seeds"]]
        per_seed = []
        for s in seeds:
            scfg = copy.deepcopy(c10_cfg)
            scfg["seed"] = s
            bnd = SplitBundle(raw_dict=raw_a, split_cfg=scfg)
            sdir = b3_dir / "seeds" / f"s{s}"
            ckpt, acc0 = _train_or_load_stage0(
                bnd, bnd.known_classes, cfg["models"]["tldnn_128"], cfg["train"], sdir / "s0", device, resume, s, logger
            )
            osr_r, sc_std, sc_ph = _evaluate_osr_suite(bnd, cfg["models"]["tldnn_128"], ckpt, cfg, device)
            aux = _train_or_load_aux_snr(
                bnd, [int(x) for x in cfg["snr_estimator"]["all_train_snrs"]],
                sdir / f"aux_snr_s{s}.pt", cfg["snr_estimator"], device, resume, logger,
            )
            sess = build_closed_loop_mixed_sessions(bnd, "order_clean", 2000, 600, 0.5, s)
            evo = ClosedLoopEvolver()
            res_base = evo.run_trajectory(
                "Base", "order_clean", _make_journal_cl_cfg(cfg, "ours", False, False, False, seed=s),
                sess, ckpt, sc_std, aux, bnd, cfg["models"]["tldnn_128"], cfg["continual"], device, logger,
            )
            res_twc = evo.run_trajectory(
                "Ours-TWC", "order_clean", _make_journal_cl_cfg(cfg, "ours", True, True, True, seed=s),
                sess, ckpt, sc_ph, aux, bnd, cfg["models"]["tldnn_128"], cfg["continual"], device, logger,
            )
            per_seed.append(
                {
                    "seed": s,
                    "s0_acc": acc0,
                    "std_osr_auroc": next(r["auroc_pct"] for r in osr_r if r["method"] == "MS-MahaProto"),
                    "phys_osr_auroc": next(r["auroc_pct"] for r in osr_r if "Physics" in r["method"]),
                    "base_s2_a_all": res_base["sessions"][-1]["a_all_pct"],
                    "twc_s2_a_all": res_twc["sessions"][-1]["a_all_pct"],
                }
            )
        ttest_osr = paired_significance_test([r["phys_osr_auroc"] for r in per_seed], [r["std_osr_auroc"] for r in per_seed])
        ttest_cl = paired_significance_test([r["twc_s2_a_all"] for r in per_seed], [r["base_s2_a_all"] for r in per_seed])
        return {
            "seeds": seeds,
            "per_seed": per_seed,
            "std_osr_auroc": summarize_scalar([r["std_osr_auroc"] for r in per_seed]),
            "phys_osr_auroc": summarize_scalar([r["phys_osr_auroc"] for r in per_seed]),
            "base_s2_a_all": summarize_scalar([r["base_s2_a_all"] for r in per_seed]),
            "twc_s2_a_all": summarize_scalar([r["twc_s2_a_all"] for r in per_seed]),
            "ttest_osr_phys_vs_std": ttest_osr,
            "ttest_cl_twc_vs_base": ttest_cl,
        }

    sig_5 = _load_or_run(b3_dir / "significance_5seed.json", resume, _run_5seed)

    # 2. Sensitivity 4-panel sweep + DOCO++ on seed=42
    bnd_42 = SplitBundle(raw_dict=raw_a, split_cfg=c10_cfg)
    ckpt_42 = b3_dir / "seeds" / f"s{base_seed}" / "s0" / "checkpoints" / "best.pt"
    if not ckpt_42.exists():
        ckpt_42, _ = _train_or_load_stage0(
            bnd_42, bnd_42.known_classes, cfg["models"]["tldnn_128"], cfg["train"], b3_dir / "s42_s0", device, resume, base_seed, logger
        )
    aux_42 = _train_or_load_aux_snr(
        bnd_42, [int(x) for x in cfg["snr_estimator"]["all_train_snrs"]],
        b3_dir / "s42_aux.pt", cfg["snr_estimator"], device, resume, logger,
    )
    _, _, sc_ph_42 = _evaluate_osr_suite(bnd_42, cfg["models"]["tldnn_128"], ckpt_42, cfg, device)
    evo = ClosedLoopEvolver()

    def _run_sens() -> dict[str, Any]:
        sess_def = build_closed_loop_mixed_sessions(bnd_42, "order_clean", 2000, 600, 0.5, base_seed)
        sens_out: dict[str, list[dict[str, Any]]] = {"exemplars": [], "snr_thr": [], "merge_cos": [], "known_ratio": []}
        for m_val in cfg["sensitivity"]["exemplars_per_class"]:
            r = evo.run_trajectory("S", "order_clean", _make_journal_cl_cfg(cfg, "ours", True, True, True, exemplars_per_class=int(m_val), seed=base_seed), sess_def, ckpt_42, sc_ph_42, aux_42, bnd_42, cfg["models"]["tldnn_128"], cfg["continual"], device, logger)
            sens_out["exemplars"].append({"val": m_val, "s2_a_all": r["sessions"][-1]["a_all_pct"], "s2_a_old": r["sessions"][-1]["a_old_pct"]})
        for thr in cfg["sensitivity"]["snr_threshold"]:
            r = evo.run_trajectory("S", "order_clean", _make_journal_cl_cfg(cfg, "ours", True, True, True, snr_threshold=float(thr), seed=base_seed), sess_def, ckpt_42, sc_ph_42, aux_42, bnd_42, cfg["models"]["tldnn_128"], cfg["continual"], device, logger)
            sens_out["snr_thr"].append({"val": thr, "s2_a_all": r["sessions"][-1]["a_all_pct"], "s1_pur": r["sessions"][1]["gating"]["unknown_purity_pct"]})
        for mc in cfg["sensitivity"]["merge_cosine_threshold"]:
            r = evo.run_trajectory("S", "order_clean", _make_journal_cl_cfg(cfg, "ours", True, True, True, merge_cosine_threshold=float(mc), seed=base_seed), sess_def, ckpt_42, sc_ph_42, aux_42, bnd_42, cfg["models"]["tldnn_128"], cfg["continual"], device, logger)
            sens_out["merge_cos"].append({"val": mc, "s2_a_all": r["sessions"][-1]["a_all_pct"], "s2_k": r["sessions"][-1]["discovered_clusters"]})
        for kr in cfg["sensitivity"]["known_ratio"]:
            s_kr = build_closed_loop_mixed_sessions(bnd_42, "order_clean", 2000, 600, float(kr), base_seed)
            r = evo.run_trajectory("S", "order_clean", _make_journal_cl_cfg(cfg, "ours", True, True, True, seed=base_seed), s_kr, ckpt_42, sc_ph_42, aux_42, bnd_42, cfg["models"]["tldnn_128"], cfg["continual"], device, logger)
            sens_out["known_ratio"].append({"val": kr, "s2_a_all": r["sessions"][-1]["a_all_pct"], "s1_a_all": r["sessions"][1]["a_all_pct"]})
        return sens_out

    sens_res = _load_or_run(b3_dir / "sensitivity.json", resume, _run_sens)

    def _run_doco_plus() -> list[dict[str, Any]]:
        m = build_backbone(copy.deepcopy(cfg["models"]["tldnn_128"]), len(bnd_42.known_classes)).to(device)
        load_checkpoint(ckpt_42, m, device=device)
        m.eval()
        k_te, u_te = bnd_42.get_open_set_test_datasets()
        va_d = bnd_42.get_closed_set_datasets()[1]
        va_f, _, _, _ = _extract_ms_with_physics(m, DataLoader(va_d, batch_size=256), device)
        thr = calibrate_threshold(sc_ph_42.score_multiscale(va_f), 0.95)
        rows = []
        for cfo in cfg["doco_plus"]["cfo_norms"]:
            for pn in cfg["doco_plus"]["phase_noise_stds"]:
                kx = apply_rf_channel_impairments(k_te.x, cfo_norm=float(cfo), phase_noise_std=float(pn), seed=base_seed)
                ux = apply_rf_channel_impairments(u_te.x, cfo_norm=float(cfo), phase_noise_std=float(pn), seed=base_seed + 1)
                kf, _, kp, ky = _extract_ms_with_physics(m, DataLoader(RMLDataset(kx, k_te.y, k_te.snrs), batch_size=256), device)
                uf, _, _, uy = _extract_ms_with_physics(m, DataLoader(RMLDataset(ux, u_te.y, u_te.snrs), batch_size=256), device)
                res_m = evaluate_osr(sc_ph_42.score_multiscale(kf), kp, ky, sc_ph_42.score_multiscale(uf), uy, {i: str(i) for i in np.unique(uy)}, thr, 0.95)
                rows.append({"cfo_norm": cfo, "phase_noise_std": pn, "known_acc_pct": round(float(res_m["known_acc"]) * 100.0, 2), "auroc_pct": round(float(res_m["auroc"]) * 100.0, 2), "oscr_pct": round(float(res_m["oscr"]) * 100.0, 2)})
        return rows

    doco_res = _load_or_run(b3_dir / "doco_plus.json", resume, _run_doco_plus)
    out = {"significance_5seed": sig_5, "sensitivity": sens_res, "doco_plus": doco_res}
    save_json(out, b3_dir / "batch3_summary.json")
    return out


def run_batch_4(cfg: dict[str, Any], out_root: Path, device: torch.device, logger: Any) -> dict[str, Any]:
    """Batch 4: Latency/memory profiler + 300 DPI PDF/SVG vector figures + Overleaf LaTeX booktabs tables."""
    b4_dir = out_root / "batch4"
    fig_dir = b4_dir / "figures"
    tex_dir = b4_dir / "latex"

    # 1. Hardware Complexity Profiling
    profiles = []
    for name, cfg_key, flen, nc in [
        ("TLDNN (128-pt)", "tldnn_128", 128, 11),
        ("IQFormer (128-pt)", "iqformer_128", 128, 11),
        ("ResNet1D (128-pt)", "resnet1d_128", 128, 11),
        ("TLDNN (1024-pt)", "tldnn_1024", 1024, 24),
        ("IQFormer (1024-pt)", "iqformer_1024", 1024, 24),
    ]:
        bb = build_backbone(copy.deepcopy(cfg["models"][cfg_key]), num_classes=nc)
        aux = AuxSNRRegressor()
        profiles.append({"model": name, **profile_system_overhead(bb, None, aux, frame_len=flen, num_classes=nc, device=device)})

    # 2. Vector Figures (Phase Continuity Zoom + Sensitivity 4-panel + Collapse Spectrum if Batch 1/3 data present)
    t = np.arange(128, dtype=np.float32)
    xa = np.stack([np.cos(0.32 * t + 0.1), np.sin(0.32 * t + 0.1)], axis=0)
    xb = np.stack([np.cos(0.32 * t + 2.2), np.sin(0.32 * t + 2.2)], axis=0)
    plot_phase_continuity_zoom(ss_iq_pair(xa, xb, 64), ss_phase_aligned_pair(xa, xb, 64), 64, fig_dir / "fig_phase_continuity")

    # Sensitivity grid from Batch 3 if available
    b3_path = out_root / "batch3" / "batch3_summary.json"
    if b3_path.exists():
        b3 = json.loads(b3_path.read_text(encoding="utf-8"))
        sr = b3.get("sensitivity", {})
        if all(sr.get(k) for k in ("exemplars", "snr_thr", "merge_cos", "known_ratio")):
            plot_sensitivity_grid(
                [
                    {"title": "(a) Replay Budget (m)", "xlabel": "Exemplars / Class", "ylabel": "Accuracy (%)", "xs": [x["val"] for x in sr["exemplars"]], "series": {"S2 A_all": [x["s2_a_all"] for x in sr["exemplars"]], "S2 A_old": [x["s2_a_old"] for x in sr["exemplars"]]}},
                    {"title": "(b) Blind SNR Threshold", "xlabel": "tau_snr (dB)", "ylabel": "Percentage (%)", "xs": [x["val"] for x in sr["snr_thr"]], "series": {"S2 A_all": [x["s2_a_all"] for x in sr["snr_thr"]], "S1 B_unk Purity": [x["s1_pur"] for x in sr["snr_thr"]]}},
                    {"title": "(c) Sub-Cluster Merge Cos", "xlabel": "merge_cosine_threshold", "ylabel": "Accuracy (%)", "xs": [x["val"] for x in sr["merge_cos"]], "series": {"S2 A_all": [x["s2_a_all"] for x in sr["merge_cos"]]}},
                    {"title": "(d) Stream Known Ratio", "xlabel": "Known Ratio", "ylabel": "Accuracy (%)", "xs": [x["val"] for x in sr["known_ratio"]], "series": {"S1 A_all": [x["s1_a_all"] for x in sr["known_ratio"]], "S2 A_all": [x["s2_a_all"] for x in sr["known_ratio"]]}},
                ],
                fig_dir / "fig_sensitivity_grid",
            )

    # 3. Export Overleaf booktabs LaTeX tables
    tables: dict[str, str] = {}
    prof_rows = [[p["model"], f"{p['params_total_k']:.1f}", f"{p['flops_per_frame_m']:.2f}", f"{p['full_online_gate_per_frame_ms']:.4f}", f"{p['replay_memory_mb']:.2f}"] for p in profiles]
    tables["table_complexity"] = render_booktabs_table(
        "Model Complexity, Online Inference Latency, and Replay Storage", "tab:complexity",
        ["Backbone", "Params (K)", "FLOPs/Frame (M)", "Gate Latency (ms)", "Replay Mem (MB)"], prof_rows,
    )
    b1_path = out_root / "batch1" / "batch1_summary.json"
    if b1_path.exists():
        b1 = json.loads(b1_path.read_text(encoding="utf-8"))
        osr_rows = [[r["method"], f"{r['known_acc_pct']:.2f}", f"{r['auroc_pct']:.2f}", f"{r['fpr_at_tpr95_pct']:.2f}", f"{r['oscr_pct']:.2f}"] for r in b1.get("osr_config_a", [])]
        tables["table_osr_config_a"] = render_booktabs_table(
            "Static Open-Set Recognition on RadioML2016.10a Config A (Openness = 11.81\\%)", "tab:osr_a",
            ["Method", "Known Acc (\\%)", "AUROC (\\%)", "FPR@TPR95 (\\%)", "OSCR (\\%)"], osr_rows,
        )
    if b3_path.exists():
        b3 = json.loads(b3_path.read_text(encoding="utf-8"))
        sig = b3.get("significance_5seed", {})
        if sig:
            sig_rows = [
                ["Static OSR AUROC (\\%)", format_mean_std(sig["std_osr_auroc"]), format_mean_std(sig["phys_osr_auroc"]), f"{sig['ttest_osr_phys_vs_std']['p_value']:.4e}"],
                ["Clean10 S2 $A_{\\text{all}}$ (\\%)", format_mean_std(sig["base_s2_a_all"]), format_mean_std(sig["twc_s2_a_all"]), f"{sig['ttest_cl_twc_vs_base']['p_value']:.4e}"],
            ]
            tables["table_significance"] = render_booktabs_table(
                "5-Seed Statistical Significance (Mean $\\pm$ Std and Two-Sided Paired t-Test)", "tab:sig",
                ["Metric", "Baseline (MS-Maha / Stage 4)", "Upgraded (Ours TWC)", "p-value"], sig_rows,
            )

    tex_files = [str(p) for p in save_latex_bundle(tables, tex_dir)]
    out = {"profiles": profiles, "latex_files": tex_files, "figure_dir": str(fig_dir)}
    save_json(out, b4_dir / "batch4_summary.json")
    logger.info("[Batch 4] Exported %d LaTeX tables and vector figures to %s", len(tex_files), b4_dir)
    return out


def main() -> None:
    args = parse_args()
    cfg = load_yaml(resolve_path(args.config))
    if args.device is not None:
        cfg["device"] = args.device
    if args.amp:
        cfg["use_amp"] = True
        cfg["train"]["use_amp"] = True
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir

    req_dev = str(cfg.get("device", "cuda"))
    device = torch.device("cpu") if (req_dev == "cuda" and not torch.cuda.is_available()) else torch.device(req_dev)
    out_root = resolve_path(cfg.get("output_dir", "outputs/journal_twc"))
    out_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("journal_twc", log_file=out_root / "journal_twc.log")
    logger.info("Journal TWC Suite | batch=%s | device=%s | amp=%s | resume=%s", args.batch, device, cfg.get("use_amp"), args.resume)

    batches = ["1", "2", "3", "4"] if args.batch == "all" else [args.batch]
    for b in batches:
        if b == "1":
            run_batch_1(cfg, out_root, device, args.resume, logger)
        elif b == "2":
            run_batch_2(cfg, out_root, device, args.resume, logger)
        elif b == "3":
            run_batch_3(cfg, out_root, device, args.resume, logger)
        elif b == "4":
            run_batch_4(cfg, out_root, device, logger)


if __name__ == "__main__":
    main()
