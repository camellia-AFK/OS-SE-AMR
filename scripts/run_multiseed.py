"""Priority 2: Leakage-free multi-seed (42, 1024, 2025) benchmark orchestration and mean +/- std aggregation."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (SRC_DIR, SCRIPTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from os_se_amr.data.rml2016 import load_rml2016_dict
from os_se_amr.data.splitter import SplitBundle
from os_se_amr.engine import ClosedLoopEvolver
from os_se_amr.engine.trainer import ClosedSetTrainer
from os_se_amr.eval import (
    aggregate_multiseed_results,
    evaluate_osr,
    format_mean_std,
)
from os_se_amr.models import build_backbone
from os_se_amr.osr import (
    BaseOSRScorer,
    MLSScorer,
    MahalanobisPrototypeScorer,
    SoftmaxScorer,
    calibrate_threshold,
)
from os_se_amr.utils import (
    load_checkpoint,
    load_stage_config,
    load_yaml,
    resolve_path,
    save_json,
    save_yaml,
    seed_everything,
    setup_logger,
)
from run_stage1 import extract_multiscale_all, extract_predictions_and_scores
from run_stage4 import (
    _make_cl_cfg,
    run_clean10_closed_loop_section,
    run_snr_estimator_section,
    run_table2_closed_loop_section,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Priority 2: Multi-Seed Statistical Evaluation")
    parser.add_argument("--seeds", type=str, default="42,1024,2025", help="Comma-separated random seeds")
    parser.add_argument("--output-dir", type=str, default="outputs/multiseed", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Target compute device (cuda or cpu)")
    parser.add_argument("--resume", action="store_true", help="Reuse completed seed artifacts when available")
    return parser.parse_args()


def _try_collect_existing_seed42(logger: Any) -> dict[str, Any] | None:
    """Reuse already-verified Priority 1 outputs for seed=42 if all required files exist."""
    s0_path = resolve_path("outputs/stage0/tldnn_config_a_seed42/metrics.json")
    s1_methods = ("tldnn_softmax", "tldnn_mls", "tldnn_maha_preln", "tldnn_ms_maha")
    s1_paths = {m: resolve_path(f"outputs/stage1/{m}_config_a_seed42/metrics.json") for m in s1_methods}
    s4_path = resolve_path("outputs/stage4/summary_stage4.json")

    if not s0_path.exists() or not s4_path.exists() or any(not p.exists() for p in s1_paths.values()):
        return None

    s4 = json.loads(s4_path.read_text(encoding="utf-8"))
    if not s4.get("clean10_closed_loop", {}).get("runs") or not s4.get("table2_closed_loop"):
        return None

    s0 = json.loads(s0_path.read_text(encoding="utf-8"))
    s1_out: dict[str, dict[str, float]] = {}
    for m, p in s1_paths.items():
        d = json.loads(p.read_text(encoding="utf-8"))["metrics"]
        s1_out[m] = {
            "known_acc_pct": round(float(d["known_acc"]) * 100.0, 2),
            "auroc_pct": round(float(d["auroc"]) * 100.0, 2),
            "fpr_at_tpr95_pct": round(float(d["fpr_at_tpr95"]) * 100.0, 2),
            "oscr_pct": round(float(d["oscr"]) * 100.0, 2),
        }

    sync_all = s4.get("table3_ablations", {}).get("sync_ablation", [])
    sync_order_b = [r for r in sync_all if r.get("order") == "order_b"]
    if len(sync_order_b) < 2:
        return None

    logger.info("Reusing verified Priority 1 artifacts for seed=42 from outputs/stage{0,1,4}.")
    return {
        "seed": 42,
        "reused_stage4_artifacts": True,
        "stage0": {
            "tldnn_test_acc_pct": round(float(s0["test_metrics"]["overall_acc"]) * 100.0, 2),
        },
        "stage1_osr": s1_out,
        "blind_snr": s4["blind_snr"],
        "table2_closed_loop": s4["table2_closed_loop"],
        "clean10_closed_loop": s4["clean10_closed_loop"],
        "sync_ablation_order_b": sync_order_b,
    }


def _run_stage0_for_seed(
    seed: int,
    bundle: SplitBundle,
    stage0_cfg: dict[str, Any],
    out_dir: Path,
    device: torch.device,
    resume: bool,
    logger: Any,
) -> tuple[Path, float]:
    """Train or reuse Stage 0 TLDNN closed-set baseline strictly on `bundle`."""
    ckpt_path = out_dir / "checkpoints" / "best.pt"
    metrics_path = out_dir / "metrics.json"
    if resume and ckpt_path.exists() and metrics_path.exists():
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        acc = round(float(m["test_metrics"]["overall_acc"]) * 100.0, 2)
        logger.info("[Seed %d | Stage 0] Reusing checkpoint (%s) Test Acc=%.2f%%", seed, ckpt_path, acc)
        return ckpt_path, acc

    seed_everything(seed, deterministic=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_ds, val_ds, test_ds = bundle.get_closed_set_datasets()
    bs = int(stage0_cfg["train"].get("batch_size", 256))
    pin_mem = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0, pin_memory=pin_mem)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=pin_mem)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=pin_mem)

    model_cfg = copy.deepcopy(stage0_cfg["model"])
    model_cfg["name"] = "tldnn"
    model = build_backbone(model_cfg, num_classes=len(bundle.known_classes))
    trainer = ClosedSetTrainer(
        model=model,
        device=device,
        train_cfg=stage0_cfg["train"],
        output_dir=out_dir,
        logger=logger,
    )
    train_summary = trainer.fit(train_loader, val_loader, bundle.known_classes)
    val_metrics = trainer.evaluate(val_loader, bundle.known_classes)
    test_metrics = trainer.evaluate(test_loader, bundle.known_classes)
    acc_pct = round(float(test_metrics["overall_acc"]) * 100.0, 2)
    save_json(
        {
            "experiment_id": f"tldnn_config_a_seed{seed}",
            "split": bundle.summary(),
            "train_summary": train_summary,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        },
        metrics_path,
    )
    logger.info("[Seed %d | Stage 0] TLDNN Final Test Acc=%.2f%%", seed, acc_pct)
    return ckpt_path, acc_pct


def _run_stage1_for_seed(
    seed: int,
    bundle: SplitBundle,
    stage1_cfg: dict[str, Any],
    stage0_ckpt_path: Path,
    out_dir: Path,
    device: torch.device,
    logger: Any,
) -> tuple[dict[str, dict[str, float]], Path, MahalanobisPrototypeScorer]:
    """Evaluate zero-retrain static OSR methods strictly on `bundle` using `stage0_ckpt_path`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed, deterministic=True)
    model_cfg = copy.deepcopy(stage1_cfg["model"])
    model_cfg["name"] = "tldnn"
    model = build_backbone(model_cfg, num_classes=len(bundle.known_classes)).to(device)
    load_checkpoint(stage0_ckpt_path, model, device=device)

    train_ds, val_ds, _ = bundle.get_closed_set_datasets()
    known_test_ds, unknown_test_ds = bundle.get_open_set_test_datasets()
    idx_to_unknown = {idx: cls for cls, idx in bundle.all_to_idx.items() if cls in bundle.unknown_classes}

    bs = int(stage1_cfg["train"].get("batch_size", 256))
    pin_mem = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=pin_mem)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=pin_mem)
    known_loader = DataLoader(known_test_ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=pin_mem)
    unknown_loader = DataLoader(unknown_test_ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=pin_mem)

    tr_feats, _, tr_labels = extract_multiscale_all(model, train_loader, device)
    va_feats, _, _ = extract_multiscale_all(model, val_loader, device)
    k_feats, k_preds, k_labels = extract_multiscale_all(model, known_loader, device)
    u_feats, _, u_labels = extract_multiscale_all(model, unknown_loader, device)

    target_tpr = float(stage1_cfg["osr"].get("target_tpr", 0.95))
    stage1_results: dict[str, dict[str, float]] = {}

    for method, scorer in [("tldnn_softmax", SoftmaxScorer()), ("tldnn_mls", MLSScorer())]:
        assert isinstance(scorer, BaseOSRScorer)
        val_scores, _, _ = extract_predictions_and_scores(model, val_loader, device, scorer=scorer)
        thr = calibrate_threshold(val_scores, target_tpr=target_tpr)
        k_scores, _, _ = extract_predictions_and_scores(model, known_loader, device, scorer=scorer)
        u_scores, _, _ = extract_predictions_and_scores(model, unknown_loader, device, scorer=scorer)
        m = evaluate_osr(
            known_scores=k_scores,
            known_preds=k_preds,
            known_labels=k_labels,
            unknown_scores=u_scores,
            unknown_labels=u_labels,
            unknown_class_names=idx_to_unknown,
            calibrated_threshold=thr,
            target_tpr=target_tpr,
        )
        stage1_results[method] = {
            "known_acc_pct": round(float(m["known_acc"]) * 100.0, 2),
            "auroc_pct": round(float(m["auroc"]) * 100.0, 2),
            "fpr_at_tpr95_pct": round(float(m["fpr_at_tpr95"]) * 100.0, 2),
            "oscr_pct": round(float(m["oscr"]) * 100.0, 2),
        }

    # Pre-LN Mahalanobis & Multi-Scale MS-MahaProto (Ours)
    maha_cfg = stage1_cfg["osr"].get("mahalanobis", {})
    ms_layers = tuple(maha_cfg.get("layers", ["stem", "pre_ln"]))
    ms_weights = maha_cfg.get("weights", {"stem": 0.7, "pre_ln": 0.3})

    ms_scorer_out: MahalanobisPrototypeScorer | None = None
    osr_state_path = out_dir / "checkpoints" / "osr_state.pt"

    for method, layers, weights in [
        ("tldnn_maha_preln", ("pre_ln",), {"pre_ln": 1.0}),
        ("tldnn_ms_maha", ms_layers, ms_weights),
    ]:
        maha = MahalanobisPrototypeScorer(layers=layers, weights=weights)
        maha.fit_multiscale(tr_feats, tr_labels, val_feats=va_feats)
        val_scores = maha.score_multiscale(va_feats)
        thr = calibrate_threshold(val_scores, target_tpr=target_tpr)
        k_scores = maha.score_multiscale(k_feats)
        u_scores = maha.score_multiscale(u_feats)
        m = evaluate_osr(
            known_scores=k_scores,
            known_preds=k_preds,
            known_labels=k_labels,
            unknown_scores=u_scores,
            unknown_labels=u_labels,
            unknown_class_names=idx_to_unknown,
            calibrated_threshold=thr,
            target_tpr=target_tpr,
        )
        stage1_results[method] = {
            "known_acc_pct": round(float(m["known_acc"]) * 100.0, 2),
            "auroc_pct": round(float(m["auroc"]) * 100.0, 2),
            "fpr_at_tpr95_pct": round(float(m["fpr_at_tpr95"]) * 100.0, 2),
            "oscr_pct": round(float(m["oscr"]) * 100.0, 2),
        }
        if method == "tldnn_ms_maha":
            ms_scorer_out = maha
            osr_state_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"backbone": "tldnn", "scorer_state": maha.state_dict(), "calibrated_threshold": thr},
                osr_state_path,
            )

    assert ms_scorer_out is not None
    save_json(stage1_results, out_dir / "summary_stage1.json")
    for meth, vals in stage1_results.items():
        logger.info(
            "[Seed %d | Stage 1 | %-16s] KnownAcc=%.2f%% | AUROC=%.2f%% | FPR@TPR95=%.2f%% | OSCR=%.2f%%",
            seed,
            meth,
            vals["known_acc_pct"],
            vals["auroc_pct"],
            vals["fpr_at_tpr95_pct"],
            vals["oscr_pct"],
        )
    return stage1_results, osr_state_path, ms_scorer_out


def run_single_seed(
    seed: int,
    raw_dict: dict[tuple[str, int], np.ndarray],
    base_split_cfg: dict[str, Any],
    stage0_cfg: dict[str, Any],
    stage1_cfg: dict[str, Any],
    stage4_cfg: dict[str, Any],
    multiseed_root: Path,
    device: torch.device,
    resume: bool,
    logger: Any,
) -> dict[str, Any]:
    """Run or resume complete isolated Stage 0 -> 1 -> 4 pipeline for a single random seed."""
    seed_dir = multiseed_root / f"seed_{seed}"
    seed_summary_path = seed_dir / "seed_summary.json"
    if resume and seed_summary_path.exists():
        cached = json.loads(seed_summary_path.read_text(encoding="utf-8"))
        if cached.get("clean10_closed_loop", {}).get("runs") and cached.get("sync_ablation_order_b"):
            logger.info("[Seed %d] Reusing completed seed_summary.json: %s", seed, seed_summary_path)
            return cached

    if seed == 42:
        existing_42 = _try_collect_existing_seed42(logger)
        if existing_42 is not None:
            seed_dir.mkdir(parents=True, exist_ok=True)
            save_json(existing_42, seed_summary_path)
            return existing_42

    seed_dir.mkdir(parents=True, exist_ok=True)
    split_cfg_s = copy.deepcopy(base_split_cfg)
    split_cfg_s["seed"] = int(seed)
    bundle = SplitBundle(raw_dict=raw_dict, split_cfg=split_cfg_s)
    bundle.verify_disjoint()

    s0_dir = seed_dir / "stage0"
    s1_dir = seed_dir / "stage1"
    s4_dir = seed_dir / "stage4"
    s4_ckpt_dir = s4_dir / "checkpoints"
    s4_ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger.info("================ Multi-Seed Run: seed=%d ================", seed)
    s0_ckpt_path, s0_acc_pct = _run_stage0_for_seed(
        seed=seed,
        bundle=bundle,
        stage0_cfg=stage0_cfg,
        out_dir=s0_dir,
        device=device,
        resume=resume,
        logger=logger,
    )
    stage1_res, s1_osr_path, base_scorer = _run_stage1_for_seed(
        seed=seed,
        bundle=bundle,
        stage1_cfg=stage1_cfg,
        stage0_ckpt_path=s0_ckpt_path,
        out_dir=s1_dir,
        device=device,
        logger=logger,
    )

    s4_cfg = copy.deepcopy(stage4_cfg)
    s4_cfg["seed"] = int(seed)
    s4_cfg["split"] = split_cfg_s
    s4_cfg["output_dir"] = str(s4_dir)
    s4_cfg["stage0_checkpoint"] = str(s0_ckpt_path)
    s4_cfg["stage1_osr_state"] = str(s1_osr_path)
    save_yaml(s4_cfg, s4_dir / "config.yaml")

    _, aux_snr, snr_res = run_snr_estimator_section(
        cfg=s4_cfg,
        bundle=bundle,
        device=device,
        ckpt_dir=s4_ckpt_dir,
        reuse_trained=resume,
        logger=logger,
    )
    table2_runs, cached_sessions = run_table2_closed_loop_section(
        cfg=s4_cfg,
        bundle=bundle,
        base_ckpt_path=s0_ckpt_path,
        base_scorer=base_scorer,
        aux_snr=aux_snr,
        device=device,
        ckpt_dir=s4_ckpt_dir,
        logger=logger,
    )
    clean10_res = run_clean10_closed_loop_section(
        cfg=s4_cfg,
        base_bundle=bundle,
        base_ckpt_path=s0_ckpt_path,
        base_scorer=base_scorer,
        aux_snr=aux_snr,
        device=device,
        ckpt_dir=s4_ckpt_dir,
        logger=logger,
    )

    # Order B prototype synchronization ablation (W/O Sync vs With Sync)
    cl_nosync = _make_cl_cfg(
        s4_cfg, method_key="ours", gate_mode="aux_snr", ss_mode="ss_phase_aligned", sync_prototypes=False
    )
    wo_sync = ClosedLoopEvolver().run_trajectory(
        method_label="W/O Prototype Sync",
        order_name="order_b",
        cl_cfg=cl_nosync,
        sessions=cached_sessions["order_b"],
        base_ckpt_path=s0_ckpt_path,
        base_scorer=base_scorer,
        snr_estimator=aux_snr,
        bundle=bundle,
        model_cfg=s4_cfg["model"],
        cil_cfg=s4_cfg["continual"],
        device=device,
        logger=logger,
    )
    ours_b = next(r for r in table2_runs if r["order"] == "order_b" and r["method"] == "Ours (Full Model + SS)")
    with_sync = copy.deepcopy(ours_b)
    with_sync["method"] = "With Prototype Sync (Ours)"

    seed_payload = {
        "seed": int(seed),
        "reused_stage4_artifacts": False,
        "stage0": {"tldnn_test_acc_pct": s0_acc_pct},
        "stage1_osr": stage1_res,
        "blind_snr": snr_res,
        "table2_closed_loop": table2_runs,
        "clean10_closed_loop": clean10_res,
        "sync_ablation_order_b": [wo_sync, with_sync],
    }
    save_json(seed_payload, seed_summary_path)
    return seed_payload


def build_multiseed_plot(agg: dict[str, Any], fig_path: Path) -> None:
    """Render 3-panel publication plot with +/- 1 sample std error bars/bands."""
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.2))

    # Panel 1: Stage 1 Static OSR
    s1 = agg.get("stage1_osr", {})
    methods_s1 = [
        ("tldnn_softmax", "Softmax"),
        ("tldnn_mls", "MLS"),
        ("tldnn_maha_preln", "64d-Maha"),
        ("tldnn_ms_maha", "MS-MahaProto (Ours)"),
    ]
    present = [(k, lbl) for k, lbl in methods_s1 if k in s1]
    x = np.arange(len(present))
    w = 0.35
    auroc_m = [s1[k]["auroc_pct"]["mean"] for k, _ in present]
    auroc_e = [s1[k]["auroc_pct"]["std"] for k, _ in present]
    oscr_m = [s1[k]["oscr_pct"]["mean"] for k, _ in present]
    oscr_e = [s1[k]["oscr_pct"]["std"] for k, _ in present]
    axes[0].bar(x - w / 2, auroc_m, width=w, yerr=auroc_e, capsize=4, label="AUROC (%)")
    axes[0].bar(x + w / 2, oscr_m, width=w, yerr=oscr_e, capsize=4, label="OSCR (%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([lbl for _, lbl in present], rotation=15, ha="right", fontsize=8.5)
    axes[0].set_ylim(0, 108)
    axes[0].set_ylabel("Percentage (%)")
    axes[0].set_title(f"Stage 1 Static OSR (n={agg['n_seeds']} seeds, mean +/- std)", fontsize=10)
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.4)
    axes[0].legend(fontsize=8.5)

    def _plot_cl_panel(ax: Any, m_dict: dict[str, Any], xs: list[int], title: str) -> None:
        for meth, stats in m_dict.items():
            means = np.array(
                [stats["s0_a_all_pct"]["mean"], stats["s1_a_all_pct"]["mean"], stats["s2_a_all_pct"]["mean"]]
            )
            stds = np.array(
                [stats["s0_a_all_pct"]["std"], stats["s1_a_all_pct"]["std"], stats["s2_a_all_pct"]["std"]]
            )
            lw = 2.5 if "Full" in meth else 1.7
            ls = "-" if "Ours" in meth else "--"
            line = ax.plot(xs, means, marker="o", linewidth=lw, linestyle=ls, label=meth)[0]
            ax.fill_between(xs, means - stds, means + stds, color=line.get_color(), alpha=0.16)
        ax.set_xticks(xs)
        ax.set_ylim(0, 104)
        ax.set_xlabel("Total Known + Discovered Classes (C)")
        ax.set_ylabel("All-Class Accuracy A_all (%)")
        ax.set_title(title, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=7.8)

    order_b = agg.get("table2_closed_loop", {}).get("order_b", {})
    if order_b:
        _plot_cl_panel(axes[1], order_b, [7, 9, 11], f"Table 2 Order B (7 -> 9 -> 11, n={agg['n_seeds']} seeds)")

    c10_m = agg.get("clean10_closed_loop", {}).get("methods", {})
    if c10_m:
        op = float(agg.get("clean10_closed_loop", {}).get("openness_pct", 9.25))
        _plot_cl_panel(
            axes[2], c10_m, [7, 9, 10], f"Clean10 (7 -> 9 -> 10, Open={op:.2f}%, n={agg['n_seeds']} seeds)"
        )

    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def write_multiseed_markdown(agg: dict[str, Any], out_path: Path) -> None:
    """Write IEEE-ready multi-seed mean +/- std summary Markdown report."""
    seeds_str = ", ".join(str(s) for s in agg["seeds"])
    lines = [
        "# 多种子统计稳定性汇总报告（Priority 2）",
        "",
        f"- 随机种子：`[{seeds_str}]`（$n={agg['n_seeds']}$，无偏样本标准差 `ddof={agg['std_ddof']}`）。",
        "- 数据隔离承诺：每个种子均从 `RML2016.10a` 独立划分 `Train / Val / Test`，独立从零训练 `TLDNN`、拟合 `MS-MahaProto`、并在对应 `Session 0` 已知类上独立训练/校准盲 SNR 估计器，全程零跨种子与零测试集泄露。",
        "",
        "## 一、 阶段 0 与阶段 1 静态开集识别（配置 A，`Openness = 11.81%`）",
        "",
        "| 阶段 / 方法 | `Known Acc (%)` | `AUROC (%)` | `FPR@TPR95 (%)` ↓ | `OSCR (%)` |",
        "| :--- | :---: | :---: | :---: | :---: |",
        f"| Stage 0 `TLDNN` (Closed-Set) | {format_mean_std(agg['stage0']['tldnn_test_acc_pct'])} | - | - | - |",
    ]
    s1_labels = {
        "tldnn_softmax": "Stage 1 `Softmax-Threshold`",
        "tldnn_mls": "Stage 1 `MLS (ICLR 2022)`",
        "tldnn_maha_preln": "Stage 1 `64d-MahaProto`",
        "tldnn_ms_maha": "Stage 1 `MS-MahaProto (Ours)`",
    }
    for key, label in s1_labels.items():
        if key not in agg.get("stage1_osr", {}):
            continue
        d = agg["stage1_osr"][key]
        lines.append(
            f"| {label} | {format_mean_std(d['known_acc_pct'])} | {format_mean_std(d['auroc_pct'])} | "
            f"{format_mean_std(d['fpr_at_tpr95_pct'])} | {format_mean_std(d['oscr_pct'])} |"
        )

    lines.extend(
        [
            "",
            "## 二、 盲 SNR 估计器跨调制泛化稳定性（$-10\\sim +18\\text{ dB}$）",
            "",
            "| 估计器 | 测试集 | MAE (dB) ↓ | RMSE (dB) ↓ | 4 dB 门控 F1 (%) ↑ |",
            "| :--- | :--- | :---: | :---: | :---: |",
        ]
    )
    for est, ename in [("m2m4", "M2M4 (Calibrated)"), ("aux_snr", "AuxSNRRegressor (Ours)")]:
        for ds_k, ds_lbl in [
            ("known_test", "Known Test"),
            ("valid_unseen_unknown_test", "Valid Unseen Unk (QAM64/WBFM/AM-DSB)"),
        ]:
            d = agg["blind_snr"][est][ds_k]
            lines.append(
                f"| {ename} | {ds_lbl} | {format_mean_std(d['mae_db'], 3)} | "
                f"{format_mean_std(d['rmse_db'], 3)} | {format_mean_std(d['f1_4db_pct'])} |"
            )

    lines.extend(
        [
            "",
            "## 三、 表 2 真实混流闭环进化统计（`7 -> 9 -> 11`，`mean ± std`）",
            "",
            "| 协议 | 方法 | S0 $A_{\\text{all}}$ (%) | S1 $A_{\\text{all}}$ (%) | S1 $A_{\\text{new}}$ (%) | S2 $A_{\\text{all}}$ (%) | S2 $A_{\\text{old}}$ (%) | S2 $A_{\\text{new}}$ (%) | 终态 $\\Delta A_{\\text{old}}$ (%) ↓ |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for order_name, meth_map in agg.get("table2_closed_loop", {}).items():
        for meth, d in meth_map.items():
            lines.append(
                f"| {order_name} | {meth} | {format_mean_std(d['s0_a_all_pct'])} | "
                f"{format_mean_std(d['s1_a_all_pct'])} | {format_mean_std(d['s1_a_new_pct'])} | "
                f"{format_mean_std(d['s2_a_all_pct'])} | {format_mean_std(d['s2_a_old_pct'])} | "
                f"{format_mean_std(d['s2_a_new_pct'])} | {format_mean_std(d['delta_a_old_pct'])} |"
            )

    c10 = agg.get("clean10_closed_loop", {})
    if c10.get("methods"):
        lines.extend(
            [
                "",
                f"## 四、 表 2B `RML2016.10a-Clean10` 无缺陷闭环统计（`7 -> 9 -> 10`，`Openness = {c10.get('openness_pct', 9.25):.2f}%`）",
                "",
                "| 方法 | S0 $A_{\\text{all}}$ (%) | S1 $A_{\\text{all}}$ / $A_{\\text{new}}$ (%) | S2 $A_{\\text{all}}$ (%) | S2 $A_{\\text{old}}$ (%) | S2 $A_{\\text{new}}$ (%) | 终态 $\\Delta A_{\\text{old}}$ (%) ↓ | S2 核心纯度 (%) |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        for meth, d in c10["methods"].items():
            lines.append(
                f"| {meth} | {format_mean_std(d['s0_a_all_pct'])} | "
                f"{format_mean_std(d['s1_a_all_pct'])} / {format_mean_std(d['s1_a_new_pct'])} | "
                f"{format_mean_std(d['s2_a_all_pct'])} | {format_mean_std(d['s2_a_old_pct'])} | "
                f"{format_mean_std(d['s2_a_new_pct'])} | {format_mean_std(d['delta_a_old_pct'])} | "
                f"{format_mean_std(d['s2_core_purity_pct'])} |"
            )

    if agg.get("sync_ablation_order_b"):
        lines.extend(
            [
                "",
                "## 五、 `Order B` 原型同步消融稳定性（S2 汇总）",
                "",
                "| 同步策略 | S2 Known FRR (%) ↓ | S2 $\\mathcal{B}_{\\text{unk}}$ 纯度 (%) ↑ | S2 $A_{\\text{all}}$ (%) | S2 $A_{\\text{old}}$ (%) | 终态 $\\Delta A_{\\text{old}}$ (%) ↓ |",
                "| :--- | :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        for meth, d in agg["sync_ablation_order_b"].items():
            lines.append(
                f"| {meth} | {format_mean_std(d['s2_known_frr_pct'])} | {format_mean_std(d['s2_unk_purity_pct'])} | "
                f"{format_mean_std(d['s2_a_all_pct'])} | {format_mean_std(d['s2_a_old_pct'])} | "
                f"{format_mean_std(d['s2_delta_a_old_pct'])} |"
            )

    lines.extend(
        [
            "",
            "## 六、 跨数据集扩展就绪说明",
            "",
            "- 当前工作区 `data/` 仅包含 `RML2016.10a_dict.pkl`；数据加载器 `load_rml2016_dict` 已原生支持与 `RML2016.10b` 一致的 `(mod, snr) -> (N, 2, 128)` pickle 格式，放入 `RML2016.10b.dat` 并指定对应 split YAML 即可直接迁移验证。",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    req_dev = str(args.device)
    device = torch.device("cpu") if (req_dev == "cuda" and not torch.cuda.is_available()) else torch.device(req_dev)

    out_root = resolve_path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("multiseed", log_file=out_root / "multiseed.log")
    logger.info("Priority 2 Multi-Seed Pipeline | seeds=%s | device=%s | resume=%s", seeds, device, args.resume)

    stage0_cfg = load_stage_config("configs/stage0_base.yaml")
    stage1_cfg = load_stage_config("configs/stage1_osr.yaml")
    stage4_cfg = load_stage_config("configs/stage4_closed_loop.yaml")
    base_split_cfg = load_yaml(resolve_path(stage4_cfg["split_config"]))

    needs_raw = any(s != 42 for s in seeds)
    raw_dict: dict[tuple[str, int], np.ndarray] = (
        load_rml2016_dict(base_split_cfg["dataset_path"]) if (needs_raw or _try_collect_existing_seed42(logger) is None) else {}
    )
    if not raw_dict and any(not (out_root / f"seed_{s}" / "seed_summary.json").exists() for s in seeds):
        raw_dict = load_rml2016_dict(base_split_cfg["dataset_path"])

    seed_summaries: list[dict[str, Any]] = []
    for s in seeds:
        res_s = run_single_seed(
            seed=s,
            raw_dict=raw_dict,
            base_split_cfg=base_split_cfg,
            stage0_cfg=stage0_cfg,
            stage1_cfg=stage1_cfg,
            stage4_cfg=stage4_cfg,
            multiseed_root=out_root,
            device=device,
            resume=args.resume,
            logger=logger,
        )
        seed_summaries.append(res_s)

    agg = aggregate_multiseed_results(seed_summaries)
    save_json({"seeds": seeds, "per_seed": seed_summaries, "aggregated": agg}, out_root / "summary_multiseed.json")
    write_multiseed_markdown(agg, out_root / "summary_multiseed.md")
    build_multiseed_plot(agg, out_root / "figures" / "multiseed_mean_std_summary.png")
    logger.info("Multi-seed aggregation & figure saved to %s", out_root)


if __name__ == "__main__":
    main()
