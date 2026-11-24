"""Stage 2 entry point: Decoupled Main Track (Unknown Discovery) & Side Track (Anti-Forgetting CIL)."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from os_se_amr.continual import DualLwFLearner, ReplayBuffer, sync_prototypes_and_mahalanobis
from os_se_amr.data import SplitBundle, build_arrival_sessions
from os_se_amr.discovery import HDBSCANDiscoverer, RejectionBuffer
from os_se_amr.eval import evaluate_cil_session, evaluate_clustering_diagnostics
from os_se_amr.models import BaseBackbone, build_backbone
from os_se_amr.osr import MahalanobisPrototypeScorer
from os_se_amr.utils import (
    load_checkpoint,
    load_stage_config,
    resolve_path,
    save_json,
    save_yaml,
    seed_everything,
    setup_logger,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2: Decoupled Unknown Discovery & Continual Learning")
    parser.add_argument("--config", type=str, default="configs/stage2_discovery_cil.yaml", help="Stage 2 YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override incremental epochs")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--device", type=str, default=None, help="Override device (cuda or cpu)")
    return parser.parse_args()


@torch.no_grad()
def extract_ms_numpy(
    backbone: BaseBackbone,
    x_iq: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> dict[str, np.ndarray]:
    backbone.eval()
    buffers: dict[str, list[np.ndarray]] = {}
    for start in range(0, len(x_iq), batch_size):
        xb = torch.from_numpy(x_iq[start : start + batch_size]).to(device, non_blocking=True)
        ms = backbone.extract_multiscale_features(xb)
        for k, v in ms.items():
            buffers.setdefault(k, []).append(v.cpu().numpy())
    return {k: np.concatenate(v_list, axis=0) for k, v_list in buffers.items()}


@torch.no_grad()
def predict_heads(
    backbone: BaseBackbone,
    ds: Any,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    backbone.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    preds, targets = [], []
    for x, y, _ in loader:
        logits, _ = backbone(x.to(device, non_blocking=True))
        preds.append(logits.argmax(dim=1).cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(targets, axis=0), np.concatenate(preds, axis=0)


def init_base_replay_buffer(
    backbone: BaseBackbone,
    bundle: SplitBundle,
    device: torch.device,
    exemplars_per_class: int,
    strategy: str,
    seed: int,
) -> ReplayBuffer:
    train_ds, _, _ = bundle.get_closed_set_datasets()
    ms_tr = extract_ms_numpy(backbone, train_ds.x, device=device)
    rb = ReplayBuffer(exemplars_per_class=exemplars_per_class, selection_strategy=strategy)
    for cid in range(len(bundle.known_classes)):
        mask = train_ds.y == cid
        rb.add_class_exemplars(
            class_id=cid,
            x_iq=train_ds.x[mask],
            features=ms_tr["pre_ln"][mask],
            snrs=train_ds.snrs[mask],
            seed=seed,
        )
    return rb


def run_main_track(
    base_model: BaseBackbone,
    bundle: SplitBundle,
    cfg: dict[str, Any],
    device: torch.device,
    fig_dir: Path,
    logger: Any,
) -> list[dict[str, Any]]:
    """Track 1: Evaluate HDBSCAN discovery across Order A & Order B comparing stem_preln vs penult."""
    disc_cfg = cfg["discovery"]
    seed = int(cfg.get("seed", 42))
    samples_per_cls = int(disc_cfg.get("samples_per_class", 1000))
    idx_to_name = {v: k for k, v in bundle.all_to_idx.items()}

    rows: list[dict[str, Any]] = []
    pca_panels: list[dict[str, Any]] = []

    for order_name in ("order_a", "order_b"):
        batches = build_arrival_sessions(
            bundle=bundle,
            order_name=order_name,
            samples_per_class=samples_per_cls,
            mode="decoupled",
            seed=seed,
        )
        for batch in batches:
            buf = RejectionBuffer()
            buf.add(batch.x_iq, true_labels=batch.true_labels)
            x_buf = buf.get_waveforms()
            y_buf = buf.get_true_labels()
            assert y_buf is not None

            ms_feats = extract_ms_numpy(base_model, x_buf, device=device)

            for feat_mode in ("stem_preln", "penult"):
                discoverer = HDBSCANDiscoverer(
                    min_cluster_size=int(disc_cfg.get("min_cluster_size", 50)),
                    min_samples=int(disc_cfg.get("min_samples", 20)),
                    silhouette_threshold=float(disc_cfg.get("silhouette_threshold", 0.45)),
                    core_prob_threshold=float(disc_cfg.get("core_prob_threshold", 0.50)),
                    core_top_ratio=float(disc_cfg.get("core_top_ratio", 0.85)),
                    feature_mode=feat_mode,
                )
                res = discoverer.discover(ms_feats)
                diag = evaluate_clustering_diagnostics(y_buf, res.cluster_ids)

                core_purity = 0.0
                if res.core_mask.any():
                    core_diag = evaluate_clustering_diagnostics(
                        y_buf[res.core_mask], res.cluster_ids[res.core_mask]
                    )
                    core_purity = float(core_diag["many_to_1_purity"])

                row = {
                    "order": order_name,
                    "session_id": batch.session_id,
                    "arriving_classes": batch.arriving_classes,
                    "feature_mode": feat_mode,
                    "accepted": res.accepted,
                    "reason": res.reason,
                    "k_hat": res.k_hat,
                    "k_true": len(batch.arriving_classes),
                    "noise_ratio_pct": round(res.noise_ratio * 100.0, 2),
                    "silhouette": round(res.silhouette, 4),
                    "nmi": round(diag["nmi"], 4),
                    "ari": round(diag["ari"], 4),
                    "purity_pct": round(diag["many_to_1_purity"] * 100.0, 2),
                    "strict_1to1_acc_pct": round(diag["strict_1to1_acc"] * 100.0, 2),
                    "core_count": int(res.core_mask.sum()),
                    "core_purity_pct": round(core_purity * 100.0, 2),
                }
                rows.append(row)
                logger.info(
                    "[Main Track] %s S%d (%s) | %-10s | acc=%s | K_hat=%d | noise=%.1f%% | sil=%.3f | NMI=%.3f | 1to1=%.1f%% | core=%d (purity=%.1f%%)",
                    order_name,
                    batch.session_id,
                    "+".join(batch.arriving_classes),
                    feat_mode,
                    res.accepted,
                    res.k_hat,
                    res.noise_ratio * 100.0,
                    res.silhouette,
                    diag["nmi"],
                    diag["strict_1to1_acc"] * 100.0,
                    int(res.core_mask.sum()),
                    core_purity * 100.0,
                )

                if feat_mode == "stem_preln":
                    x_clust, _ = discoverer.build_clustering_space(ms_feats)
                    pca_2d = PCA(n_components=2, random_state=seed).fit_transform(x_clust)
                    pca_panels.append(
                        {
                            "title": f"{order_name.upper()} S{batch.session_id}: {' + '.join(batch.arriving_classes)}",
                            "pca_2d": pca_2d,
                            "y_true": y_buf,
                            "cluster_ids": res.cluster_ids,
                            "sil": res.silhouette,
                            "acc": diag["strict_1to1_acc"],
                            "idx_to_name": idx_to_name,
                        }
                    )

    if pca_panels:
        fig, axes = plt.subplots(2, len(pca_panels), figsize=(4.8 * len(pca_panels), 8.2))
        for col, panel in enumerate(pca_panels):
            pts = panel["pca_2d"]
            ax_top = axes[0, col]
            for tid in sorted(np.unique(panel["y_true"]).tolist()):
                m = panel["y_true"] == tid
                ax_top.scatter(pts[m, 0], pts[m, 1], s=10, alpha=0.6, label=panel["idx_to_name"][tid])
            ax_top.set_title(f"{panel['title']}\n(Ground Truth)", fontsize=9.5)
            ax_top.legend(fontsize=8)
            ax_top.grid(True, linestyle="--", alpha=0.3)

            ax_bot = axes[1, col]
            cids = panel["cluster_ids"]
            noise_m = cids == -1
            if noise_m.any():
                ax_bot.scatter(pts[noise_m, 0], pts[noise_m, 1], s=8, alpha=0.3, color="gray", label="Noise (-1)")
            for cid in sorted([c for c in np.unique(cids).tolist() if c >= 0]):
                m = cids == cid
                ax_bot.scatter(pts[m, 0], pts[m, 1], s=10, alpha=0.65, label=f"Cluster {cid}")
            ax_bot.set_title(
                f"HDBSCAN (sil={panel['sil']:.3f}, 1-to-1={panel['acc'] * 100:.1f}%)",
                fontsize=9.5,
            )
            ax_bot.legend(fontsize=8)
            ax_bot.grid(True, linestyle="--", alpha=0.3)

        fig.tight_layout()
        fig.savefig(fig_dir / "stage2_discovery_clusters.png", dpi=150)
        plt.close(fig)

    return rows


def run_cil_trajectory(
    order_name: str,
    method_name: str,
    use_lwf_replay: bool,
    exemplars_per_class: int,
    base_ckpt_path: Path,
    base_osr_scorer: MahalanobisPrototypeScorer,
    bundle: SplitBundle,
    cfg: dict[str, Any],
    device: torch.device,
    logger: Any,
    save_final_ckpt: Path | None = None,
) -> dict[str, Any]:
    """Run decoupled unsupervised continual evolution trajectory for one arrival order & learner."""
    seed = int(cfg.get("seed", 42))
    seed_everything(seed, deterministic=True)
    disc_cfg = cfg["discovery"]
    cil_cfg = cfg["continual"]
    num_base = len(bundle.known_classes)

    model = build_backbone(cfg["model"], num_classes=num_base).to(device)
    load_checkpoint(base_ckpt_path, model, device=device)

    replay_buf = init_base_replay_buffer(
        backbone=model,
        bundle=bundle,
        device=device,
        exemplars_per_class=exemplars_per_class,
        strategy=str(cil_cfg.get("selection_strategy", "herding")),
        seed=seed,
    )

    _, val_known_ds, known_test = bundle.get_closed_set_datasets()
    val_known_loader = DataLoader(val_known_ds, batch_size=256, shuffle=False, num_workers=0)

    y_t0, y_p0 = predict_heads(model, known_test, device=device)
    s0_eval = evaluate_cil_session(
        y_true=y_t0,
        y_pred=y_p0,
        num_base_classes=num_base,
        current_true_classes=list(range(num_base)),
    )
    base_acc_0 = s0_eval["a_old"]
    prev_per_class_accs = [s0_eval["per_class_acc"]]

    sessions_metrics: list[dict[str, Any]] = [
        {
            "session": 0,
            "num_classes": num_base,
            "arriving_classes": [],
            "a_all_pct": round(s0_eval["a_all"] * 100.0, 2),
            "a_old_pct": round(s0_eval["a_old"] * 100.0, 2),
            "a_new_pct": None,
            "delta_a_old_pct": 0.0,
            "fm_cls_pct": 0.0,
        }
    ]

    batches = build_arrival_sessions(
        bundle=bundle,
        order_name=order_name,
        samples_per_class=int(disc_cfg.get("samples_per_class", 1000)),
        mode="decoupled",
        seed=seed,
    )
    max_sessions = 1 if order_name == "order_a" else len(batches)

    discoverer = HDBSCANDiscoverer(
        min_cluster_size=int(disc_cfg.get("min_cluster_size", 50)),
        min_samples=int(disc_cfg.get("min_samples", 20)),
        silhouette_threshold=float(disc_cfg.get("silhouette_threshold", 0.45)),
        core_prob_threshold=float(disc_cfg.get("core_prob_threshold", 0.50)),
        core_top_ratio=float(disc_cfg.get("core_top_ratio", 0.85)),
        feature_mode=str(disc_cfg.get("feature_mode", "stem_preln")),
    )

    learner = DualLwFLearner(
        epochs=int(cil_cfg.get("epochs", 35)),
        steps_per_epoch=int(cil_cfg.get("steps_per_epoch", 30)),
        samples_per_class_per_step=int(cil_cfg.get("samples_per_class_per_step", 24)),
        batch_size_new=int(cil_cfg.get("batch_size_new", 128)),
        batch_size_old=int(cil_cfg.get("batch_size_old", 128)),
        lr=float(cil_cfg.get("lr", 8e-4)),
        weight_decay=float(cil_cfg.get("weight_decay", 1e-4)),
        grad_clip=float(cil_cfg.get("grad_clip", 1.0)),
        beta_feat=float(cil_cfg.get("beta_feat", 0.05)) if use_lwf_replay else 0.0,
        beta_logit=float(cil_cfg.get("beta_logit", 0.5)) if use_lwf_replay else 0.0,
        distill_temp=float(cil_cfg.get("distill_temp", 2.0)),
        use_replay=use_lwf_replay,
        use_rf_aug=bool(cil_cfg.get("use_rf_aug", True)) if use_lwf_replay else False,
        bic_max_steal=float(cil_cfg.get("bic_max_steal", 0.042)) if use_lwf_replay else None,
    )

    seen_classes = list(bundle.known_classes)
    active_scorer: MahalanobisPrototypeScorer = copy.deepcopy(base_osr_scorer)
    idx_to_name = {v: k for k, v in bundle.all_to_idx.items()}

    for batch in batches[:max_sessions]:
        old_num_classes = model.num_classes
        ms_batch = extract_ms_numpy(model, batch.x_iq, device=device)
        disc = discoverer.discover(ms_batch)
        if not disc.accepted or disc.k_hat == 0:
            raise RuntimeError(
                f"Discovery rejected in {order_name} S{batch.session_id}: reason={disc.reason}, k_hat={disc.k_hat}"
            )

        x_core = batch.x_iq[disc.core_mask]
        y_core_local = disc.cluster_ids[disc.core_mask]

        evo_report = learner.evolve(
            backbone=model,
            replay_buffer=replay_buf,
            new_samples=x_core,
            new_pseudo_labels=y_core_local,
            prototypes_64d=None,
            sample_weights=None,
            val_known_loader=val_known_loader,
            num_base_classes=num_base,
            device=device,
            seed=seed + batch.session_id,
        )

        if use_lwf_replay:
            ms_core_updated = extract_ms_numpy(model, x_core, device=device)
            for cid in disc.cluster_ids_unique:
                c_mask = y_core_local == cid
                replay_buf.add_class_exemplars(
                    class_id=old_num_classes + cid,
                    x_iq=x_core[c_mask],
                    features=ms_core_updated["pre_ln"][c_mask],
                    snrs=None,  # Strict no-oracle-SNR rule for discovered novel clusters
                    seed=seed + batch.session_id,
                )

            sync_info = sync_prototypes_and_mahalanobis(
                backbone=model,
                replay_buffer=replay_buf,
                new_core_x=x_core,
                new_core_labels=y_core_local,
                prev_scorer=active_scorer,
                device=device,
            )
            active_scorer = sync_info["scorer"]
        else:
            sync_info = {"old_proto_shift_pre_ln": None, "old_proto_shift_stem": None}

        seen_classes.extend(batch.arriving_classes)
        seen_true_ids = [bundle.all_to_idx[c] for c in seen_classes]
        test_ds = bundle.get_dataset("test", seen_classes, bundle.main_snrs, bundle.all_to_idx)
        yt, yp = predict_heads(model, test_ds, device=device)

        cil_eval = evaluate_cil_session(
            y_true=yt,
            y_pred=yp,
            num_base_classes=num_base,
            current_true_classes=seen_true_ids,
            prev_per_class_accs=prev_per_class_accs,
            base_acc_0=base_acc_0,
        )
        prev_per_class_accs.append(cil_eval["per_class_acc"])

        per_cls_named = {
            idx_to_name[cid]: round(acc * 100.0, 2)
            for cid, acc in cil_eval["per_class_acc"].items()
        }

        sess_record = {
            "session": batch.session_id,
            "num_classes": len(seen_classes),
            "model_heads": model.num_classes,
            "exemplars_per_class": exemplars_per_class if use_lwf_replay else 0,
            "arriving_classes": batch.arriving_classes,
            "discovered_clusters": disc.k_hat,
            "discovery_silhouette": round(disc.silhouette, 4),
            "core_samples": int(len(x_core)),
            "a_all_pct": round(cil_eval["a_all"] * 100.0, 2),
            "a_old_pct": round(cil_eval["a_old"] * 100.0, 2),
            "a_new_pct": round((cil_eval["a_new"] or 0.0) * 100.0, 2),
            "delta_a_old_pct": round(cil_eval["delta_a_old"] * 100.0, 2),
            "fm_cls_pct": round(cil_eval["fm_cls"] * 100.0, 2),
            "per_class_acc_pct": per_cls_named,
            "head_to_true": cil_eval["head_to_true"],
            "bic_gammas": evo_report.bic_gammas,
            "old_proto_shift_pre_ln": (
                round(sync_info["old_proto_shift_pre_ln"], 4)
                if sync_info["old_proto_shift_pre_ln"] is not None
                else None
            ),
            "old_proto_shift_stem": (
                round(sync_info["old_proto_shift_stem"], 4)
                if sync_info["old_proto_shift_stem"] is not None
                else None
            ),
            "train_final_loss": evo_report.final_loss,
            "elapsed_sec": evo_report.elapsed_sec,
        }
        sessions_metrics.append(sess_record)

        logger.info(
            "[Side Track | %-7s | %-28s] S%d (C=%2d) | A_all=%.2f%% | A_old=%.2f%% | A_new=%.2f%% | dA_old=%.2f%% | FM=%.2f%% | gammas=%s",
            order_name,
            method_name,
            batch.session_id,
            len(seen_classes),
            sess_record["a_all_pct"],
            sess_record["a_old_pct"],
            sess_record["a_new_pct"],
            sess_record["delta_a_old_pct"],
            sess_record["fm_cls_pct"],
            evo_report.bic_gammas,
        )

    if save_final_ckpt is not None:
        save_final_ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "num_classes": model.num_classes,
                "scorer_state": active_scorer.state_dict(),
                "sessions": sessions_metrics,
            },
            save_final_ckpt,
        )

    return {
        "order": order_name,
        "method": method_name,
        "sessions": sessions_metrics,
    }


def plot_cil_evolution(cil_runs: list[dict[str, Any]], fig_dir: Path) -> None:
    """Plot Order B (7 -> 9 -> 11) continual trajectories and Order A (7 -> 9) bar comparison."""
    order_b_runs = [r for r in cil_runs if r["order"] == "order_b"]
    order_a_runs = [r for r in cil_runs if r["order"] == "order_a"]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0))

    ax = axes[0]
    for r in order_b_runs:
        xs = [s["num_classes"] for s in r["sessions"]]
        a_all = [s["a_all_pct"] for s in r["sessions"]]
        a_old = [s["a_old_pct"] for s in r["sessions"]]
        style = "-" if "Ours" in r["method"] else "--"
        ax.plot(xs, a_all, marker="o", linestyle=style, linewidth=2, label=f"{r['method']} - A_all")
        ax.plot(xs, a_old, marker="s", linestyle=style, linewidth=1.6, alpha=0.85, label=f"{r['method']} - A_old")
    ax.set_xticks([7, 9, 11])
    ax.set_xlabel("Total Discovered + Known Classes (C)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 104)
    ax.set_title("Order B Unsupervised CIL Trajectory (7 -> 9 -> 11)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)

    ax2 = axes[1]
    labels = ["A_all", "A_old", "A_new"]
    x_pos = np.arange(len(labels))
    n_runs = max(1, len(order_a_runs))
    width = 0.78 / float(n_runs)
    for idx, r in enumerate(order_a_runs):
        s1 = r["sessions"][-1]
        vals = [s1["a_all_pct"], s1["a_old_pct"], s1["a_new_pct"] or 0.0]
        offset = (idx - 0.5 * (n_runs - 1)) * width
        bars = ax2.bar(x_pos + offset, vals, width=width, label=r["method"])
        for b in bars:
            h = b.get_height()
            ax2.text(b.get_x() + b.get_width() / 2, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.5)
    ax2.axhline(75.0, color="gray", linestyle=":", linewidth=1.2, label="Stage 2 A_new Gate (75%)")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, 108)
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Order A Session 1 (7 -> 9: QAM64 + AM-SSB)")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax2.legend(fontsize=7.5)

    fig.tight_layout()
    fig.savefig(fig_dir / "stage2_cil_evolution.png", dpi=150)
    plt.close(fig)


def write_stage2_summary(
    main_rows: list[dict[str, Any]],
    cil_runs: list[dict[str, Any]],
    out_root: Path,
) -> None:
    save_json({"main_track": main_rows, "side_track": cil_runs}, out_root / "summary_stage2.json")

    lines = [
        "# 阶段 2 解耦复现汇总报告",
        "",
        "## 一、 主线：无监督未知发现（HDBSCAN + 轮廓系数门控）",
        "",
        "| 到达协议 | 会话 | 新增类别 | 特征空间 | 通过门控 | 簇数 $\\hat{K}/K$ | 噪点率 (%) | 轮廓系数 $S_{\\text{sil}}$ | NMI | 1-to-1 Acc (%) | 核心样本纯度 (%) |",
        "| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in main_rows:
        lines.append(
            f"| {r['order']} | S{r['session_id']} | {' + '.join(r['arriving_classes'])} | `{r['feature_mode']}` | "
            f"{'Pass' if r['accepted'] else 'Hold'} | {r['k_hat']}/{r['k_true']} | {r['noise_ratio_pct']:.1f} | "
            f"{r['silhouette']:.3f} | {r['nmi']:.3f} | {r['strict_1to1_acc_pct']:.2f} | {r['core_purity_pct']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## 二、 支线：抗遗忘增量学习（SNR 分层 Herding + RF 增广 + 双层 LwF + 逐头 BiC + 原型同步）",
            "",
            "| 协议 | 方法 | 会话 | 类别数 $C$ | $A_{\\text{all}}$ (%) | $A_{\\text{old}}$ (%) | $A_{\\text{new}}$ (%) | 基础类跌幅 $\\Delta A_{\\text{old}}$ (%) | $FM_{\\text{cls}}$ (%) |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for run in cil_runs:
        for s in run["sessions"]:
            a_new_str = f"{s['a_new_pct']:.2f}" if s["a_new_pct"] is not None else "-"
            lines.append(
                f"| {run['order']} | {run['method']} | S{s['session']} | {s['num_classes']} | "
                f"{s['a_all_pct']:.2f} | {s['a_old_pct']:.2f} | {a_new_str} | "
                f"{s['delta_a_old_pct']:.2f} | {s['fm_cls_pct']:.2f} |"
            )

    (out_root / "summary_stage2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = load_stage_config(args.config)
    if args.epochs is not None:
        cfg["continual"]["epochs"] = int(args.epochs)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
        cfg["split"]["seed"] = int(args.seed)
    if args.device is not None:
        cfg["device"] = args.device

    seed = int(cfg.get("seed", 42))
    seed_everything(seed, deterministic=True)
    req_dev = str(cfg.get("device", "cuda"))
    device = torch.device("cpu") if (req_dev == "cuda" and not torch.cuda.is_available()) else torch.device(req_dev)

    out_root = resolve_path(cfg.get("output_dir", "outputs/stage2"))
    fig_dir = out_root / "figures"
    ckpt_dir = out_root / "checkpoints"
    fig_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, out_root / "config.yaml")

    logger = setup_logger("stage2", log_file=out_root / "stage2.log")
    logger.info("Stage 2 running on device=%s | seed=%d", device, seed)

    bundle = SplitBundle.from_config(cfg["split"])
    bundle.verify_disjoint()

    base_ckpt_path = resolve_path(cfg["stage0_checkpoint"])
    base_model = build_backbone(cfg["model"], num_classes=len(bundle.known_classes)).to(device)
    load_checkpoint(base_ckpt_path, base_model, device=device)

    osr_state_path = resolve_path(cfg["stage1_osr_state"])
    osr_payload = torch.load(osr_state_path, map_location="cpu", weights_only=False)
    base_scorer = MahalanobisPrototypeScorer()
    base_scorer.load_state_dict(osr_payload["scorer_state"])

    logger.info("========== Running Stage 2 Main Track: Unknown Discovery ==========")
    main_rows = run_main_track(base_model, bundle, cfg, device, fig_dir, logger)

    logger.info("========== Running Stage 2 Side Track: Continual Learning ==========")
    m_main = int(cfg["continual"].get("exemplars_per_class", 100))
    m_ablation = int(cfg["continual"].get("ablation_exemplars_per_class", 30))

    cil_runs: list[dict[str, Any]] = []
    # Order A (7 -> 9): Naive Fine-Tune vs m=30 ablation vs m=100 Ours
    for method_name, use_lwf, m_budget, ckpt_name in [
        ("Naive Fine-Tune", False, m_main, None),
        (f"Dual-LwF + BiC (m={m_ablation}, Abl)", True, m_ablation, f"tldnn_order_a_m{m_ablation}.pt"),
        (f"Dual-LwF + BiC (m={m_main}, Ours)", True, m_main, "tldnn_order_a_ours.pt"),
    ]:
        cil_runs.append(
            run_cil_trajectory(
                order_name="order_a",
                method_name=method_name,
                use_lwf_replay=use_lwf,
                exemplars_per_class=m_budget,
                base_ckpt_path=base_ckpt_path,
                base_osr_scorer=base_scorer,
                bundle=bundle,
                cfg=cfg,
                device=device,
                logger=logger,
                save_final_ckpt=(ckpt_dir / ckpt_name) if ckpt_name else None,
            )
        )

    # Order B (7 -> 9 -> 11): Naive Fine-Tune vs m=100 Ours
    for method_name, use_lwf, m_budget, ckpt_name in [
        ("Naive Fine-Tune", False, m_main, None),
        (f"Dual-LwF + BiC (m={m_main}, Ours)", True, m_main, "tldnn_order_b_ours.pt"),
    ]:
        cil_runs.append(
            run_cil_trajectory(
                order_name="order_b",
                method_name=method_name,
                use_lwf_replay=use_lwf,
                exemplars_per_class=m_budget,
                base_ckpt_path=base_ckpt_path,
                base_osr_scorer=base_scorer,
                bundle=bundle,
                cfg=cfg,
                device=device,
                logger=logger,
                save_final_ckpt=(ckpt_dir / ckpt_name) if ckpt_name else None,
            )
        )

    plot_cil_evolution(cil_runs, fig_dir)
    write_stage2_summary(main_rows, cil_runs, out_root)
    logger.info("Stage 2 summary & figures saved to %s", out_root)


if __name__ == "__main__":
    main()
