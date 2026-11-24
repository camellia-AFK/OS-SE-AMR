"""Stage 1 entry point: static OSR baselines, boundary-constrained training, and Table 1 evaluation."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from os_se_amr.data.splitter import SplitBundle
from os_se_amr.eval.osr_metrics import evaluate_osr
from os_se_amr.models import BaseBackbone, build_backbone
from os_se_amr.osr import (
    ARPLLoss,
    ARPLScorer,
    BaseOSRScorer,
    DistancePrototypeLoss,
    DistancePrototypeScorer,
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

METHOD_LABELS = {
    "tldnn_softmax": ("TLDNN", "Softmax-Threshold"),
    "tldnn_mls": ("TLDNN", "MLS (ICLR 2022)"),
    "tldnn_arpl": ("TLDNN", "ARPL (TPAMI 2022)"),
    "iqformer_mls": ("IQFormer", "MLS (ICLR 2022)"),
    "tldnn_proto": ("TLDNN", "Euclidean-Prototype (Ablation)"),
    "tldnn_maha_preln": ("TLDNN", "64d-MahaProto (Ablation)"),
    "tldnn_ms_maha": ("TLDNN (Ours)", "MS-MahaProto"),
    "iqformer_ms_maha": ("IQFormer (Ours)", "MS-MahaProto"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1: Static OSR Baseline & Metric Solidification")
    parser.add_argument("--config", type=str, default="configs/stage1_osr.yaml", help="Path to Stage 1 YAML")
    parser.add_argument(
        "--methods",
        type=str,
        default="all",
        help="Comma-separated list of methods or 'all'",
    )
    parser.add_argument(
        "--reuse-trained",
        action="store_true",
        help="Reuse existing best.pt checkpoints for boundary-trained methods (ARPL / Euclidean-Proto)",
    )
    parser.add_argument("--split-config", type=str, default=None, help="Override split YAML path")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs for trainable OSR")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--device", type=str, default=None, help="Override device (cuda or cpu)")
    return parser.parse_args()


@torch.no_grad()
def extract_predictions_and_scores(
    model: BaseBackbone,
    loader: DataLoader,
    device: torch.device,
    scorer: BaseOSRScorer,
    osr_head: nn.Module | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference over loader returning (scores, preds, labels)."""
    model.eval()
    if osr_head is not None:
        osr_head.eval()

    scores_list: list[np.ndarray] = []
    preds_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        logits, feats = model(x)

        if osr_head is not None:
            osr_logits, _ = osr_head(feats, labels=None)
            batch_preds = osr_logits.argmax(dim=1)
        else:
            batch_preds = logits.argmax(dim=1)

        batch_scores = scorer.score(logits, feats)
        scores_list.append(batch_scores.cpu().numpy())
        preds_list.append(batch_preds.cpu().numpy())
        labels_list.append(y.numpy())

    return (
        np.concatenate(scores_list, axis=0),
        np.concatenate(preds_list, axis=0),
        np.concatenate(labels_list, axis=0),
    )


@torch.no_grad()
def extract_multiscale_all(
    model: BaseBackbone,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Extract multi-scale feature dicts, closed-set predictions, and labels over a loader."""
    model.eval()
    layer_buffers: dict[str, list[np.ndarray]] = {}
    preds_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        ms = model.extract_multiscale_features(x)
        logits = model.classifier(ms["penult"])
        for k, v in ms.items():
            layer_buffers.setdefault(k, []).append(v.cpu().numpy())
        preds_list.append(logits.argmax(dim=1).cpu().numpy())
        labels_list.append(y.numpy())

    feats_dict = {k: np.concatenate(v_list, axis=0) for k, v_list in layer_buffers.items()}
    return (
        feats_dict,
        np.concatenate(preds_list, axis=0),
        np.concatenate(labels_list, axis=0),
    )


def train_boundary_osr(
    model: BaseBackbone,
    osr_loss: nn.Module,
    scorer: BaseOSRScorer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_cfg: dict[str, Any],
    device: torch.device,
    ckpt_path: Path,
    logger: Any,
) -> dict[str, Any]:
    """Train backbone jointly with ARPL or Distance-Prototype loss."""
    model.to(device)
    osr_loss.to(device)

    epochs = int(train_cfg.get("epochs", 50))
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    patience = int(train_cfg.get("early_stopping_patience", 12))

    params = list(model.parameters()) + list(osr_loss.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs), eta_min=1e-6
    )

    best_val_acc = -1.0
    best_epoch = 0
    no_improve = 0
    history: list[dict[str, Any]] = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        osr_loss.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            feats = model.extract_features(x)
            logits, loss = osr_loss(feats, y)
            assert loss is not None
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            optimizer.step()

            bs = x.size(0)
            total_loss += float(loss.item()) * bs
            total_correct += int((logits.argmax(dim=1) == y).sum().item())
            total_samples += bs

        scheduler.step()
        train_loss = total_loss / max(1, total_samples)
        train_acc = total_correct / max(1, total_samples)

        _, val_preds, val_labels = extract_predictions_and_scores(
            model, val_loader, device, scorer=scorer, osr_head=osr_loss
        )
        val_acc = float((val_preds == val_labels).mean())
        current_lr = float(optimizer.param_groups[0]["lr"])

        history.append(
            {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": round(train_loss, 5),
                "train_acc": round(train_acc, 5),
                "val_acc": round(val_acc, 5),
            }
        )

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve = 0
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "osr_state_dict": osr_loss.state_dict(),
                    "val_acc": val_acc,
                },
                ckpt_path,
            )
        else:
            no_improve += 1

        if epoch == 1 or epoch % 5 == 0 or improved or epoch == epochs:
            logger.info(
                "Epoch %02d/%02d | lr=%.2e | train_loss=%.4f train_acc=%.4f | val_acc=%.4f%s",
                epoch,
                epochs,
                current_lr,
                train_loss,
                train_acc,
                val_acc,
                " [BEST]" if improved else "",
            )

        if no_improve >= patience:
            logger.info("Early stopping at epoch %d (best epoch=%d, val_acc=%.4f)", epoch, best_epoch, best_val_acc)
            break

    best_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    osr_loss.load_state_dict(best_ckpt["osr_state_dict"])

    return {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "elapsed_sec": round(time.time() - start_time, 2),
        "history": history,
    }


def build_stage1_plots(
    all_results: dict[str, dict[str, Any]],
    score_cache: dict[str, dict[str, Any]],
    fig_dir: Path,
) -> None:
    """Plot ROC comparison, OSCR comparison, and known/unknown score distributions."""
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. ROC Comparison
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    for method, res in all_results.items():
        backbone, algo = METHOD_LABELS.get(method, (method, ""))
        roc = res["metrics"]["curves"]["roc"]
        ax.plot(
            roc["x"],
            roc["y"],
            linewidth=2,
            label=f"{backbone} + {algo} (AUROC={res['metrics']['auroc'] * 100:.2f}%)",
        )
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate (Unknown misclassified as Known)")
    ax.set_ylabel("True Positive Rate (Known detection)")
    ax.set_title("Stage 1 Static OSR - ROC Comparison (Config A)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "roc_comparison.png", dpi=150)
    plt.close(fig)

    # 2. OSCR Comparison
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    for method, res in all_results.items():
        backbone, algo = METHOD_LABELS.get(method, (method, ""))
        oscr = res["metrics"]["curves"]["oscr"]
        ax.plot(
            oscr["x"],
            oscr["y"],
            linewidth=2,
            label=f"{backbone} + {algo} (OSCR={res['metrics']['oscr'] * 100:.2f}%)",
        )
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("Correct Classification Rate (CCR)")
    ax.set_title("Stage 1 Static OSR - OSCR Comparison (Config A)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "oscr_comparison.png", dpi=150)
    plt.close(fig)

    # 3. Score Distributions
    n_methods = len(score_cache)
    if n_methods > 0:
        cols = min(4, n_methods)
        rows = (n_methods + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 3.8 * rows), squeeze=False)
        flat_axes = axes.flatten()
        for idx, (method, sc) in enumerate(score_cache.items()):
            ax = flat_axes[idx]
            backbone, algo = METHOD_LABELS.get(method, (method, ""))
            ax.hist(sc["known"], bins=50, density=True, alpha=0.55, label="Known Test", color="#1f77b4")
            ax.hist(sc["unknown"], bins=50, density=True, alpha=0.55, label="Unknown Test", color="#d62728")
            ax.axvline(sc["threshold"], color="black", linestyle="--", linewidth=1.4, label="Val TPR95 Thr")
            ax.set_title(f"{backbone} + {algo}", fontsize=9.5)
            ax.set_xlabel("Normality Score (Higher = Known)", fontsize=8.5)
            ax.set_ylabel("Density", fontsize=8.5)
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend(fontsize=7.5)
        for j in range(n_methods, len(flat_axes)):
            flat_axes[j].axis("off")
        fig.tight_layout()
        fig.savefig(fig_dir / "score_distributions.png", dpi=150)
        plt.close(fig)


def write_summary_table(all_results: dict[str, dict[str, Any]], out_root: Path) -> None:
    """Save aggregated Table 1 metrics to JSON and Markdown."""
    rows: list[dict[str, Any]] = []
    md_lines = [
        "| 骨干网络 | 拒识算法 | Known Class Acc (%) | AUROC (%) | FPR@TPR95 (%) | OSCR (%) | Val-Cal TPR (%) | Val-Cal FPR (%) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for method, res in all_results.items():
        m = res["metrics"]
        backbone, algo = METHOD_LABELS.get(method, (method, ""))
        row = {
            "method": method,
            "backbone": backbone,
            "algorithm": algo,
            "known_acc_pct": round(m["known_acc"] * 100.0, 2),
            "auroc_pct": round(m["auroc"] * 100.0, 2),
            "fpr_at_tpr95_pct": round(m["fpr_at_tpr95"] * 100.0, 2),
            "oscr_pct": round(m["oscr"] * 100.0, 2),
            "val_calibrated_tpr_pct": round(m.get("val_calibrated_tpr", 0.0) * 100.0, 2),
            "val_calibrated_fpr_pct": round(m.get("val_calibrated_fpr", 0.0) * 100.0, 2),
            "per_unknown_class_auroc_pct": {
                k: round(v * 100.0, 2) for k, v in m.get("per_unknown_class_auroc", {}).items()
            },
        }
        rows.append(row)
        md_lines.append(
            f"| {backbone} | {algo} | {row['known_acc_pct']:.2f} | {row['auroc_pct']:.2f} | "
            f"{row['fpr_at_tpr95_pct']:.2f} | {row['oscr_pct']:.2f} | "
            f"{row['val_calibrated_tpr_pct']:.2f} | {row['val_calibrated_fpr_pct']:.2f} |"
        )

    save_json({"table1": rows}, out_root / "summary_table1.json")
    md_path = out_root / "summary_table1.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = load_stage_config(args.config)

    if args.split_config is not None:
        cfg["split_config"] = args.split_config
        cfg["split"] = load_yaml(args.split_config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = int(args.epochs)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
        cfg["split"]["seed"] = int(args.seed)
    if args.device is not None:
        cfg["device"] = args.device

    seed = int(cfg.get("seed", 42))
    req_device = str(cfg.get("device", "cuda"))
    device = torch.device("cpu") if (req_device == "cuda" and not torch.cuda.is_available()) else torch.device(req_device)

    out_root = resolve_path(cfg.get("output_dir", "outputs/stage1"))
    out_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("stage1", log_file=out_root / "stage1.log")
    logger.info("Stage 1 running on device=%s | seed=%d | reuse_trained=%s", device, seed, args.reuse_trained)

    bundle = SplitBundle.from_config(cfg["split"])
    bundle.verify_disjoint()
    split_name = bundle.name
    split_summary = bundle.summary()
    logger.info(
        "Split=%s | C_train=%d | C_target=%d | Openness=%.2f%%",
        split_name,
        split_summary["c_train"],
        split_summary["c_target"],
        split_summary["openness_pct"],
    )

    train_ds, val_ds, _ = bundle.get_closed_set_datasets()
    known_test_ds, unknown_test_ds = bundle.get_open_set_test_datasets()
    idx_to_unknown = {idx: cls for cls, idx in bundle.all_to_idx.items() if cls in bundle.unknown_classes}

    batch_size = int(cfg["train"].get("batch_size", 256))
    num_workers = int(cfg["train"].get("num_workers", 0))
    pin_mem = device.type == "cuda"

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_mem)
    train_eval_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
    known_loader = DataLoader(
        known_test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem
    )
    unknown_loader = DataLoader(
        unknown_test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem
    )

    target_tpr = float(cfg["osr"].get("target_tpr", 0.95))
    configured_methods = list(cfg.get("methods", list(METHOD_LABELS.keys())))
    if args.methods.lower() == "all":
        methods_to_run = configured_methods
    else:
        methods_to_run = [m.strip() for m in args.methods.split(",") if m.strip()]

    all_results: dict[str, dict[str, Any]] = {}
    score_cache: dict[str, dict[str, Any]] = {}
    ms_feature_cache: dict[str, dict[str, tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]]] = {}

    def get_stage0_multiscale(backbone_name: str) -> dict[str, tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]]:
        if backbone_name not in ms_feature_cache:
            model_cfg = copy.deepcopy(cfg["model"])
            model_cfg["name"] = backbone_name
            model = build_backbone(model_cfg, num_classes=len(bundle.known_classes)).to(device)
            load_checkpoint(resolve_path(cfg["stage0_checkpoints"][backbone_name]), model, device=device)
            ms_feature_cache[backbone_name] = {
                "train": extract_multiscale_all(model, train_eval_loader, device),
                "val": extract_multiscale_all(model, val_loader, device),
                "known": extract_multiscale_all(model, known_loader, device),
                "unknown": extract_multiscale_all(model, unknown_loader, device),
            }
        return ms_feature_cache[backbone_name]

    for method in methods_to_run:
        if method not in METHOD_LABELS:
            raise ValueError(f"Unsupported Stage 1 method '{method}'; choose from {list(METHOD_LABELS.keys())}")

        seed_everything(seed)
        exp_id = f"{method}_{split_name}_seed{seed}"
        exp_dir = out_root / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        save_yaml(cfg, exp_dir / "config.yaml")

        logger.info("========== Running method: %s (%s) ==========", method, exp_id)
        num_known = len(bundle.known_classes)
        model_cfg = copy.deepcopy(cfg["model"])
        train_summary: dict[str, Any] | None = None
        osr_head: nn.Module | None = None

        if method in ("tldnn_maha_preln", "tldnn_ms_maha", "iqformer_ms_maha"):
            backbone_name = "iqformer" if method == "iqformer_ms_maha" else "tldnn"
            cached = get_stage0_multiscale(backbone_name)
            tr_feats, _, tr_labels = cached["train"]
            va_feats, _, _ = cached["val"]
            k_feats, k_preds, k_labels = cached["known"]
            u_feats, _, u_labels = cached["unknown"]

            if method == "tldnn_maha_preln":
                maha_scorer = MahalanobisPrototypeScorer(layers=("pre_ln",), weights={"pre_ln": 1.0})
            else:
                maha_cfg = cfg["osr"].get("mahalanobis", {})
                layers = tuple(maha_cfg.get("layers", ["stem", "pre_ln"]))
                weights = maha_cfg.get("weights", None)
                maha_scorer = MahalanobisPrototypeScorer(layers=layers, weights=weights)

            maha_scorer.fit_multiscale(tr_feats, tr_labels, val_feats=va_feats)
            val_scores = maha_scorer.score_multiscale(va_feats)
            threshold = calibrate_threshold(val_scores, target_tpr=target_tpr)

            state_path = exp_dir / "checkpoints" / "osr_state.pt"
            state_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "backbone": backbone_name,
                    "scorer_state": maha_scorer.state_dict(),
                    "calibrated_threshold": threshold,
                },
                state_path,
            )

            k_scores = maha_scorer.score_multiscale(k_feats)
            u_scores = maha_scorer.score_multiscale(u_feats)

        else:
            if method in ("tldnn_softmax", "tldnn_mls"):
                model_cfg["name"] = "tldnn"
                model = build_backbone(model_cfg, num_classes=num_known).to(device)
                load_checkpoint(resolve_path(cfg["stage0_checkpoints"]["tldnn"]), model, device=device)
                scorer: BaseOSRScorer = SoftmaxScorer() if method == "tldnn_softmax" else MLSScorer()

            elif method == "iqformer_mls":
                model_cfg["name"] = "iqformer"
                model = build_backbone(model_cfg, num_classes=num_known).to(device)
                load_checkpoint(resolve_path(cfg["stage0_checkpoints"]["iqformer"]), model, device=device)
                scorer = MLSScorer()

            elif method == "tldnn_arpl":
                model_cfg["name"] = "tldnn"
                model = build_backbone(model_cfg, num_classes=num_known).to(device)
                arpl_cfg = cfg["osr"]["arpl"]
                osr_head = ARPLLoss(
                    num_classes=num_known,
                    feat_dim=model.feature_dim,
                    weight_pl=float(arpl_cfg.get("weight_pl", 0.1)),
                    temp=float(arpl_cfg.get("temp", 1.0)),
                ).to(device)
                scorer = ARPLScorer(osr_head)
                ckpt_file = exp_dir / "checkpoints" / "best.pt"
                if args.reuse_trained and ckpt_file.exists():
                    logger.info("Reusing existing checkpoint: %s", ckpt_file)
                    ckpt_data = torch.load(ckpt_file, map_location=device, weights_only=False)
                    model.load_state_dict(ckpt_data["model_state_dict"])
                    osr_head.load_state_dict(ckpt_data["osr_state_dict"])
                else:
                    train_summary = train_boundary_osr(
                        model=model,
                        osr_loss=osr_head,
                        scorer=scorer,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        train_cfg=cfg["train"],
                        device=device,
                        ckpt_path=ckpt_file,
                        logger=logger,
                    )

            elif method == "tldnn_proto":
                model_cfg["name"] = "tldnn"
                model = build_backbone(model_cfg, num_classes=num_known).to(device)
                proto_cfg = cfg["osr"]["distance_prototype"]
                osr_head = DistancePrototypeLoss(
                    num_classes=num_known,
                    feat_dim=model.feature_dim,
                    lambda_proto=float(proto_cfg.get("lambda_proto", 0.1)),
                    temp=float(proto_cfg.get("temp", 0.1)),
                ).to(device)
                scorer = DistancePrototypeScorer(osr_head)
                ckpt_file = exp_dir / "checkpoints" / "best.pt"
                if args.reuse_trained and ckpt_file.exists():
                    logger.info("Reusing existing checkpoint: %s", ckpt_file)
                    ckpt_data = torch.load(ckpt_file, map_location=device, weights_only=False)
                    model.load_state_dict(ckpt_data["model_state_dict"])
                    osr_head.load_state_dict(ckpt_data["osr_state_dict"])
                else:
                    train_summary = train_boundary_osr(
                        model=model,
                        osr_loss=osr_head,
                        scorer=scorer,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        train_cfg=cfg["train"],
                        device=device,
                        ckpt_path=ckpt_file,
                        logger=logger,
                    )

            val_scores, _, _ = extract_predictions_and_scores(model, val_loader, device, scorer=scorer, osr_head=osr_head)
            threshold = calibrate_threshold(val_scores, target_tpr=target_tpr)

            if osr_head is not None:
                ckpt_file = exp_dir / "checkpoints" / "best.pt"
                ckpt_data = torch.load(ckpt_file, map_location="cpu", weights_only=False)
                ckpt_data["calibrated_threshold"] = threshold
                torch.save(ckpt_data, ckpt_file)

            k_scores, k_preds, k_labels = extract_predictions_and_scores(
                model, known_loader, device, scorer=scorer, osr_head=osr_head
            )
            u_scores, _, u_labels = extract_predictions_and_scores(
                model, unknown_loader, device, scorer=scorer, osr_head=osr_head
            )

        metrics = evaluate_osr(
            known_scores=k_scores,
            known_preds=k_preds,
            known_labels=k_labels,
            unknown_scores=u_scores,
            unknown_labels=u_labels,
            unknown_class_names=idx_to_unknown,
            calibrated_threshold=threshold,
            target_tpr=target_tpr,
        )

        logger.info(
            "[%s] KnownAcc=%.2f%% | AUROC=%.2f%% | FPR@TPR95=%.2f%% | OSCR=%.2f%% | CalThr=%.4f (TPR=%.2f%%, FPR=%.2f%%)",
            method,
            metrics["known_acc"] * 100.0,
            metrics["auroc"] * 100.0,
            metrics["fpr_at_tpr95"] * 100.0,
            metrics["oscr"] * 100.0,
            threshold,
            metrics["val_calibrated_tpr"] * 100.0,
            metrics["val_calibrated_fpr"] * 100.0,
        )
        if "per_unknown_class_auroc" in metrics:
            per_str = ", ".join(f"{k}={v * 100:.2f}%" for k, v in metrics["per_unknown_class_auroc"].items())
            logger.info("  Per-Unknown AUROC: %s", per_str)

        payload = {
            "experiment_id": exp_id,
            "method": method,
            "split": split_summary,
            "train_summary": train_summary,
            "metrics": metrics,
        }
        save_json(payload, exp_dir / "metrics.json")
        all_results[method] = payload
        score_cache[method] = {"known": k_scores, "unknown": u_scores, "threshold": threshold}

    ordered_results: dict[str, dict[str, Any]] = {}
    for m in configured_methods:
        if m in all_results:
            ordered_results[m] = all_results[m]
        else:
            cand = out_root / f"{m}_{split_name}_seed{seed}" / "metrics.json"
            if cand.exists():
                ordered_results[m] = json.loads(cand.read_text(encoding="utf-8"))

    write_summary_table(ordered_results, out_root)
    build_stage1_plots(ordered_results, score_cache, out_root / "figures")
    logger.info("Stage 1 summary & figures saved to %s", out_root)


if __name__ == "__main__":
    main()
