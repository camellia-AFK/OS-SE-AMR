"""Stage 3 entry point: Cross-Domain Baselines (Config A/B OSR + CIL) & Cross-SNR Environment Drift."""

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
import torch.nn.functional as F
from torch.utils.data import DataLoader

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from os_se_amr.continual import (
    DualLwFLearner,
    NaiveFineTuneLearner,
    ReplayBuffer,
    SimpleCILLearner,
    iCaRLLearner,
    sync_prototypes_and_mahalanobis,
)
from os_se_amr.data import SplitBundle, build_arrival_sessions
from os_se_amr.discovery import HDBSCANDiscoverer
from os_se_amr.engine.trainer import ClosedSetTrainer
from os_se_amr.eval import (
    apply_locked_mapping,
    evaluate_cil_session,
    evaluate_osr,
    evaluate_snr_cil_drift,
    evaluate_snr_osr_drift,
    locked_known_hungarian_match,
)
from os_se_amr.models import BaseBackbone, build_backbone
from os_se_amr.osr import (
    ARPLLoss,
    ARPLScorer,
    BaseOSRScorer,
    MLSScorer,
    MahalanobisPrototypeScorer,
    SSDMahalanobisScorer,
    SSDProjectionHead,
    SoftmaxScorer,
    SupConLoss,
    calibrate_threshold,
    ssd_rf_augment,
)
from os_se_amr.utils import (
    load_checkpoint,
    load_yaml,
    resolve_path,
    save_json,
    save_yaml,
    seed_everything,
    setup_logger,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3: Baselines & Cross-SNR Environment Drift")
    parser.add_argument("--config", type=str, default="configs/stage3_baselines.yaml", help="Stage 3 YAML")
    parser.add_argument(
        "--sections",
        type=str,
        default="all",
        help="Comma-separated subset of {osr,cil,drift} or 'all'",
    )
    parser.add_argument(
        "--reuse-trained",
        action="store_true",
        help="Reuse existing Stage 3 checkpoints when present",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--device", type=str, default=None, help="Override device (cuda or cpu)")
    return parser.parse_args()


@torch.no_grad()
def extract_ms_with_snrs(
    backbone: BaseBackbone,
    ds: Any,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract multi-scale features, logits, preds, labels, and SNRs over a dataset."""
    backbone.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    ms_buf: dict[str, list[np.ndarray]] = {}
    logits_list: list[np.ndarray] = []
    preds_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    snrs_list: list[np.ndarray] = []

    for x, y, snr in loader:
        xd = x.to(device, non_blocking=True)
        ms = backbone.extract_multiscale_features(xd)
        logits = backbone.classifier(ms["penult"])
        for k, v in ms.items():
            ms_buf.setdefault(k, []).append(v.cpu().numpy())
        logits_list.append(logits.cpu().numpy())
        preds_list.append(logits.argmax(dim=1).cpu().numpy())
        labels_list.append(y.numpy())
        snrs_list.append(snr.numpy())

    ms_feats = {k: np.concatenate(v, axis=0) for k, v in ms_buf.items()}
    return (
        ms_feats,
        np.concatenate(logits_list, axis=0),
        np.concatenate(preds_list, axis=0),
        np.concatenate(labels_list, axis=0),
        np.concatenate(snrs_list, axis=0),
    )


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
def predict_heads_with_snrs(
    backbone: BaseBackbone,
    ds: Any,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    backbone.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    targets, preds, snrs = [], [], []
    for x, y, snr in loader:
        logits, _ = backbone(x.to(device, non_blocking=True))
        preds.append(logits.argmax(dim=1).cpu().numpy())
        targets.append(y.numpy())
        snrs.append(snr.numpy())
    return np.concatenate(targets, axis=0), np.concatenate(preds, axis=0), np.concatenate(snrs, axis=0)


@torch.no_grad()
def eval_osr_standard(
    model: BaseBackbone,
    scorer: BaseOSRScorer,
    val_loader: DataLoader,
    known_loader: DataLoader,
    unknown_loader: DataLoader,
    idx_to_unknown: dict[int, str],
    device: torch.device,
    target_tpr: float = 0.95,
    osr_head: nn.Module | None = None,
) -> dict[str, Any]:
    """Calibrate threshold on Val and evaluate a scorer + model on Test Known vs Unknown."""
    model.eval()
    if osr_head is not None:
        osr_head.eval()

    def _run_loader(loader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sc_l, pr_l, lb_l = [], [], []
        for x, y, _ in loader:
            xd = x.to(device, non_blocking=True)
            logits, feats = model(xd)
            if osr_head is not None:
                osr_logits, _ = osr_head(feats, labels=None)
                preds = osr_logits.argmax(dim=1)
            else:
                preds = logits.argmax(dim=1)
            sc = scorer.score(logits, feats)
            sc_l.append(sc.cpu().numpy())
            pr_l.append(preds.cpu().numpy())
            lb_l.append(y.numpy())
        return np.concatenate(sc_l, axis=0), np.concatenate(pr_l, axis=0), np.concatenate(lb_l, axis=0)

    val_scores, _, _ = _run_loader(val_loader)
    thr = calibrate_threshold(val_scores, target_tpr=target_tpr)
    k_scores, k_preds, k_labels = _run_loader(known_loader)
    u_scores, _, u_labels = _run_loader(unknown_loader)
    return evaluate_osr(
        known_scores=k_scores,
        known_preds=k_preds,
        known_labels=k_labels,
        unknown_scores=u_scores,
        unknown_labels=u_labels,
        unknown_class_names=idx_to_unknown,
        calibrated_threshold=thr,
        target_tpr=target_tpr,
    )


def train_or_load_ssd_amc(
    model_cfg: dict[str, Any],
    num_known: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_cfg: dict[str, Any],
    ssd_cfg: dict[str, Any],
    device: torch.device,
    ckpt_path: Path,
    seed: int,
    reuse_trained: bool,
    logger: Any,
) -> tuple[BaseBackbone, SSDMahalanobisScorer, dict[str, Any] | None]:
    """Train or load TLDNN jointly with SupCon contrastive views for SSD-AMC-Open-set."""
    seed_everything(seed, deterministic=True)
    model = build_backbone(model_cfg, num_classes=num_known).to(device)
    proj_head = SSDProjectionHead(
        in_dim=model.feature_dim,
        hidden_dim=model.feature_dim,
        proj_dim=int(ssd_cfg.get("proj_dim", 64)),
    ).to(device)
    scorer = SSDMahalanobisScorer()

    if reuse_trained and ckpt_path.exists():
        logger.info("Reusing SSD-AMC checkpoint: %s", ckpt_path)
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        scorer.load_state_dict(payload["scorer_state"])
        return model, scorer, None

    epochs = int(train_cfg.get("epochs", 50))
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    patience = int(train_cfg.get("early_stopping_patience", 12))

    lambda_supcon = float(ssd_cfg.get("lambda_supcon", 0.5))
    temp = float(ssd_cfg.get("temp", 0.1))
    noise_std = float(ssd_cfg.get("noise_std", 0.03))
    max_shift = int(ssd_cfg.get("max_shift", 4))
    supcon_fn = SupConLoss(temperature=temp).to(device)

    params = list(model.parameters()) + list(proj_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=1e-6)
    gen_dev = torch.Generator(device=device).manual_seed(int(seed))

    best_val_acc = -1.0
    best_epoch = 0
    no_improve = 0
    history: list[dict[str, Any]] = []
    t0 = time.time()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        proj_head.train()
        total_loss, total_ce, total_sc, total_correct, total_n = 0.0, 0.0, 0.0, 0, 0

        for x, y, _ in train_loader:
            xb = x.to(device, non_blocking=True)
            yb = y.to(device, non_blocking=True)

            x1 = ssd_rf_augment(xb, gen=gen_dev, noise_std=noise_std, max_shift=max_shift)
            x2 = ssd_rf_augment(xb, gen=gen_dev, noise_std=noise_std, max_shift=max_shift)

            optimizer.zero_grad(set_to_none=True)
            logits_clean, _ = model(xb)
            ce_loss = F.cross_entropy(logits_clean, yb)

            z1 = proj_head(model.extract_features(x1))
            z2 = proj_head(model.extract_features(x2))
            sc_loss = supcon_fn(torch.cat([z1, z2], dim=0), torch.cat([yb, yb], dim=0))
            loss = ce_loss + lambda_supcon * sc_loss
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            optimizer.step()

            bs = len(xb)
            total_loss += float(loss.item()) * bs
            total_ce += float(ce_loss.item()) * bs
            total_sc += float(sc_loss.item()) * bs
            total_correct += int((logits_clean.argmax(dim=1) == yb).sum().item())
            total_n += bs

        scheduler.step()
        model.eval()
        val_correct, val_n = 0, 0
        with torch.no_grad():
            for vx, vy, _ in val_loader:
                vl, _ = model(vx.to(device, non_blocking=True))
                val_correct += int((vl.argmax(dim=1) == vy.to(device, non_blocking=True)).sum().item())
                val_n += len(vx)
        val_acc = val_correct / max(1, val_n)
        improved = val_acc > best_val_acc
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(total_loss / max(1, total_n), 5),
                "train_ce": round(total_ce / max(1, total_n), 5),
                "train_supcon": round(total_sc / max(1, total_n), 5),
                "train_acc": round(total_correct / max(1, total_n), 5),
                "val_acc": round(val_acc, 5),
            }
        )
        if improved:
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "proj_state_dict": proj_head.state_dict(),
                    "val_acc": val_acc,
                },
                ckpt_path,
            )
        else:
            no_improve += 1

        if epoch == 1 or epoch % 5 == 0 or improved or epoch == epochs:
            logger.info(
                "[SSD-AMC] Epoch %02d/%02d | loss=%.4f (ce=%.4f sc=%.4f) | tr_acc=%.4f | val_acc=%.4f%s",
                epoch,
                epochs,
                history[-1]["train_loss"],
                history[-1]["train_ce"],
                history[-1]["train_supcon"],
                history[-1]["train_acc"],
                val_acc,
                " [BEST]" if improved else "",
            )
        if no_improve >= patience:
            logger.info("[SSD-AMC] Early stopping at epoch %d (best=%d, val_acc=%.4f)", epoch, best_epoch, best_val_acc)
            break

    best_payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_payload["model_state_dict"])
    model.eval()

    # Fit Ledoit-Wolf Mahalanobis scorer strictly on Train L2-normalized penult features
    tr_eval_loader = DataLoader(train_loader.dataset, batch_size=256, shuffle=False, num_workers=0)
    feats_l, labels_l = [], []
    with torch.no_grad():
        for fx, fy, _ in tr_eval_loader:
            feats_l.append(model.extract_features(fx.to(device, non_blocking=True)).cpu().numpy())
            labels_l.append(fy.numpy())
    scorer.fit_numpy(np.concatenate(feats_l, axis=0), np.concatenate(labels_l, axis=0))

    best_payload["scorer_state"] = scorer.state_dict()
    torch.save(best_payload, ckpt_path)

    return model, scorer, {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "elapsed_sec": round(time.time() - t0, 2),
        "history": history,
    }


def train_or_load_arpl(
    model_cfg: dict[str, Any],
    num_known: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_cfg: dict[str, Any],
    arpl_cfg: dict[str, Any],
    device: torch.device,
    ckpt_path: Path,
    seed: int,
    reuse_trained: bool,
    logger: Any,
) -> tuple[BaseBackbone, ARPLLoss, ARPLScorer, dict[str, Any] | None]:
    """Train or load TLDNN + ARPL on a split (used for Config B)."""
    seed_everything(seed, deterministic=True)
    model = build_backbone(model_cfg, num_classes=num_known).to(device)
    osr_head = ARPLLoss(
        num_classes=num_known,
        feat_dim=model.feature_dim,
        weight_pl=float(arpl_cfg.get("weight_pl", 0.1)),
        temp=float(arpl_cfg.get("temp", 1.0)),
    ).to(device)
    scorer = ARPLScorer(osr_head)

    if reuse_trained and ckpt_path.exists():
        logger.info("Reusing ARPL checkpoint: %s", ckpt_path)
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        osr_head.load_state_dict(payload["osr_state_dict"])
        return model, osr_head, scorer, None

    epochs = int(train_cfg.get("epochs", 50))
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    patience = int(train_cfg.get("early_stopping_patience", 12))

    params = list(model.parameters()) + list(osr_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=1e-6)

    best_val_acc = -1.0
    best_epoch = 0
    no_improve = 0
    t0 = time.time()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        osr_head.train()
        tr_loss, tr_corr, tr_n = 0.0, 0, 0
        for x, y, _ in train_loader:
            xb = x.to(device, non_blocking=True)
            yb = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            feats = model.extract_features(xb)
            lg, loss = osr_head(feats, yb)
            assert loss is not None
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            optimizer.step()
            tr_loss += float(loss.item()) * len(xb)
            tr_corr += int((lg.argmax(dim=1) == yb).sum().item())
            tr_n += len(xb)
        scheduler.step()

        model.eval()
        osr_head.eval()
        vc, vn = 0, 0
        with torch.no_grad():
            for vx, vy, _ in val_loader:
                vf = model.extract_features(vx.to(device, non_blocking=True))
                vl, _ = osr_head(vf, labels=None)
                vc += int((vl.argmax(dim=1) == vy.to(device, non_blocking=True)).sum().item())
                vn += len(vx)
        val_acc = vc / max(1, vn)
        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "osr_state_dict": osr_head.state_dict(),
                    "val_acc": val_acc,
                },
                ckpt_path,
            )
        else:
            no_improve += 1

        if epoch == 1 or epoch % 5 == 0 or improved or epoch == epochs:
            logger.info(
                "[ARPL Config B] Epoch %02d/%02d | tr_loss=%.4f tr_acc=%.4f | val_acc=%.4f%s",
                epoch,
                epochs,
                tr_loss / max(1, tr_n),
                tr_corr / max(1, tr_n),
                val_acc,
                " [BEST]" if improved else "",
            )
        if no_improve >= patience:
            break

    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    osr_head.load_state_dict(payload["osr_state_dict"])
    return model, osr_head, scorer, {"best_epoch": best_epoch, "best_val_acc": best_val_acc, "elapsed_sec": round(time.time() - t0, 2)}


def _format_osr_row(
    split_name: str,
    openness_pct: float,
    backbone: str,
    algorithm: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "split": split_name,
        "openness_pct": round(openness_pct, 2),
        "backbone": backbone,
        "algorithm": algorithm,
        "known_acc_pct": round(metrics["known_acc"] * 100.0, 2),
        "auroc_pct": round(metrics["auroc"] * 100.0, 2),
        "fpr_at_tpr95_pct": round(metrics["fpr_at_tpr95"] * 100.0, 2),
        "oscr_pct": round(metrics["oscr"] * 100.0, 2),
        "val_calibrated_tpr_pct": round(metrics.get("val_calibrated_tpr", 0.0) * 100.0, 2),
        "val_calibrated_fpr_pct": round(metrics.get("val_calibrated_fpr", 0.0) * 100.0, 2),
        "per_unknown_class_auroc_pct": {
            k: round(v * 100.0, 2) for k, v in metrics.get("per_unknown_class_auroc", {}).items()
        },
    }


def run_static_osr_section(
    cfg: dict[str, Any],
    bundle_a: SplitBundle,
    bundle_b: SplitBundle,
    device: torch.device,
    ckpt_dir: Path,
    reuse_trained: bool,
    logger: Any,
) -> dict[str, Any]:
    """Section 1: Complete Config A OSR (SSD-AMC-Open-set) & evaluate full Config B (Openness=20.94%)."""
    seed = int(cfg.get("seed", 42))
    train_cfg = cfg["train"]
    osr_cfg = cfg["osr"]
    target_tpr = float(osr_cfg.get("target_tpr", 0.95))
    batch_size = int(train_cfg.get("batch_size", 256))
    pin_mem = device.type == "cuda"

    # 1. Config A (Openness=11.81%): Load Stage 1 Table 1 + run SSD-AMC-Open-set
    config_a_rows: list[dict[str, Any]] = []
    s1_summary_path = resolve_path(cfg["stage1_summary_a"])
    if s1_summary_path.exists():
        s1_data = json.loads(s1_summary_path.read_text(encoding="utf-8"))
        for r in s1_data.get("table1", []):
            config_a_rows.append(
                {
                    "split": "config_a",
                    "openness_pct": round(bundle_a.openness * 100.0, 2),
                    "backbone": r["backbone"],
                    "algorithm": r["algorithm"],
                    "known_acc_pct": r["known_acc_pct"],
                    "auroc_pct": r["auroc_pct"],
                    "fpr_at_tpr95_pct": r["fpr_at_tpr95_pct"],
                    "oscr_pct": r["oscr_pct"],
                    "val_calibrated_tpr_pct": r.get("val_calibrated_tpr_pct", 0.0),
                    "val_calibrated_fpr_pct": r.get("val_calibrated_fpr_pct", 0.0),
                    "per_unknown_class_auroc_pct": r.get("per_unknown_class_auroc_pct", {}),
                }
            )

    tr_a, va_a, _ = bundle_a.get_closed_set_datasets()
    kn_test_a, un_test_a = bundle_a.get_open_set_test_datasets()
    idx_to_unk_a = {idx: cls for cls, idx in bundle_a.all_to_idx.items() if cls in bundle_a.unknown_classes}
    tr_loader_a = DataLoader(tr_a, batch_size=batch_size, shuffle=True, pin_memory=pin_mem)
    va_loader_a = DataLoader(va_a, batch_size=batch_size, shuffle=False, pin_memory=pin_mem)
    kn_loader_a = DataLoader(kn_test_a, batch_size=batch_size, shuffle=False, pin_memory=pin_mem)
    un_loader_a = DataLoader(un_test_a, batch_size=batch_size, shuffle=False, pin_memory=pin_mem)

    ssd_model_a, ssd_scorer_a, _ = train_or_load_ssd_amc(
        model_cfg=copy.deepcopy(cfg["model"]),
        num_known=len(bundle_a.known_classes),
        train_loader=tr_loader_a,
        val_loader=va_loader_a,
        train_cfg=train_cfg,
        ssd_cfg=osr_cfg["ssd_amc"],
        device=device,
        ckpt_path=ckpt_dir / f"tldnn_ssd_amc_config_a_seed{seed}.pt",
        seed=seed,
        reuse_trained=reuse_trained,
        logger=logger,
    )
    ssd_metrics_a = eval_osr_standard(
        ssd_model_a,
        ssd_scorer_a,
        va_loader_a,
        kn_loader_a,
        un_loader_a,
        idx_to_unk_a,
        device=device,
        target_tpr=target_tpr,
    )
    ssd_row_a = _format_osr_row(
        "config_a",
        bundle_a.openness * 100.0,
        "TLDNN",
        "SSD-AMC-Open-set (RadarConf 2026)",
        ssd_metrics_a,
    )
    # Place SSD-AMC alongside the static OSR baselines before MS-MahaProto (Ours)
    insert_pos = next((i for i, r in enumerate(config_a_rows) if "MS-MahaProto" in r["algorithm"]), len(config_a_rows))
    config_a_rows.insert(insert_pos, ssd_row_a)
    logger.info(
        "[OSR Config A | SSD-AMC-Open-set] KnownAcc=%.2f%% | AUROC=%.2f%% | FPR@TPR95=%.2f%% | OSCR=%.2f%%",
        ssd_row_a["known_acc_pct"],
        ssd_row_a["auroc_pct"],
        ssd_row_a["fpr_at_tpr95_pct"],
        ssd_row_a["oscr_pct"],
    )

    # 2. Config B (Openness=20.94%, 5 Known + 6 Unknown)
    tr_b, va_b, te_b = bundle_b.get_closed_set_datasets()
    kn_test_b, un_test_b = bundle_b.get_open_set_test_datasets()
    idx_to_unk_b = {idx: cls for cls, idx in bundle_b.all_to_idx.items() if cls in bundle_b.unknown_classes}
    num_known_b = len(bundle_b.known_classes)

    tr_loader_b = DataLoader(tr_b, batch_size=batch_size, shuffle=True, pin_memory=pin_mem)
    tr_eval_b = DataLoader(tr_b, batch_size=batch_size, shuffle=False, pin_memory=pin_mem)
    va_loader_b = DataLoader(va_b, batch_size=batch_size, shuffle=False, pin_memory=pin_mem)
    te_loader_b = DataLoader(te_b, batch_size=batch_size, shuffle=False, pin_memory=pin_mem)
    kn_loader_b = DataLoader(kn_test_b, batch_size=batch_size, shuffle=False, pin_memory=pin_mem)
    un_loader_b = DataLoader(un_test_b, batch_size=batch_size, shuffle=False, pin_memory=pin_mem)

    ce_dir_b = ckpt_dir / f"tldnn_ce_config_b_seed{seed}"
    ce_ckpt_b = ce_dir_b / "checkpoints" / "best.pt"
    seed_everything(seed, deterministic=True)
    ce_model_b = build_backbone(copy.deepcopy(cfg["model"]), num_classes=num_known_b).to(device)
    if reuse_trained and ce_ckpt_b.exists():
        logger.info("Reusing Config B CE checkpoint: %s", ce_ckpt_b)
        load_checkpoint(ce_ckpt_b, ce_model_b, device=device)
    else:
        trainer_b = ClosedSetTrainer(
            model=ce_model_b,
            device=device,
            train_cfg=train_cfg,
            output_dir=ce_dir_b,
            logger=logger,
        )
        trainer_b.fit(tr_loader_b, va_loader_b, bundle_b.known_classes)
        te_res_b = trainer_b.evaluate(te_loader_b, bundle_b.known_classes)
        logger.info("[Config B Closed-Set TLDNN] Test Acc=%.2f%%", te_res_b["overall_acc"] * 100.0)

    config_b_rows: list[dict[str, Any]] = []
    # 2a. Softmax-Threshold & MLS
    for algo_name, scorer_obj in [("Softmax-Threshold", SoftmaxScorer()), ("MLS (ICLR 2022)", MLSScorer())]:
        m = eval_osr_standard(
            ce_model_b, scorer_obj, va_loader_b, kn_loader_b, un_loader_b, idx_to_unk_b, device=device, target_tpr=target_tpr
        )
        row = _format_osr_row("config_b", bundle_b.openness * 100.0, "TLDNN", algo_name, m)
        config_b_rows.append(row)
        logger.info(
            "[OSR Config B | %-26s] KnownAcc=%.2f%% | AUROC=%.2f%% | FPR@TPR95=%.2f%% | OSCR=%.2f%%",
            algo_name,
            row["known_acc_pct"],
            row["auroc_pct"],
            row["fpr_at_tpr95_pct"],
            row["oscr_pct"],
        )

    # 2b. ARPL
    arpl_model_b, arpl_head_b, arpl_scorer_b, _ = train_or_load_arpl(
        model_cfg=copy.deepcopy(cfg["model"]),
        num_known=num_known_b,
        train_loader=tr_loader_b,
        val_loader=va_loader_b,
        train_cfg=train_cfg,
        arpl_cfg=osr_cfg["arpl"],
        device=device,
        ckpt_path=ckpt_dir / f"tldnn_arpl_config_b_seed{seed}.pt",
        seed=seed,
        reuse_trained=reuse_trained,
        logger=logger,
    )
    m_arpl_b = eval_osr_standard(
        arpl_model_b,
        arpl_scorer_b,
        va_loader_b,
        kn_loader_b,
        un_loader_b,
        idx_to_unk_b,
        device=device,
        target_tpr=target_tpr,
        osr_head=arpl_head_b,
    )
    row_arpl_b = _format_osr_row("config_b", bundle_b.openness * 100.0, "TLDNN", "ARPL (TPAMI 2022)", m_arpl_b)
    config_b_rows.append(row_arpl_b)
    logger.info(
        "[OSR Config B | %-26s] KnownAcc=%.2f%% | AUROC=%.2f%% | FPR@TPR95=%.2f%% | OSCR=%.2f%%",
        "ARPL (TPAMI 2022)",
        row_arpl_b["known_acc_pct"],
        row_arpl_b["auroc_pct"],
        row_arpl_b["fpr_at_tpr95_pct"],
        row_arpl_b["oscr_pct"],
    )

    # 2c. SSD-AMC-Open-set
    ssd_model_b, ssd_scorer_b, _ = train_or_load_ssd_amc(
        model_cfg=copy.deepcopy(cfg["model"]),
        num_known=num_known_b,
        train_loader=tr_loader_b,
        val_loader=va_loader_b,
        train_cfg=train_cfg,
        ssd_cfg=osr_cfg["ssd_amc"],
        device=device,
        ckpt_path=ckpt_dir / f"tldnn_ssd_amc_config_b_seed{seed}.pt",
        seed=seed,
        reuse_trained=reuse_trained,
        logger=logger,
    )
    m_ssd_b = eval_osr_standard(
        ssd_model_b, ssd_scorer_b, va_loader_b, kn_loader_b, un_loader_b, idx_to_unk_b, device=device, target_tpr=target_tpr
    )
    row_ssd_b = _format_osr_row(
        "config_b", bundle_b.openness * 100.0, "TLDNN", "SSD-AMC-Open-set (RadarConf 2026)", m_ssd_b
    )
    config_b_rows.append(row_ssd_b)
    logger.info(
        "[OSR Config B | %-26s] KnownAcc=%.2f%% | AUROC=%.2f%% | FPR@TPR95=%.2f%% | OSCR=%.2f%%",
        "SSD-AMC-Open-set",
        row_ssd_b["known_acc_pct"],
        row_ssd_b["auroc_pct"],
        row_ssd_b["fpr_at_tpr95_pct"],
        row_ssd_b["oscr_pct"],
    )

    # 2d. 64d-MahaProto & MS-MahaProto (Ours)
    tr_ms_b, _, _, tr_y_b, _ = extract_ms_with_snrs(ce_model_b, tr_b, device=device)
    va_ms_b, _, _, _, _ = extract_ms_with_snrs(ce_model_b, va_b, device=device)
    kn_ms_b, _, kn_p_b, kn_y_b, _ = extract_ms_with_snrs(ce_model_b, kn_test_b, device=device)
    un_ms_b, _, _, un_y_b, _ = extract_ms_with_snrs(ce_model_b, un_test_b, device=device)

    maha_cfg = osr_cfg.get("mahalanobis", {})
    ms_layers = tuple(maha_cfg.get("layers", ["stem", "pre_ln"]))
    ms_weights = dict(maha_cfg.get("weights", {"stem": 0.7, "pre_ln": 0.3}))

    for algo_label, layers, weights, backbone_label in [
        ("64d-MahaProto (Ablation)", ("pre_ln",), {"pre_ln": 1.0}, "TLDNN"),
        ("MS-MahaProto", ms_layers, ms_weights, "TLDNN (Ours)"),
    ]:
        sc = MahalanobisPrototypeScorer(layers=layers, weights=weights)
        sc.fit_multiscale(tr_ms_b, tr_y_b, val_feats=va_ms_b)
        thr = calibrate_threshold(sc.score_multiscale(va_ms_b), target_tpr=target_tpr)
        m_maha = evaluate_osr(
            known_scores=sc.score_multiscale(kn_ms_b),
            known_preds=kn_p_b,
            known_labels=kn_y_b,
            unknown_scores=sc.score_multiscale(un_ms_b),
            unknown_labels=un_y_b,
            unknown_class_names=idx_to_unk_b,
            calibrated_threshold=thr,
            target_tpr=target_tpr,
        )
        row_m = _format_osr_row("config_b", bundle_b.openness * 100.0, backbone_label, algo_label, m_maha)
        config_b_rows.append(row_m)
        logger.info(
            "[OSR Config B | %-26s] KnownAcc=%.2f%% | AUROC=%.2f%% | FPR@TPR95=%.2f%% | OSCR=%.2f%%",
            algo_label,
            row_m["known_acc_pct"],
            row_m["auroc_pct"],
            row_m["fpr_at_tpr95_pct"],
            row_m["oscr_pct"],
        )

    return {"config_a": config_a_rows, "config_b": config_b_rows}


def _make_discoverer(disc_cfg: dict[str, Any]) -> HDBSCANDiscoverer:
    return HDBSCANDiscoverer(
        min_cluster_size=int(disc_cfg.get("min_cluster_size", 50)),
        min_samples=int(disc_cfg.get("min_samples", 20)),
        silhouette_threshold=float(disc_cfg.get("silhouette_threshold", 0.45)),
        core_prob_threshold=float(disc_cfg.get("core_prob_threshold", 0.50)),
        core_top_ratio=float(disc_cfg.get("core_top_ratio", 0.85)),
        feature_mode=str(disc_cfg.get("feature_mode", "stem_preln")),
    )


def _run_single_cil_method(
    order_name: str,
    method_key: str,
    display_name: str,
    base_ckpt_path: Path,
    base_scorer: MahalanobisPrototypeScorer,
    bundle: SplitBundle,
    cfg: dict[str, Any],
    device: torch.device,
    logger: Any,
) -> dict[str, Any]:
    """Run one CIL method on Order A (7->9->11) or Order B (7->9->11)."""
    seed = int(cfg.get("seed", 42))
    seed_everything(seed, deterministic=True)
    disc_cfg = cfg["discovery"]
    cil_cfg = cfg["continual"]
    num_base = len(bundle.known_classes)
    idx_to_name = {v: k for k, v in bundle.all_to_idx.items()}

    model = build_backbone(copy.deepcopy(cfg["model"]), num_classes=num_base).to(device)
    load_checkpoint(base_ckpt_path, model, device=device)

    train_ds, val_known_ds, known_test = bundle.get_closed_set_datasets()
    val_known_loader = DataLoader(val_known_ds, batch_size=256, shuffle=False, num_workers=0)
    ms_tr = extract_ms_numpy(model, train_ds.x, device=device)

    m_budget = int(cil_cfg.get("exemplars_per_class", 100))
    strategy = str(cil_cfg.get("selection_strategy", "herding"))

    replay_buf = ReplayBuffer(exemplars_per_class=m_budget, selection_strategy=strategy)
    simplecil: SimpleCILLearner | None = None
    icarl: iCaRLLearner | None = None
    lwf_learner: DualLwFLearner | None = None

    # Base Session 0 initialization
    if method_key == "simplecil":
        sc_cfg = cil_cfg.get("simplecil", {})
        simplecil = SimpleCILLearner(feature_layer=str(sc_cfg.get("feature_layer", "penult")))
        simplecil.fit_base(model, train_ds.x, train_ds.y, device=device)
        y_t0, y_p0 = simplecil.predict_dataset(model, known_test, device=device)
    elif method_key == "icarl":
        for cid in range(num_base):
            mask = train_ds.y == cid
            replay_buf.add_class_exemplars(cid, train_ds.x[mask], features=ms_tr["pre_ln"][mask], snrs=None, seed=seed)
        ic_cfg = cil_cfg.get("icarl", {})
        icarl = iCaRLLearner(
            epochs=int(ic_cfg.get("epochs", 35)),
            batch_size=int(ic_cfg.get("batch_size", 128)),
            lr=float(ic_cfg.get("lr", 8e-4)),
            weight_decay=float(cil_cfg.get("weight_decay", 1e-4)),
            grad_clip=float(cil_cfg.get("grad_clip", 1.0)),
            lambda_kd=float(ic_cfg.get("lambda_kd", 1.0)),
            distill_temp=float(ic_cfg.get("distill_temp", 2.0)),
        )
        icarl.update_class_means(model, replay_buf, device=device)
        y_t0, y_p0 = icarl.predict_nme(model, known_test, device=device)
    elif method_key == "ours":
        for cid in range(num_base):
            mask = train_ds.y == cid
            replay_buf.add_class_exemplars(
                cid, train_ds.x[mask], features=ms_tr["pre_ln"][mask], snrs=train_ds.snrs[mask], seed=seed
            )
        lwf_learner = DualLwFLearner(
            epochs=int(cil_cfg.get("epochs", 35)),
            steps_per_epoch=int(cil_cfg.get("steps_per_epoch", 30)),
            samples_per_class_per_step=int(cil_cfg.get("samples_per_class_per_step", 24)),
            batch_size_new=int(cil_cfg.get("batch_size_new", 128)),
            batch_size_old=int(cil_cfg.get("batch_size_old", 128)),
            lr=float(cil_cfg.get("lr", 8e-4)),
            weight_decay=float(cil_cfg.get("weight_decay", 1e-4)),
            grad_clip=float(cil_cfg.get("grad_clip", 1.0)),
            beta_feat=float(cil_cfg.get("beta_feat", 0.05)),
            beta_logit=float(cil_cfg.get("beta_logit", 0.5)),
            distill_temp=float(cil_cfg.get("distill_temp", 2.0)),
            use_replay=True,
            use_rf_aug=bool(cil_cfg.get("use_rf_aug", True)),
            bic_max_steal=float(cil_cfg.get("bic_max_steal", 0.046)),
        )
        y_t0, y_p0, _ = predict_heads_with_snrs(model, known_test, device=device)
    elif method_key == "naive":
        lwf_learner = NaiveFineTuneLearner(
            epochs=int(cil_cfg.get("epochs", 35)),
            batch_size_new=int(cil_cfg.get("batch_size_new", 128)),
            lr=float(cil_cfg.get("lr", 8e-4)),
            weight_decay=float(cil_cfg.get("weight_decay", 1e-4)),
            grad_clip=float(cil_cfg.get("grad_clip", 1.0)),
        )
        y_t0, y_p0, _ = predict_heads_with_snrs(model, known_test, device=device)
    else:  # static
        y_t0, y_p0, _ = predict_heads_with_snrs(model, known_test, device=device)

    s0_eval = evaluate_cil_session(
        y_true=y_t0, y_pred=y_p0, num_base_classes=num_base, current_true_classes=list(range(num_base))
    )
    base_acc_0 = s0_eval["a_old"]
    prev_per_class_accs = [s0_eval["per_class_acc"]]

    # Static OSR threshold for Static Baseline rejection diagnostic
    va_ms_known, _, _, _, _ = extract_ms_with_snrs(model, val_known_ds, device=device)
    s0_thr = calibrate_threshold(base_scorer.score_multiscale(va_ms_known), target_tpr=float(cfg["osr"].get("target_tpr", 0.95)))

    sessions_metrics: list[dict[str, Any]] = [
        {
            "session": 0,
            "num_classes": num_base,
            "arriving_classes": [],
            "discovery_accepted": True,
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
    discoverer = _make_discoverer(disc_cfg)
    seen_classes = list(bundle.known_classes)
    active_scorer: MahalanobisPrototypeScorer = copy.deepcopy(base_scorer)

    for batch in batches:
        seen_classes.extend(batch.arriving_classes)
        seen_true_ids = [bundle.all_to_idx[c] for c in seen_classes]
        test_ds = bundle.get_dataset("test", seen_classes, bundle.main_snrs, bundle.all_to_idx)

        if method_key == "static":
            yt, yp, _ = predict_heads_with_snrs(model, test_ds, device=device)
            unk_only_ds = bundle.get_dataset("test", seen_classes[num_base:], bundle.main_snrs, bundle.all_to_idx)
            unk_ms, _, _, _, _ = extract_ms_with_snrs(model, unk_only_ds, device=device)
            unk_reject_pct = float((base_scorer.score_multiscale(unk_ms) < s0_thr).mean() * 100.0)
            cil_eval = evaluate_cil_session(
                y_true=yt,
                y_pred=yp,
                num_base_classes=num_base,
                current_true_classes=seen_true_ids,
                prev_per_class_accs=prev_per_class_accs,
                base_acc_0=base_acc_0,
                head_to_true={},
            )
            prev_per_class_accs.append(cil_eval["per_class_acc"])
            rec = {
                "session": batch.session_id,
                "num_classes": len(seen_classes),
                "model_heads": model.num_classes,
                "arriving_classes": batch.arriving_classes,
                "discovery_accepted": True,
                "discovered_clusters": 0,
                "osr_unknown_reject_pct": round(unk_reject_pct, 2),
                "a_all_pct": round(cil_eval["a_all"] * 100.0, 2),
                "a_old_pct": round(cil_eval["a_old"] * 100.0, 2),
                "a_new_pct": 0.0,
                "delta_a_old_pct": round(cil_eval["delta_a_old"] * 100.0, 2),
                "fm_cls_pct": round(cil_eval["fm_cls"] * 100.0, 2),
                "head_to_true": {},
                "elapsed_sec": 0.0,
            }
            sessions_metrics.append(rec)
            logger.info(
                "[CIL | %-7s | %-25s] S%d (C=%2d) | A_all=%.2f%% | A_old=%.2f%% | A_new=0.00%% (OSR Rej=%.2f%%)",
                order_name,
                display_name,
                batch.session_id,
                len(seen_classes),
                rec["a_all_pct"],
                rec["a_old_pct"],
                unk_reject_pct,
            )
            continue

        # Discovery step
        ms_batch = extract_ms_numpy(model, batch.x_iq, device=device)
        disc = discoverer.discover(ms_batch)
        accepted = bool(disc.accepted and disc.k_hat > 0 and disc.core_mask.any())
        elapsed = 0.0
        old_proto_shift_pre_ln = None
        old_proto_shift_stem = None

        if accepted:
            old_num_classes = model.num_classes
            x_core = batch.x_iq[disc.core_mask]
            y_core_local = disc.cluster_ids[disc.core_mask]

            if method_key == "simplecil":
                assert simplecil is not None
                evo = simplecil.evolve(model, x_core, y_core_local, device=device)
                elapsed = evo.elapsed_sec
            elif method_key == "icarl":
                assert icarl is not None
                evo = icarl.evolve(
                    model, replay_buf, x_core, y_core_local, device=device, seed=seed + batch.session_id
                )
                elapsed = evo.elapsed_sec
                ms_upd = extract_ms_numpy(model, x_core, device=device)
                for cid in disc.cluster_ids_unique:
                    c_mask = y_core_local == cid
                    replay_buf.add_class_exemplars(
                        class_id=old_num_classes + cid,
                        x_iq=x_core[c_mask],
                        features=ms_upd["pre_ln"][c_mask],
                        snrs=None,
                        seed=seed + batch.session_id,
                    )
                icarl.update_class_means(model, replay_buf, device=device)
            elif method_key == "ours":
                assert lwf_learner is not None
                evo = lwf_learner.evolve(
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
                elapsed = evo.elapsed_sec
                ms_upd = extract_ms_numpy(model, x_core, device=device)
                for cid in disc.cluster_ids_unique:
                    c_mask = y_core_local == cid
                    replay_buf.add_class_exemplars(
                        class_id=old_num_classes + cid,
                        x_iq=x_core[c_mask],
                        features=ms_upd["pre_ln"][c_mask],
                        snrs=None,
                        seed=seed + batch.session_id,
                    )
                sync_info = sync_prototypes_and_mahalanobis(
                    backbone=model,
                    replay_buffer=replay_buf,
                    new_core_x=x_core,
                    new_core_labels=y_core_local,
                    prev_scorer=active_scorer,
                    device=device,
                    num_old_classes=old_num_classes,
                )
                active_scorer = sync_info["scorer"]
                old_proto_shift_pre_ln = round(float(sync_info["old_proto_shift_pre_ln"]), 4)
                old_proto_shift_stem = round(float(sync_info["old_proto_shift_stem"]), 4)
            else:  # naive
                assert lwf_learner is not None
                evo = lwf_learner.evolve(
                    backbone=model,
                    replay_buffer=replay_buf,
                    new_samples=x_core,
                    new_pseudo_labels=y_core_local,
                    prototypes_64d=None,
                    sample_weights=None,
                    val_known_loader=None,
                    num_base_classes=num_base,
                    device=device,
                    seed=seed + batch.session_id,
                )
                elapsed = evo.elapsed_sec
        else:
            logger.warning(
                "[CIL | %-7s | %-25s] S%d discovery not accepted (%s, k_hat=%d) -> holding previous model",
                order_name,
                display_name,
                batch.session_id,
                disc.reason,
                disc.k_hat,
            )

        if method_key == "simplecil":
            assert simplecil is not None
            yt, yp = simplecil.predict_dataset(model, test_ds, device=device)
        elif method_key == "icarl":
            assert icarl is not None
            yt, yp = icarl.predict_nme(model, test_ds, device=device)
        else:
            yt, yp, _ = predict_heads_with_snrs(model, test_ds, device=device)

        cil_eval = evaluate_cil_session(
            y_true=yt,
            y_pred=yp,
            num_base_classes=num_base,
            current_true_classes=seen_true_ids,
            prev_per_class_accs=prev_per_class_accs,
            base_acc_0=base_acc_0,
        )
        prev_per_class_accs.append(cil_eval["per_class_acc"])
        per_cls_named = {idx_to_name[cid]: round(acc * 100.0, 2) for cid, acc in cil_eval["per_class_acc"].items()}

        rec = {
            "session": batch.session_id,
            "num_classes": len(seen_classes),
            "model_heads": model.num_classes,
            "arriving_classes": batch.arriving_classes,
            "discovery_accepted": accepted,
            "discovery_reason": disc.reason,
            "discovered_clusters": disc.k_hat,
            "discovery_silhouette": round(disc.silhouette, 4),
            "a_all_pct": round(cil_eval["a_all"] * 100.0, 2),
            "a_old_pct": round(cil_eval["a_old"] * 100.0, 2),
            "a_new_pct": round((cil_eval["a_new"] or 0.0) * 100.0, 2),
            "delta_a_old_pct": round(cil_eval["delta_a_old"] * 100.0, 2),
            "fm_cls_pct": round(cil_eval["fm_cls"] * 100.0, 2),
            "per_class_acc_pct": per_cls_named,
            "head_to_true": cil_eval["head_to_true"],
            "old_proto_shift_pre_ln": old_proto_shift_pre_ln,
            "old_proto_shift_stem": old_proto_shift_stem,
            "elapsed_sec": elapsed,
        }
        sessions_metrics.append(rec)
        logger.info(
            "[CIL | %-7s | %-25s] S%d (C=%2d) | acc=%s | K_hat=%d | A_all=%.2f%% | A_old=%.2f%% | A_new=%.2f%% | dA_old=%.2f%%",
            order_name,
            display_name,
            batch.session_id,
            len(seen_classes),
            accepted,
            disc.k_hat,
            rec["a_all_pct"],
            rec["a_old_pct"],
            rec["a_new_pct"],
            rec["delta_a_old_pct"],
        )

    return {"order": order_name, "method_key": method_key, "method": display_name, "sessions": sessions_metrics}


def run_cil_baselines_section(
    cfg: dict[str, Any],
    bundle_a: SplitBundle,
    device: torch.device,
    logger: Any,
) -> list[dict[str, Any]]:
    """Section 2: Unified 5-method continual baseline benchmark across Order A and Order B (7->9->11)."""
    base_ckpt_path = resolve_path(cfg["stage0_checkpoint_a"])
    osr_payload = torch.load(resolve_path(cfg["stage1_osr_state_a"]), map_location="cpu", weights_only=False)
    base_scorer = MahalanobisPrototypeScorer()
    base_scorer.load_state_dict(osr_payload["scorer_state"])

    m_budget = int(cfg["continual"].get("exemplars_per_class", 100))
    methods = [
        ("static", "Static Baseline"),
        ("naive", "Naive Fine-Tune"),
        ("simplecil", "SimpleCIL"),
        ("icarl", f"iCaRL (m={m_budget})"),
        ("ours", f"Dual-LwF + BiC (m={m_budget}, Ours)"),
    ]

    results: list[dict[str, Any]] = []
    for order_name in ("order_a", "order_b"):
        for method_key, display_name in methods:
            results.append(
                _run_single_cil_method(
                    order_name=order_name,
                    method_key=method_key,
                    display_name=display_name,
                    base_ckpt_path=base_ckpt_path,
                    base_scorer=base_scorer,
                    bundle=bundle_a,
                    cfg=cfg,
                    device=device,
                    logger=logger,
                )
            )
    return results


def run_snr_drift_section(
    cfg: dict[str, Any],
    bundle_a: SplitBundle,
    device: torch.device,
    ckpt_dir: Path,
    reuse_trained: bool,
    logger: Any,
) -> dict[str, Any]:
    """Section 3: DOCO-style Cross-SNR environment drift (train & evolve on >=10dB, eval -10..+18dB)."""
    seed = int(cfg.get("seed", 42))
    seed_everything(seed, deterministic=True)

    drift_cfg = cfg["snr_drift"]
    train_snrs = [int(s) for s in drift_cfg["train_snrs"]]
    regimes = {k: [int(s) for s in v] for k, v in drift_cfg["regimes"].items()}
    all_eval_snrs = [int(s) for s in drift_cfg["all_eval_snrs"]]
    evo_order = str(drift_cfg.get("evolution_order", "order_a"))
    samples_per_cls = int(drift_cfg.get("samples_per_class", 1000))
    target_tpr = float(cfg["osr"].get("target_tpr", 0.95))

    train_cfg = cfg["train"]
    osr_cfg = cfg["osr"]
    disc_cfg = cfg["discovery"]
    cil_cfg = cfg["continual"]
    batch_size = int(train_cfg.get("batch_size", 256))
    pin_mem = device.type == "cuda"

    num_base = len(bundle_a.known_classes)
    hi_train_ds = bundle_a.get_dataset("train", bundle_a.known_classes, train_snrs, bundle_a.known_to_idx)
    hi_val_ds = bundle_a.get_dataset("val", bundle_a.known_classes, train_snrs, bundle_a.known_to_idx)
    hi_test_ds = bundle_a.get_dataset("test", bundle_a.known_classes, train_snrs, bundle_a.known_to_idx)

    hi_train_loader = DataLoader(hi_train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin_mem)
    hi_val_loader = DataLoader(hi_val_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_mem)
    hi_test_loader = DataLoader(hi_test_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_mem)

    hi_s0_dir = ckpt_dir / f"tldnn_high_snr_s0_seed{seed}"
    hi_s0_ckpt = hi_s0_dir / "checkpoints" / "best.pt"
    s0_model = build_backbone(copy.deepcopy(cfg["model"]), num_classes=num_base).to(device)

    if reuse_trained and hi_s0_ckpt.exists():
        logger.info("Reusing High-SNR S0 checkpoint: %s", hi_s0_ckpt)
        load_checkpoint(hi_s0_ckpt, s0_model, device=device)
    else:
        trainer_hi = ClosedSetTrainer(
            model=s0_model, device=device, train_cfg=train_cfg, output_dir=hi_s0_dir, logger=logger
        )
        trainer_hi.fit(hi_train_loader, hi_val_loader, bundle_a.known_classes)
        hi_eval = trainer_hi.evaluate(hi_test_loader, bundle_a.known_classes)
        logger.info("[SNR Drift | High-SNR S0 TLDNN] High-SNR Test Acc=%.2f%%", hi_eval["overall_acc"] * 100.0)

    # Session 0 OSR Drift (MLS vs MS-MahaProto) across -10 .. +18 dB
    tr_ms0, _, _, tr_y0, _ = extract_ms_with_snrs(s0_model, hi_train_ds, device=device)
    va_ms0, va_lg0, _, _, _ = extract_ms_with_snrs(s0_model, hi_val_ds, device=device)

    eval_known_s0 = bundle_a.get_dataset("test", bundle_a.known_classes, all_eval_snrs, bundle_a.all_to_idx)
    eval_unk_s0 = bundle_a.get_dataset("test", bundle_a.unknown_classes, all_eval_snrs, bundle_a.all_to_idx)
    k_ms0, k_lg0, k_p0, k_y0, k_s0 = extract_ms_with_snrs(s0_model, eval_known_s0, device=device)
    u_ms0, u_lg0, _, u_y0, u_s0 = extract_ms_with_snrs(s0_model, eval_unk_s0, device=device)

    # S0 MLS
    mls_scorer = MLSScorer()
    va_mls_scores = mls_scorer.score(torch.from_numpy(va_lg0), torch.from_numpy(va_ms0["penult"])).numpy()
    thr_mls = calibrate_threshold(va_mls_scores, target_tpr=target_tpr)
    s0_mls_drift = evaluate_snr_osr_drift(
        known_scores=mls_scorer.score(torch.from_numpy(k_lg0), torch.from_numpy(k_ms0["penult"])).numpy(),
        known_preds=k_p0,
        known_labels=k_y0,
        known_snrs=k_s0,
        unknown_scores=mls_scorer.score(torch.from_numpy(u_lg0), torch.from_numpy(u_ms0["penult"])).numpy(),
        unknown_labels=u_y0,
        unknown_snrs=u_s0,
        calibrated_threshold=thr_mls,
        regimes=regimes,
        target_tpr=target_tpr,
    )

    # S0 MS-MahaProto
    maha_cfg = osr_cfg.get("mahalanobis", {})
    ms_layers = tuple(maha_cfg.get("layers", ["stem", "pre_ln"]))
    ms_weights = dict(maha_cfg.get("weights", {"stem": 0.7, "pre_ln": 0.3}))
    s0_maha = MahalanobisPrototypeScorer(layers=ms_layers, weights=ms_weights)
    s0_maha.fit_multiscale(tr_ms0, tr_y0, val_feats=va_ms0)
    thr_maha_s0 = calibrate_threshold(s0_maha.score_multiscale(va_ms0), target_tpr=target_tpr)
    s0_maha_drift = evaluate_snr_osr_drift(
        known_scores=s0_maha.score_multiscale(k_ms0),
        known_preds=k_p0,
        known_labels=k_y0,
        known_snrs=k_s0,
        unknown_scores=s0_maha.score_multiscale(u_ms0),
        unknown_labels=u_y0,
        unknown_snrs=u_s0,
        calibrated_threshold=thr_maha_s0,
        regimes=regimes,
        target_tpr=target_tpr,
    )

    # S0 base accuracy per regime and per SNR (used for delta_a_old in CIL drift)
    base_acc_0_by_regime: dict[str, float] = {}
    for rname, snr_l in regimes.items():
        m = np.isin(k_s0, snr_l)
        base_acc_0_by_regime[rname] = float((k_p0[m] == k_y0[m]).mean())
    per_snr_base_acc_0: dict[int, float] = {}
    for sv in all_eval_snrs:
        m = k_s0 == sv
        if m.any():
            per_snr_base_acc_0[int(sv)] = float((k_p0[m] == k_y0[m]).mean())

    # High-SNR Order A Session 1 discovery (QAM64 + AM-SSB, strictly High-SNR 10..18 dB)
    batches_hi = build_arrival_sessions(
        bundle=bundle_a,
        order_name=evo_order,
        samples_per_class=samples_per_cls,
        mode="decoupled",
        snrs=train_snrs,
        seed=seed,
    )
    batch_s1 = batches_hi[0]
    discoverer = _make_discoverer(disc_cfg)
    ms_batch_hi = extract_ms_numpy(s0_model, batch_s1.x_iq, device=device)
    disc_s1 = discoverer.discover(ms_batch_hi)
    logger.info(
        "[SNR Drift | High-SNR Discovery] S1 (%s) | accepted=%s | K_hat=%d | sil=%.3f | core=%d",
        "+".join(batch_s1.arriving_classes),
        disc_s1.accepted,
        disc_s1.k_hat,
        disc_s1.silhouette,
        int(disc_s1.core_mask.sum()),
    )
    if not disc_s1.accepted or disc_s1.k_hat == 0:
        raise RuntimeError(f"High-SNR S1 discovery rejected: {disc_s1.reason}")

    x_core_s1 = batch_s1.x_iq[disc_s1.core_mask]
    y_core_s1 = disc_s1.cluster_ids[disc_s1.core_mask]

    seen_s1_classes = list(bundle_a.known_classes) + list(batch_s1.arriving_classes)
    rem_unk_classes = [c for c in bundle_a.unknown_classes if c not in batch_s1.arriving_classes]
    seen_s1_ids = [bundle_a.all_to_idx[c] for c in seen_s1_classes]
    eval_s1_ds = bundle_a.get_dataset("test", seen_s1_classes, all_eval_snrs, bundle_a.all_to_idx)
    eval_rem_unk_ds = bundle_a.get_dataset("test", rem_unk_classes, all_eval_snrs, bundle_a.all_to_idx)

    m_budget = int(cil_cfg.get("exemplars_per_class", 100))
    cil_drift_results: dict[str, Any] = {}
    post_s1_osr_drift: dict[str, Any] = {}

    # Helper to solve High-SNR Hungarian mapping cleanly before evaluating all SNRs
    def _eval_drift_for_preds(yt: np.ndarray, yp: np.ndarray, snrs: np.ndarray) -> dict[str, Any]:
        hi_mask = np.isin(snrs, regimes["high_snr"])
        novel_true = [c for c in seen_s1_ids if c >= num_base]
        if int(yp.max()) < num_base:
            fixed_map: dict[int, int] = {}
        else:
            match = locked_known_hungarian_match(
                y_true=yt[hi_mask], y_pred=yp[hi_mask], num_known=num_base, new_true_labels=novel_true
            )
            fixed_map = match["head_to_true"]
        return evaluate_snr_cil_drift(
            y_true=yt,
            y_pred=yp,
            snrs=snrs,
            num_base_classes=num_base,
            current_true_classes=seen_s1_ids,
            fixed_head_to_true=fixed_map,
            regimes=regimes,
            base_acc_0_by_regime=base_acc_0_by_regime,
            per_snr_base_acc_0=per_snr_base_acc_0,
        )

    # 1. Static Baseline post-S1 drift
    yt_st, yp_st, snr_st = predict_heads_with_snrs(s0_model, eval_s1_ds, device=device)
    cil_drift_results["Static Baseline"] = _eval_drift_for_preds(yt_st, yp_st, snr_st)

    # 2. SimpleCIL post-S1 drift
    seed_everything(seed, deterministic=True)
    m_sc = copy.deepcopy(s0_model).to(device)
    sc_cfg = cil_cfg.get("simplecil", {})
    learner_sc = SimpleCILLearner(feature_layer=str(sc_cfg.get("feature_layer", "penult")))
    learner_sc.fit_base(m_sc, hi_train_ds.x, hi_train_ds.y, device=device)
    learner_sc.evolve(m_sc, x_core_s1, y_core_s1, device=device)
    yt_sc, yp_sc = learner_sc.predict_dataset(m_sc, eval_s1_ds, device=device)
    snr_sc = eval_s1_ds.snrs
    cil_drift_results["SimpleCIL"] = _eval_drift_for_preds(yt_sc, yp_sc, snr_sc)

    # 3. iCaRL post-S1 drift
    seed_everything(seed, deterministic=True)
    m_ic = copy.deepcopy(s0_model).to(device)
    rb_ic = ReplayBuffer(exemplars_per_class=m_budget, selection_strategy="herding")
    for cid in range(num_base):
        mask = hi_train_ds.y == cid
        rb_ic.add_class_exemplars(cid, hi_train_ds.x[mask], features=tr_ms0["pre_ln"][mask], snrs=None, seed=seed)
    ic_cfg = cil_cfg.get("icarl", {})
    learner_ic = iCaRLLearner(
        epochs=int(ic_cfg.get("epochs", 35)),
        batch_size=int(ic_cfg.get("batch_size", 128)),
        lr=float(ic_cfg.get("lr", 8e-4)),
        weight_decay=float(cil_cfg.get("weight_decay", 1e-4)),
        grad_clip=float(cil_cfg.get("grad_clip", 1.0)),
        lambda_kd=float(ic_cfg.get("lambda_kd", 1.0)),
        distill_temp=float(ic_cfg.get("distill_temp", 2.0)),
    )
    learner_ic.evolve(m_ic, rb_ic, x_core_s1, y_core_s1, device=device, seed=seed + 1)
    ms_ic_upd = extract_ms_numpy(m_ic, x_core_s1, device=device)
    for cid in disc_s1.cluster_ids_unique:
        c_mask = y_core_s1 == cid
        rb_ic.add_class_exemplars(num_base + cid, x_core_s1[c_mask], features=ms_ic_upd["pre_ln"][c_mask], snrs=None, seed=seed + 1)
    learner_ic.update_class_means(m_ic, rb_ic, device=device)
    yt_ic, yp_ic = learner_ic.predict_nme(m_ic, eval_s1_ds, device=device)
    cil_drift_results[f"iCaRL (m={m_budget})"] = _eval_drift_for_preds(yt_ic, yp_ic, eval_s1_ds.snrs)

    # 4. Dual-LwF + BiC (Ours) post-S1 drift + Synced vs Unsynced OSR drift
    seed_everything(seed, deterministic=True)
    m_ours = copy.deepcopy(s0_model).to(device)
    rb_ours = ReplayBuffer(exemplars_per_class=m_budget, selection_strategy="herding")
    for cid in range(num_base):
        mask = hi_train_ds.y == cid
        rb_ours.add_class_exemplars(
            cid, hi_train_ds.x[mask], features=tr_ms0["pre_ln"][mask], snrs=hi_train_ds.snrs[mask], seed=seed
        )
    learner_ours = DualLwFLearner(
        epochs=int(cil_cfg.get("epochs", 35)),
        steps_per_epoch=int(cil_cfg.get("steps_per_epoch", 30)),
        samples_per_class_per_step=int(cil_cfg.get("samples_per_class_per_step", 24)),
        batch_size_new=int(cil_cfg.get("batch_size_new", 128)),
        batch_size_old=int(cil_cfg.get("batch_size_old", 128)),
        lr=float(cil_cfg.get("lr", 8e-4)),
        weight_decay=float(cil_cfg.get("weight_decay", 1e-4)),
        grad_clip=float(cil_cfg.get("grad_clip", 1.0)),
        beta_feat=float(cil_cfg.get("beta_feat", 0.05)),
        beta_logit=float(cil_cfg.get("beta_logit", 0.5)),
        distill_temp=float(cil_cfg.get("distill_temp", 2.0)),
        use_replay=True,
        use_rf_aug=bool(cil_cfg.get("use_rf_aug", True)),
        bic_max_steal=float(cil_cfg.get("bic_max_steal", 0.046)),
    )
    learner_ours.evolve(
        backbone=m_ours,
        replay_buffer=rb_ours,
        new_samples=x_core_s1,
        new_pseudo_labels=y_core_s1,
        val_known_loader=hi_val_loader,
        num_base_classes=num_base,
        device=device,
        seed=seed + 1,
    )
    ms_ours_upd = extract_ms_numpy(m_ours, x_core_s1, device=device)
    for cid in disc_s1.cluster_ids_unique:
        c_mask = y_core_s1 == cid
        rb_ours.add_class_exemplars(
            num_base + cid, x_core_s1[c_mask], features=ms_ours_upd["pre_ln"][c_mask], snrs=None, seed=seed + 1
        )

    sync_res = sync_prototypes_and_mahalanobis(
        backbone=m_ours,
        replay_buffer=rb_ours,
        new_core_x=x_core_s1,
        new_core_labels=y_core_s1,
        prev_scorer=s0_maha,
        device=device,
        num_old_classes=num_base,
    )
    assert sync_res["num_synced_classes"] == m_ours.num_classes
    synced_scorer: MahalanobisPrototypeScorer = sync_res["scorer"]

    # Recalibrate threshold for Synced MS-MahaProto using High-SNR Val Known + accepted S1 Core
    hi_val_ms_post, _, _, _, _ = extract_ms_with_snrs(m_ours, hi_val_ds, device=device)
    cal_ms = {k: np.concatenate([hi_val_ms_post[k], ms_ours_upd[k]], axis=0) for k in synced_scorer.layers}
    thr_synced = calibrate_threshold(synced_scorer.score_multiscale(cal_ms), target_tpr=target_tpr)

    s1_ms, _, s1_raw_preds, s1_yt, s1_snrs = extract_ms_with_snrs(m_ours, eval_s1_ds, device=device)
    rem_ms, _, _, rem_yt, rem_snrs = extract_ms_with_snrs(m_ours, eval_rem_unk_ds, device=device)

    ours_key = f"Dual-LwF + BiC (m={m_budget}, Ours)"
    ours_cil_drift = _eval_drift_for_preds(s1_yt, s1_raw_preds, s1_snrs)
    cil_drift_results[ours_key] = ours_cil_drift

    fixed_map_ours = ours_cil_drift["fixed_head_to_true"]
    s1_mapped_preds = apply_locked_mapping(s1_raw_preds, num_known=num_base, head_to_true=fixed_map_ours)

    post_s1_osr_drift["Synced MS-MahaProto (Ours)"] = evaluate_snr_osr_drift(
        known_scores=synced_scorer.score_multiscale(s1_ms),
        known_preds=s1_mapped_preds,
        known_labels=s1_yt,
        known_snrs=s1_snrs,
        unknown_scores=synced_scorer.score_multiscale(rem_ms),
        unknown_labels=rem_yt,
        unknown_snrs=rem_snrs,
        calibrated_threshold=thr_synced,
        regimes=regimes,
        target_tpr=target_tpr,
    )
    post_s1_osr_drift["Unsynced S0 MS-MahaProto"] = evaluate_snr_osr_drift(
        known_scores=s0_maha.score_multiscale(s1_ms),
        known_preds=s1_mapped_preds,
        known_labels=s1_yt,
        known_snrs=s1_snrs,
        unknown_scores=s0_maha.score_multiscale(rem_ms),
        unknown_labels=rem_yt,
        unknown_snrs=rem_snrs,
        calibrated_threshold=thr_maha_s0,
        regimes=regimes,
        target_tpr=target_tpr,
    )

    for label, res in cil_drift_results.items():
        regs = res["regimes"]
        logger.info(
            "[SNR Drift CIL | %-28s] High=%.2f%% | Mid-Low=%.2f%% | Neg=%.2f%%",
            label,
            regs["high_snr"]["a_all_pct"],
            regs["mid_low_snr"]["a_all_pct"],
            regs["neg_snr"]["a_all_pct"],
        )

    return {
        "session0_osr_drift": {
            "MLS": s0_mls_drift,
            "MS-MahaProto": s0_maha_drift,
        },
        "high_snr_discovery": {
            "order": evo_order,
            "arriving_classes": batch_s1.arriving_classes,
            "accepted": disc_s1.accepted,
            "k_hat": disc_s1.k_hat,
            "silhouette": round(disc_s1.silhouette, 4),
            "core_samples": int(disc_s1.core_mask.sum()),
            "num_synced_classes": int(sync_res["num_synced_classes"]),
            "old_proto_shift_pre_ln": round(float(sync_res["old_proto_shift_pre_ln"]), 4),
            "old_proto_shift_stem": round(float(sync_res["old_proto_shift_stem"]), 4),
        },
        "post_s1_osr_drift": post_s1_osr_drift,
        "post_s1_cil_drift": cil_drift_results,
    }


def build_stage3_plots(
    osr_res: dict[str, Any],
    cil_res: list[dict[str, Any]],
    drift_res: dict[str, Any],
    fig_dir: Path,
) -> None:
    """Generate the 3 Stage 3 publication-quality comparison figures."""
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Static OSR Config A vs Config B
    if osr_res and osr_res.get("config_a") and osr_res.get("config_b"):
        fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.2))
        for ax, split_key, title in [
            (axes[0], "config_a", "Config A: 7 Known + 4 Unknown (Openness = 11.81%)"),
            (axes[1], "config_b", "Config B: 5 Known + 6 Unknown (Openness = 20.94%)"),
        ]:
            rows = osr_res[split_key]
            labels = [
                f"{r['algorithm'].split(' (')[0]}\n({r['backbone'].replace(' (Ours)', '*')})"
                for r in rows
            ]
            x = np.arange(len(rows))
            w = 0.26
            auroc = [r["auroc_pct"] for r in rows]
            oscr = [r["oscr_pct"] for r in rows]
            fpr95 = [r["fpr_at_tpr95_pct"] for r in rows]
            ax.bar(x - w, auroc, width=w, label="AUROC (%) ↑")
            ax.bar(x, oscr, width=w, label="OSCR (%) ↑")
            ax.bar(x + w, fpr95, width=w, label="FPR@TPR95 (%) ↓")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=7.5, rotation=20, ha="right")
            ax.set_ylim(0, 108)
            ax.set_ylabel("Percentage (%)")
            ax.set_title(title, fontsize=10)
            ax.grid(True, axis="y", linestyle="--", alpha=0.4)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "stage3_osr_config_a_vs_b.png", dpi=150)
        plt.close(fig)

    # Figure 2: CIL Baselines Order A & Order B
    if cil_res:
        fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.2))
        for ax, order_name, title in [
            (axes[0], "order_a", "Order A Trajectories (7 -> 9 -> 11: QAM64+AM-SSB, WBFM+AM-DSB)"),
            (axes[1], "order_b", "Order B Trajectories (7 -> 9 -> 11: QAM64+WBFM, AM-SSB+AM-DSB)"),
        ]:
            runs = [r for r in cil_res if r["order"] == order_name]
            for r in runs:
                xs = [s["num_classes"] for s in r["sessions"]]
                ys = [s["a_all_pct"] for s in r["sessions"]]
                lw = 2.6 if "Ours" in r["method"] else 1.8
                ls = "-" if ("Ours" in r["method"] or "iCaRL" in r["method"]) else "--"
                ax.plot(xs, ys, marker="o", linewidth=lw, linestyle=ls, label=r["method"])
            ax.set_xticks([7, 9, 11])
            ax.set_ylim(0, 104)
            ax.set_xlabel("Total Seen Classes (C)")
            ax.set_ylabel("All-Class Accuracy A_all (%)")
            ax.set_title(title, fontsize=9.8)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "stage3_cil_baselines.png", dpi=150)
        plt.close(fig)

    # Figure 3: Cross-SNR Drift (OSR & CIL)
    if drift_res:
        fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

        # Panel 1: Session 0 OSR AUROC vs SNR
        ax = axes[0]
        for label, dr in drift_res["session0_osr_drift"].items():
            snrs = sorted(int(k) for k in dr["per_snr"].keys())
            vals = [dr["per_snr"][str(s)]["auroc_pct"] for s in snrs]
            ax.plot(snrs, vals, marker="o", linewidth=2, label=f"S0 {label}")
        ax.axvline(9.0, color="gray", linestyle=":", label="Train Cutoff (10 dB)")
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("AUROC (%)")
        ax.set_ylim(30, 102)
        ax.set_title("Session 0 OSR AUROC vs SNR")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)

        # Panel 2: Post-S1 OSR Known FRR vs SNR (Synced vs Unsynced)
        ax2 = axes[1]
        for label, dr in drift_res["post_s1_osr_drift"].items():
            snrs = sorted(int(k) for k in dr["per_snr"].keys())
            vals = [dr["per_snr"][str(s)]["known_frr_pct"] for s in snrs]
            ax2.plot(snrs, vals, marker="s", linewidth=2, label=label)
        ax2.axvline(9.0, color="gray", linestyle=":", label="Train Cutoff (10 dB)")
        ax2.set_xlabel("SNR (dB)")
        ax2.set_ylabel("Known False Rejection Rate (%) ↓")
        ax2.set_ylim(0, 102)
        ax2.set_title("Post-S1 OSR Boundary Drift (Fixed High-SNR Thr)")
        ax2.grid(True, linestyle="--", alpha=0.4)
        ax2.legend(fontsize=8)

        # Panel 3: Post-S1 CIL A_all vs SNR
        ax3 = axes[2]
        for label, dr in drift_res["post_s1_cil_drift"].items():
            snrs = sorted(int(k) for k in dr["per_snr"].keys())
            vals = [dr["per_snr"][str(s)]["a_all_pct"] for s in snrs]
            lw = 2.5 if "Ours" in label else 1.8
            ax3.plot(snrs, vals, marker="o", linewidth=lw, label=label)
        ax3.axvline(9.0, color="gray", linestyle=":", label="Train Cutoff (10 dB)")
        ax3.set_xlabel("SNR (dB)")
        ax3.set_ylabel("All-Class Accuracy A_all (%)")
        ax3.set_ylim(0, 104)
        ax3.set_title("Post-S1 (C=9) Accuracy Across SNR Regimes")
        ax3.grid(True, linestyle="--", alpha=0.4)
        ax3.legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(fig_dir / "stage3_snr_drift.png", dpi=150)
        plt.close(fig)


def write_stage3_summary(
    osr_res: dict[str, Any],
    cil_res: list[dict[str, Any]],
    drift_res: dict[str, Any],
    out_root: Path,
) -> None:
    save_json({"static_osr": osr_res, "cil_baselines": cil_res, "snr_drift": drift_res}, out_root / "summary_stage3.json")

    lines = ["# 阶段 3 跨基线对齐与 SNR 环境漂移汇总报告", ""]

    if osr_res:
        for key, heading in [
            ("config_a", "## 一、 静态开集基线对比（配置 A：7 已知 + 4 未知，Openness = 11.81%）"),
            ("config_b", "## 二、 高开放度泛化对比（配置 B：5 已知 + 6 未知，Openness = 20.94%）"),
        ]:
            rows = osr_res.get(key, [])
            if not rows:
                continue
            lines.extend(
                [
                    heading,
                    "",
                    "| 骨干网络 | 拒识算法 | Known Acc (%) | AUROC (%) ↑ | FPR@TPR95 (%) ↓ | OSCR (%) ↑ |",
                    "| :--- | :--- | :---: | :---: | :---: | :---: |",
                ]
            )
            for r in rows:
                lines.append(
                    f"| {r['backbone']} | {r['algorithm']} | {r['known_acc_pct']:.2f} | "
                    f"{r['auroc_pct']:.2f} | {r['fpr_at_tpr95_pct']:.2f} | {r['oscr_pct']:.2f} |"
                )
            lines.append("")

    if cil_res:
        lines.extend(
            [
                "## 三、 增量学习横向基线对比（配置 A：Order A / Order B，$7 \\rightarrow 9 \\rightarrow 11$）",
                "",
                "| 协议 | 方法 | 会话 | $C$ | 发现门控 | $A_{\\text{all}}$ (%) | $A_{\\text{old}}$ (%) | $A_{\\text{new}}$ (%) | $\\Delta A_{\\text{old}}$ (%) | $FM_{\\text{cls}}$ (%) |",
                "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        for run in cil_res:
            for s in run["sessions"]:
                a_new_str = f"{s['a_new_pct']:.2f}" if s["a_new_pct"] is not None else "-"
                gate_str = "Pass" if s.get("discovery_accepted", True) else "Hold"
                lines.append(
                    f"| {run['order']} | {run['method']} | S{s['session']} | {s['num_classes']} | {gate_str} | "
                    f"{s['a_all_pct']:.2f} | {s['a_old_pct']:.2f} | {a_new_str} | "
                    f"{s['delta_a_old_pct']:.2f} | {s['fm_cls_pct']:.2f} |"
                )
        lines.append("")

    if drift_res:
        lines.extend(
            [
                "## 四、 DOCO 跨 SNR 环境漂移评估（严格在 High-SNR $\\ge 10\\text{ dB}$ 训练与增量）",
                "",
                "### 4.1 开集拒识边界漂移（固定 High-SNR Val 标定阈值）",
                "",
                "| 阶段与方法 | High (10~18 dB) Acc / FRR / AUROC | Mid-Low (0~6 dB) Acc / FRR / AUROC | Neg (-10~-2 dB) Acc / FRR / AUROC | FRR 漂移 (Mid-Low / Neg) |",
                "| :--- | :---: | :---: | :---: | :---: |",
            ]
        )
        for prefix, bucket in [("S0", drift_res["session0_osr_drift"]), ("Post-S1", drift_res["post_s1_osr_drift"])]:
            for name, dr in bucket.items():
                r = dr["regimes"]
                d = dr["drift_summary"]
                hi = f"{r['high_snr']['known_acc_pct']:.1f} / {r['high_snr']['known_frr_pct']:.1f} / {r['high_snr']['auroc_pct']:.1f}"
                ml = f"{r['mid_low_snr']['known_acc_pct']:.1f} / {r['mid_low_snr']['known_frr_pct']:.1f} / {r['mid_low_snr']['auroc_pct']:.1f}"
                ng = f"{r['neg_snr']['known_acc_pct']:.1f} / {r['neg_snr']['known_frr_pct']:.1f} / {r['neg_snr']['auroc_pct']:.1f}"
                lines.append(
                    f"| {prefix} {name} | {hi} | {ml} | {ng} | +{d['frr_drift_mid_low_pct']:.1f}% / +{d['frr_drift_neg_pct']:.1f}% |"
                )

        lines.extend(
            [
                "",
                "### 4.2 Post-S1 ($C=9$) 增量准确率跨 SNR 退化",
                "",
                "| 方法 | High (10~18 dB) $A_{\\text{all}} / A_{\\text{old}} / A_{\\text{new}}$ | Mid-Low (0~6 dB) $A_{\\text{all}} / A_{\\text{old}} / A_{\\text{new}}$ | Neg (-10~-2 dB) $A_{\\text{all}} / A_{\\text{old}} / A_{\\text{new}}$ | $A_{\\text{all}}$ 退化 (Mid-Low / Neg) |",
                "| :--- | :---: | :---: | :---: | :---: |",
            ]
        )
        for name, dr in drift_res["post_s1_cil_drift"].items():
            r = dr["regimes"]
            d = dr["drift_summary"]
            hi = f"{r['high_snr']['a_all_pct']:.1f} / {r['high_snr']['a_old_pct']:.1f} / {r['high_snr']['a_new_pct']:.1f}"
            ml = f"{r['mid_low_snr']['a_all_pct']:.1f} / {r['mid_low_snr']['a_old_pct']:.1f} / {r['mid_low_snr']['a_new_pct']:.1f}"
            ng = f"{r['neg_snr']['a_all_pct']:.1f} / {r['neg_snr']['a_old_pct']:.1f} / {r['neg_snr']['a_new_pct']:.1f}"
            lines.append(
                f"| {name} | {hi} | {ml} | {ng} | -{d['a_all_drop_mid_low_pct']:.1f}% / -{d['a_all_drop_neg_pct']:.1f}% |"
            )

    (out_root / "summary_stage3.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    split_a_cfg = load_yaml(cfg["split_config_a"])
    split_b_cfg = load_yaml(cfg["split_config_b"])

    if args.seed is not None:
        cfg["seed"] = int(args.seed)
        split_a_cfg["seed"] = int(args.seed)
        split_b_cfg["seed"] = int(args.seed)
    if args.device is not None:
        cfg["device"] = args.device

    seed = int(cfg.get("seed", 42))
    seed_everything(seed, deterministic=True)
    req_dev = str(cfg.get("device", "cuda"))
    device = torch.device("cpu") if (req_dev == "cuda" and not torch.cuda.is_available()) else torch.device(req_dev)

    out_root = resolve_path(cfg.get("output_dir", "outputs/stage3"))
    ckpt_dir = out_root / "checkpoints"
    fig_dir = out_root / "figures"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, out_root / "config.yaml")

    logger = setup_logger("stage3", log_file=out_root / "stage3.log")
    logger.info("Stage 3 running on device=%s | seed=%d | sections=%s", device, seed, args.sections)

    bundle_a = SplitBundle.from_config(split_a_cfg)
    bundle_a.verify_disjoint()
    bundle_b = SplitBundle.from_config(split_b_cfg)
    bundle_b.verify_disjoint()

    if args.sections.lower() == "all":
        active_sections = {"osr", "cil", "drift"}
    else:
        active_sections = {s.strip().lower() for s in args.sections.split(",") if s.strip()}

    osr_res: dict[str, Any] = {}
    cil_res: list[dict[str, Any]] = []
    drift_res: dict[str, Any] = {}

    if "osr" in active_sections:
        logger.info("========== Stage 3 Section 1: Static OSR (SSD-AMC & Config B) ==========")
        osr_res = run_static_osr_section(
            cfg=cfg,
            bundle_a=bundle_a,
            bundle_b=bundle_b,
            device=device,
            ckpt_dir=ckpt_dir,
            reuse_trained=args.reuse_trained,
            logger=logger,
        )

    if "cil" in active_sections:
        logger.info("========== Stage 3 Section 2: Continual Learning Horizontal Baselines ==========")
        cil_res = run_cil_baselines_section(cfg=cfg, bundle_a=bundle_a, device=device, logger=logger)

    if "drift" in active_sections:
        logger.info("========== Stage 3 Section 3: Cross-SNR Environment Drift Evaluation ==========")
        drift_res = run_snr_drift_section(
            cfg=cfg,
            bundle_a=bundle_a,
            device=device,
            ckpt_dir=ckpt_dir,
            reuse_trained=args.reuse_trained,
            logger=logger,
        )

    # Preserve previously generated sections when running a partial --sections subset
    existing_json = out_root / "summary_stage3.json"
    if active_sections != {"osr", "cil", "drift"} and existing_json.exists():
        prev = json.loads(existing_json.read_text(encoding="utf-8"))
        if not osr_res:
            osr_res = prev.get("static_osr", {})
        if not cil_res:
            cil_res = prev.get("cil_baselines", [])
        if not drift_res:
            drift_res = prev.get("snr_drift", {})

    build_stage3_plots(osr_res, cil_res, drift_res, fig_dir)
    write_stage3_summary(osr_res, cil_res, drift_res, out_root)
    logger.info("Stage 3 summary & figures saved to %s", out_root)


if __name__ == "__main__":
    main()
