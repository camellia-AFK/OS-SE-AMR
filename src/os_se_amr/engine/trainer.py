"""Closed-set supervised trainer and Stage 0 evaluation / visualization utilities."""

from __future__ import annotations

import logging
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

from os_se_amr.models.base import BaseBackbone
from os_se_amr.utils.checkpoint import load_checkpoint, save_checkpoint


class ClosedSetTrainer:
    """Supervised trainer for Stage 0 closed-set AMR baselines."""

    def __init__(
        self,
        model: BaseBackbone,
        device: torch.device,
        train_cfg: dict[str, Any],
        output_dir: str | Path,
        logger: logging.Logger,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.train_cfg = train_cfg
        self.output_dir = Path(output_dir)
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.fig_dir = self.output_dir / "figures"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger

        self.epochs = int(train_cfg.get("epochs", 50))
        self.lr = float(train_cfg.get("lr", 1e-3))
        self.weight_decay = float(train_cfg.get("weight_decay", 1e-4))
        self.grad_clip = float(train_cfg.get("grad_clip", 1.0))
        self.patience = int(train_cfg.get("early_stopping_patience", 12))
        self.use_amp = bool(train_cfg.get("use_amp", False)) and self.device.type == "cuda"
        self._amp_dtype = (
            torch.bfloat16 if (self.use_amp and torch.cuda.is_bf16_supported()) else torch.float16
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.use_amp and self._amp_dtype == torch.float16))

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, self.epochs),
            eta_min=1e-6,
        )

    def _run_epoch(self, loader: DataLoader) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for x, y, _ in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=self._amp_dtype, enabled=self.use_amp):
                logits, _ = self.model(x)
                loss = self.criterion(logits, y)
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                self.optimizer.step()

            bs = x.size(0)
            total_loss += float(loss.item()) * bs
            total_correct += int((logits.argmax(dim=1) == y).sum().item())
            total_samples += bs

        return total_loss / max(1, total_samples), total_correct / max(1, total_samples)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, class_names: list[str]) -> dict[str, Any]:
        """Evaluate accuracy, per-class/SNR breakdown, confusion matrix, and 64-d feature health."""
        self.model.eval()
        num_classes = len(class_names)
        total_loss = 0.0
        all_preds: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []
        all_snrs: list[np.ndarray] = []
        all_feats: list[torch.Tensor] = []

        for x, y, snr in loader:
            x = x.to(self.device, non_blocking=True)
            y_dev = y.to(self.device, non_blocking=True)
            logits, feats = self.model(x)
            loss = self.criterion(logits, y_dev)

            total_loss += float(loss.item()) * x.size(0)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_targets.append(y.numpy())
            all_snrs.append(snr.numpy())
            all_feats.append(feats.cpu())

        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        snrs = np.concatenate(all_snrs, axis=0)
        feats_cat = torch.cat(all_feats, dim=0)

        n_samples = len(targets)
        overall_acc = float((preds == targets).mean()) if n_samples > 0 else 0.0

        per_class_acc: dict[str, float] = {}
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for t_idx, p_idx in zip(targets, preds):
            cm[t_idx, p_idx] += 1
        for c_idx, c_name in enumerate(class_names):
            mask = targets == c_idx
            per_class_acc[c_name] = float((preds[mask] == targets[mask]).mean()) if mask.any() else 0.0

        per_snr_acc: dict[str, float] = {}
        for snr_val in sorted(np.unique(snrs).tolist()):
            mask = snrs == snr_val
            per_snr_acc[str(int(snr_val))] = float((preds[mask] == targets[mask]).mean())

        norms = feats_cat.norm(p=2, dim=1)
        dim_std = feats_cat.std(dim=0, unbiased=False)
        feature_stats = {
            "dim": int(feats_cat.shape[1]),
            "has_nan": bool(torch.isnan(feats_cat).any().item()),
            "has_inf": bool(torch.isinf(feats_cat).any().item()),
            "mean_norm": float(norms.mean().item()),
            "std_norm": float(norms.std(unbiased=False).item()),
            "min_dim_std": float(dim_std.min().item()),
            "max_dim_std": float(dim_std.max().item()),
        }

        return {
            "loss": total_loss / max(1, n_samples),
            "overall_acc": overall_acc,
            "per_class_acc": per_class_acc,
            "per_snr_acc": per_snr_acc,
            "confusion_matrix": cm.tolist(),
            "feature_stats": feature_stats,
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_names: list[str],
    ) -> dict[str, Any]:
        """Train closed-set baseline with cosine annealing and Val-acc early stopping."""
        best_val_acc = -1.0
        best_epoch = 0
        no_improve = 0
        history: list[dict[str, Any]] = []
        best_ckpt_path = self.ckpt_dir / "best.pt"
        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self._run_epoch(train_loader)
            val_metrics = self.evaluate(val_loader, class_names)
            val_acc = val_metrics["overall_acc"]
            val_loss = val_metrics["loss"]
            current_lr = float(self.optimizer.param_groups[0]["lr"])
            self.scheduler.step()

            record = {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": round(train_loss, 5),
                "train_acc": round(train_acc, 5),
                "val_loss": round(val_loss, 5),
                "val_acc": round(val_acc, 5),
            }
            history.append(record)

            improved = val_acc > best_val_acc
            if improved:
                best_val_acc = val_acc
                best_epoch = epoch
                no_improve = 0
                save_checkpoint(
                    path=best_ckpt_path,
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    metrics=val_metrics,
                    extra={"class_names": class_names},
                )
            else:
                no_improve += 1

            if epoch == 1 or epoch % 5 == 0 or improved or epoch == self.epochs:
                self.logger.info(
                    "Epoch %02d/%02d | lr=%.2e | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f%s",
                    epoch,
                    self.epochs,
                    current_lr,
                    train_loss,
                    train_acc,
                    val_loss,
                    val_acc,
                    " [BEST]" if improved else "",
                )

            if no_improve >= self.patience:
                self.logger.info(
                    "Early stopping triggered at epoch %d (best epoch=%d, best_val_acc=%.4f)",
                    epoch,
                    best_epoch,
                    best_val_acc,
                )
                break

        load_checkpoint(best_ckpt_path, self.model, device=self.device)
        elapsed = time.time() - start_time
        return {
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "elapsed_sec": round(elapsed, 2),
            "history": history,
            "best_checkpoint": str(best_ckpt_path),
        }

    def save_figures(self, test_metrics: dict[str, Any], class_names: list[str], title_prefix: str) -> None:
        """Save Acc-vs-SNR curve and normalized confusion matrix plots."""
        per_snr = test_metrics["per_snr_acc"]
        snr_keys = sorted(per_snr.keys(), key=lambda s: int(s))
        snr_vals = [int(k) for k in snr_keys]
        accs = [per_snr[k] * 100.0 for k in snr_keys]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(snr_vals, accs, marker="o", linewidth=2)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("Classification Accuracy (%)")
        ax.set_title(f"{title_prefix} - Accuracy vs SNR (Overall: {test_metrics['overall_acc'] * 100:.2f}%)")
        ax.set_ylim(0, 102)
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(self.fig_dir / "acc_vs_snr.png", dpi=150)
        plt.close(fig)

        cm = np.asarray(test_metrics["confusion_matrix"], dtype=np.float64)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, np.maximum(row_sums, 1.0)) * 100.0

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=100.0)
        fig.colorbar(im, ax=ax, label="Recall (%)")
        ticks = np.arange(len(class_names))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{title_prefix} - Confusion Matrix")

        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                val = cm_norm[i, j]
                color = "white" if val > 55.0 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", color=color, fontsize=8)

        fig.tight_layout()
        fig.savefig(self.fig_dir / "confusion_matrix.png", dpi=150)
        plt.close(fig)
