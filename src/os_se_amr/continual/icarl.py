"""iCaRL baseline: combined exemplar + new replay, logit KD, and Nearest-Mean-of-Exemplars (NME)."""

from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from os_se_amr.continual.lwf import EvolutionReport
from os_se_amr.continual.replay import ReplayBuffer
from os_se_amr.models.base import BaseBackbone


class iCaRLLearner:
    """Faithful iCaRL baseline with old-head logit distillation and NME evaluation."""

    def __init__(
        self,
        epochs: int = 35,
        batch_size: int = 128,
        lr: float = 8e-4,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
        lambda_kd: float = 1.0,
        distill_temp: float = 2.0,
        eval_batch_size: int = 256,
    ) -> None:
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.grad_clip = float(grad_clip)
        self.lambda_kd = float(lambda_kd)
        self.distill_temp = float(distill_temp)
        self.eval_batch_size = int(eval_batch_size)
        self.class_means: dict[int, np.ndarray] = {}

    def evolve(
        self,
        backbone: BaseBackbone,
        replay_buffer: ReplayBuffer,
        new_samples: np.ndarray | torch.Tensor,
        new_pseudo_labels: np.ndarray | torch.Tensor,
        device: str | torch.device = "cuda",
        seed: int = 42,
    ) -> EvolutionReport:
        """Expand classifier and jointly train on replay exemplars + new samples with logit KD."""
        dev = torch.device(device)
        x_new = torch.as_tensor(new_samples, dtype=torch.float32)
        y_local = torch.as_tensor(new_pseudo_labels, dtype=torch.long).reshape(-1)
        if len(x_new) == 0:
            raise ValueError("new_samples must not be empty")
        if len(x_new) != len(y_local):
            raise ValueError(f"Length mismatch: new_samples={len(x_new)}, new_pseudo_labels={len(y_local)}")

        min_lbl = int(y_local.min().item())
        max_lbl = int(y_local.max().item())
        if min_lbl < 0:
            raise ValueError(f"Noise labels (-1) must be filtered out before evolve(), got min={min_lbl}")

        num_old = int(backbone.num_classes)
        num_new = max_lbl + 1

        teacher = copy.deepcopy(backbone).to(dev).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        backbone.expand_classifier(num_new)
        backbone.to(dev)

        y_global_new = y_local + num_old
        old_x, old_y = replay_buffer.as_tensors()
        if len(old_x) > 0:
            x_all = torch.cat([old_x, x_new], dim=0)
            y_all = torch.cat([old_y, y_global_new], dim=0)
        else:
            x_all = x_new
            y_all = y_global_new

        train_ds = TensorDataset(x_all, y_all)
        gen_cpu = torch.Generator().manual_seed(int(seed))
        loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, generator=gen_cpu)

        optimizer = torch.optim.AdamW(
            backbone.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, self.epochs), eta_min=1e-6
        )

        temp = self.distill_temp
        history: list[dict[str, float]] = []
        t0 = time.time()

        for epoch in range(1, self.epochs + 1):
            backbone.train()
            ep_total, ep_ce, ep_kd, steps = 0.0, 0.0, 0.0, 0

            for xb, yb in loader:
                xb = xb.to(dev, non_blocking=True)
                yb = yb.to(dev, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                s_logits, _ = backbone(xb)
                ce_loss = F.cross_entropy(s_logits, yb)

                logit_kd = xb.new_zeros(())
                if self.lambda_kd > 0.0:
                    with torch.no_grad():
                        t_logits, _ = teacher(xb)
                    p_t = F.softmax(t_logits / temp, dim=-1)
                    log_p_s = F.log_softmax(s_logits[:, :num_old] / temp, dim=-1)
                    logit_kd = F.kl_div(log_p_s, p_t, reduction="batchmean") * (temp * temp)

                loss = ce_loss + self.lambda_kd * logit_kd
                loss.backward()
                if self.grad_clip > 0.0:
                    nn.utils.clip_grad_norm_(backbone.parameters(), max_norm=self.grad_clip)
                optimizer.step()

                ep_total += float(loss.item())
                ep_ce += float(ce_loss.item())
                ep_kd += float(logit_kd.item())
                steps += 1

            scheduler.step()
            denom = float(max(1, steps))
            history.append(
                {
                    "epoch": float(epoch),
                    "loss": round(ep_total / denom, 5),
                    "ce_loss": round(ep_ce / denom, 5),
                    "feat_kd": 0.0,
                    "logit_kd": round(ep_kd / denom, 5),
                }
            )

        last = history[-1]
        return EvolutionReport(
            num_old_classes=num_old,
            num_new_classes=num_new,
            epochs=self.epochs,
            final_loss=last["loss"],
            final_ce_loss=last["ce_loss"],
            final_feat_kd=0.0,
            final_logit_kd=last["logit_kd"],
            elapsed_sec=round(time.time() - t0, 2),
            history=history,
            bic_gammas=[],
        )

    @torch.no_grad()
    def update_class_means(
        self,
        backbone: BaseBackbone,
        replay_buffer: ReplayBuffer,
        device: str | torch.device = "cuda",
    ) -> None:
        """Recompute L2-normalized exemplar class means in current `penult` feature space."""
        dev = torch.device(device)
        backbone.to(dev)
        backbone.eval()

        old_x, old_y = replay_buffer.as_tensors()
        if len(old_x) == 0:
            raise ValueError("replay_buffer must not be empty when updating iCaRL class means")

        feats_list: list[np.ndarray] = []
        for start in range(0, len(old_x), self.eval_batch_size):
            xb = old_x[start : start + self.eval_batch_size].to(dev, non_blocking=True)
            feats = F.normalize(backbone.extract_features(xb), p=2, dim=-1)
            feats_list.append(feats.cpu().numpy())
        unit_feats = np.concatenate(feats_list, axis=0)
        y_np = old_y.numpy()

        self.class_means = {}
        for cid in replay_buffer.class_ids:
            mask = y_np == cid
            if not mask.any():
                continue
            mean = unit_feats[mask].astype(np.float64).mean(axis=0)
            norm = max(float(np.linalg.norm(mean)), 1e-8)
            self.class_means[int(cid)] = (mean / norm).astype(np.float32)

    @torch.no_grad()
    def predict_nme(
        self,
        backbone: BaseBackbone,
        ds: Any,
        device: str | torch.device = "cuda",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict labels via Nearest-Mean-of-Exemplars (NME) over L2-normalized `penult`."""
        if not self.class_means:
            raise RuntimeError("iCaRLLearner must call update_class_means() before predict_nme()")

        dev = torch.device(device)
        backbone.to(dev)
        backbone.eval()

        class_ids = sorted(self.class_means.keys())
        mean_mat = np.stack([self.class_means[c] for c in class_ids], axis=0).astype(np.float32)
        mean_t = torch.from_numpy(mean_mat).to(dev)
        class_arr = np.asarray(class_ids, dtype=np.int64)

        loader = DataLoader(ds, batch_size=self.eval_batch_size, shuffle=False, num_workers=0)
        targets: list[np.ndarray] = []
        preds: list[np.ndarray] = []

        for batch in loader:
            xb = batch[0].to(dev, non_blocking=True)
            yb = batch[1].numpy()
            z_unit = F.normalize(backbone.extract_features(xb), p=2, dim=-1)
            sims = z_unit @ mean_t.T
            best_pos = sims.argmax(dim=1).cpu().numpy()
            preds.append(class_arr[best_pos])
            targets.append(yb)

        return np.concatenate(targets, axis=0), np.concatenate(preds, axis=0)
