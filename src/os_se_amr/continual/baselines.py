"""Additional SOTA Continual Learning baselines: EWC (PNAS 2017), WA (CVPR 2020), and DER (CVPR 2021)."""

from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from os_se_amr.continual.lwf import EvolutionReport, apply_weight_alignment
from os_se_amr.continual.replay import ReplayBuffer
from os_se_amr.models import build_backbone
from os_se_amr.models.base import BaseBackbone
from os_se_amr.models.heads import ExpandableLinearHead


class EWCLearner:
    """Elastic Weight Consolidation (Kirkpatrick et al., PNAS 2017) with exemplar replay."""

    def __init__(
        self,
        epochs: int = 35,
        batch_size: int = 128,
        lr: float = 8e-4,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
        lambda_ewc: float = 400.0,
    ) -> None:
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.grad_clip = float(grad_clip)
        self.lambda_ewc = float(lambda_ewc)
        self._fisher: dict[str, torch.Tensor] = {}
        self._prev_params: dict[str, torch.Tensor] = {}

    def _compute_fisher(
        self,
        backbone: BaseBackbone,
        replay_buffer: ReplayBuffer,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        old_x, old_y = replay_buffer.as_tensors()
        if len(old_x) == 0:
            return {}
        backbone.to(device)
        backbone.eval()
        fisher = {n: torch.zeros_like(p, device=device) for n, p in backbone.named_parameters() if p.requires_grad}
        loader = DataLoader(TensorDataset(old_x, old_y), batch_size=min(64, len(old_x)), shuffle=False)
        total = 0
        # Disable cuDNN RNN kernel during eval-mode Fisher backward so BatchNorm/Dropout remain frozen on CUDA.
        with torch.backends.cudnn.flags(enabled=False):
            for xb, yb in loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                backbone.zero_grad(set_to_none=True)
                logits, _ = backbone(xb)
                loss = F.cross_entropy(logits, yb, reduction="sum")
                loss.backward()
                for n, p in backbone.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        fisher[n] += p.grad.detach() ** 2
                total += len(xb)
        backbone.zero_grad(set_to_none=True)
        denom = float(max(1, total))
        return {k: (v / denom).detach() for k, v in fisher.items()}

    def evolve(
        self,
        backbone: BaseBackbone,
        replay_buffer: ReplayBuffer,
        new_samples: np.ndarray | torch.Tensor,
        new_pseudo_labels: np.ndarray | torch.Tensor,
        device: str | torch.device = "cuda",
        seed: int = 42,
    ) -> EvolutionReport:
        dev = torch.device(device)
        x_new = torch.as_tensor(new_samples, dtype=torch.float32)
        y_local = torch.as_tensor(new_pseudo_labels, dtype=torch.long).reshape(-1)
        if len(x_new) == 0:
            raise ValueError("new_samples must not be empty")
        if len(x_new) != len(y_local):
            raise ValueError("new_samples and new_pseudo_labels length mismatch")
        if int(y_local.min().item()) < 0:
            raise ValueError("Noise labels (-1) must be removed before evolve()")

        num_old = int(backbone.num_classes)
        num_new = int(y_local.max().item()) + 1

        new_f = self._compute_fisher(backbone, replay_buffer, dev)
        for k, v in new_f.items():
            if k in self._fisher and self._fisher[k].shape == v.shape:
                self._fisher[k] = 0.5 * self._fisher[k].to(dev) + 0.5 * v
            else:
                self._fisher[k] = v
        self._prev_params = {
            n: p.detach().clone().to(dev) for n, p in backbone.named_parameters() if p.requires_grad
        }

        backbone.expand_classifier(num_new)
        backbone.to(dev)

        y_global_new = y_local + num_old
        old_x, old_y = replay_buffer.as_tensors()
        x_all = torch.cat([old_x, x_new], dim=0) if len(old_x) > 0 else x_new
        y_all = torch.cat([old_y, y_global_new], dim=0) if len(old_y) > 0 else y_global_new
        loader = DataLoader(
            TensorDataset(x_all, y_all),
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(int(seed)),
        )
        optimizer = torch.optim.AdamW(backbone.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, self.epochs), eta_min=1e-6)

        history: list[dict[str, float]] = []
        t0 = time.time()
        for epoch in range(1, self.epochs + 1):
            backbone.train()
            ep_tot, ep_ce, ep_reg, steps = 0.0, 0.0, 0.0, 0
            for xb, yb in loader:
                xb = xb.to(dev, non_blocking=True)
                yb = yb.to(dev, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits, _ = backbone(xb)
                ce_loss = F.cross_entropy(logits, yb)

                ewc_reg = xb.new_zeros(())
                for n, p in backbone.named_parameters():
                    if n in self._fisher and n in self._prev_params:
                        prev_p = self._prev_params[n]
                        fish = self._fisher[n]
                        if p.shape == prev_p.shape:
                            ewc_reg = ewc_reg + (fish * (p - prev_p) ** 2).sum()
                        elif p.ndim == prev_p.ndim and p.shape[1:] == prev_p.shape[1:]:
                            # Protect old rows of expanded classifier head
                            rows = prev_p.shape[0]
                            ewc_reg = ewc_reg + (fish[:rows] * (p[:rows] - prev_p[:rows]) ** 2).sum()

                loss = ce_loss + 0.5 * self.lambda_ewc * ewc_reg
                loss.backward()
                if self.grad_clip > 0.0:
                    nn.utils.clip_grad_norm_(backbone.parameters(), max_norm=self.grad_clip)
                optimizer.step()
                ep_tot += float(loss.item())
                ep_ce += float(ce_loss.item())
                ep_reg += float(ewc_reg.item())
                steps += 1
            scheduler.step()
            denom = float(max(1, steps))
            history.append(
                {
                    "epoch": float(epoch),
                    "loss": round(ep_tot / denom, 5),
                    "ce_loss": round(ep_ce / denom, 5),
                    "feat_kd": 0.0,
                    "logit_kd": round(ep_reg / denom, 5),
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
        )


class WALearner:
    """Weight Aligning baseline (Zhao et al., CVPR 2020): replay + logit KD + weight norm alignment."""

    def __init__(
        self,
        epochs: int = 35,
        batch_size: int = 128,
        lr: float = 8e-4,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
        lambda_kd: float = 1.0,
        distill_temp: float = 2.0,
    ) -> None:
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.grad_clip = float(grad_clip)
        self.lambda_kd = float(lambda_kd)
        self.distill_temp = float(distill_temp)

    def evolve(
        self,
        backbone: BaseBackbone,
        replay_buffer: ReplayBuffer,
        new_samples: np.ndarray | torch.Tensor,
        new_pseudo_labels: np.ndarray | torch.Tensor,
        device: str | torch.device = "cuda",
        seed: int = 42,
    ) -> EvolutionReport:
        dev = torch.device(device)
        x_new = torch.as_tensor(new_samples, dtype=torch.float32)
        y_local = torch.as_tensor(new_pseudo_labels, dtype=torch.long).reshape(-1)
        if len(x_new) == 0:
            raise ValueError("new_samples must not be empty")
        if int(y_local.min().item()) < 0:
            raise ValueError("Noise labels (-1) must be removed before evolve()")

        num_old = int(backbone.num_classes)
        num_new = int(y_local.max().item()) + 1

        teacher = copy.deepcopy(backbone).to(dev).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        backbone.expand_classifier(num_new)
        backbone.to(dev)

        y_global_new = y_local + num_old
        old_x, old_y = replay_buffer.as_tensors()
        x_all = torch.cat([old_x, x_new], dim=0) if len(old_x) > 0 else x_new
        y_all = torch.cat([old_y, y_global_new], dim=0) if len(old_y) > 0 else y_global_new
        loader = DataLoader(
            TensorDataset(x_all, y_all),
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(int(seed)),
        )
        optimizer = torch.optim.AdamW(backbone.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, self.epochs), eta_min=1e-6)
        temp = self.distill_temp

        history: list[dict[str, float]] = []
        t0 = time.time()
        for epoch in range(1, self.epochs + 1):
            backbone.train()
            ep_tot, ep_ce, ep_kd, steps = 0.0, 0.0, 0.0, 0
            for xb, yb in loader:
                xb = xb.to(dev, non_blocking=True)
                yb = yb.to(dev, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                s_logits, _ = backbone(xb)
                ce_loss = F.cross_entropy(s_logits, yb)
                kd_loss = xb.new_zeros(())
                if self.lambda_kd > 0.0:
                    with torch.no_grad():
                        t_logits, _ = teacher(xb)
                    kd_loss = (
                        F.kl_div(
                            F.log_softmax(s_logits[:, :num_old] / temp, dim=-1),
                            F.softmax(t_logits / temp, dim=-1),
                            reduction="batchmean",
                        )
                        * (temp * temp)
                    )
                loss = ce_loss + self.lambda_kd * kd_loss
                loss.backward()
                if self.grad_clip > 0.0:
                    nn.utils.clip_grad_norm_(backbone.parameters(), max_norm=self.grad_clip)
                optimizer.step()
                ep_tot += float(loss.item())
                ep_ce += float(ce_loss.item())
                ep_kd += float(kd_loss.item())
                steps += 1
            scheduler.step()
            denom = float(max(1, steps))
            history.append(
                {
                    "epoch": float(epoch),
                    "loss": round(ep_tot / denom, 5),
                    "ce_loss": round(ep_ce / denom, 5),
                    "feat_kd": 0.0,
                    "logit_kd": round(ep_kd / denom, 5),
                }
            )

        wa_gamma = apply_weight_alignment(backbone, num_old)
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
            wa_gamma=round(float(wa_gamma), 4),
        )


class DERWrapperBackbone(BaseBackbone):
    """Multi-branch Dynamic Expandable Representation wrapper (Yan et al., CVPR 2021)."""

    def __init__(self, initial_backbone: BaseBackbone) -> None:
        d = int(initial_backbone.feature_dim)
        super().__init__(num_classes=int(initial_backbone.num_classes), feature_dim=d)
        self.branches = nn.ModuleList([initial_backbone])
        self.classifier = ExpandableLinearHead(d, int(initial_backbone.num_classes))
        with torch.no_grad():
            if hasattr(initial_backbone.classifier, "fc"):
                self.classifier.fc.weight.copy_(initial_backbone.classifier.fc.weight.detach())
                if self.classifier.fc.bias is not None and initial_backbone.classifier.fc.bias is not None:
                    self.classifier.fc.bias.copy_(initial_backbone.classifier.fc.bias.detach())

    def add_branch(self, new_branch: BaseBackbone, num_new_classes: int) -> None:
        """Freeze prior branches, append a new feature branch, and expand super-feature classifier."""
        for branch in self.branches:
            branch.eval()
            for p in branch.parameters():
                p.requires_grad_(False)
        self.branches.append(new_branch)

        old_cls = int(self.num_classes)
        new_total = old_cls + int(num_new_classes)
        old_in = int(self.classifier.fc.in_features)
        new_in = sum(int(b.feature_dim) for b in self.branches)

        old_w = self.classifier.fc.weight.detach()
        old_b = self.classifier.fc.bias.detach() if self.classifier.fc.bias is not None else None
        new_head = ExpandableLinearHead(new_in, new_total).to(device=old_w.device, dtype=old_w.dtype)
        with torch.no_grad():
            new_head.fc.weight.zero_()
            new_head.fc.weight[:old_cls, :old_in].copy_(old_w)
            if new_head.fc.bias is not None and old_b is not None:
                new_head.fc.bias[:old_cls].copy_(old_b)
        self.classifier = new_head
        self.num_classes = new_total
        self.feature_dim = new_in

    def extract_super_features(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate penultimate features across all frozen + active branches."""
        parts = []
        for idx, branch in enumerate(self.branches):
            if idx < len(self.branches) - 1:
                with torch.no_grad():
                    parts.append(branch.extract_features(x))
            else:
                parts.append(branch.extract_features(x))
        return torch.cat(parts, dim=-1)

    def extract_multiscale_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Expose fixed-dim active stem/pre_ln plus concatenated penult features."""
        ms_last = self.branches[-1].extract_multiscale_features(x)
        return {
            "stem": ms_last["stem"],
            "pre_ln": ms_last["pre_ln"],
            "penult": self.extract_super_features(x),
        }

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.extract_super_features(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sup = self.extract_super_features(x)
        return self.classifier(sup), sup


class DERLearner:
    """DER incremental learner freezing historical backbones and training an auxiliary-guided new branch."""

    def __init__(
        self,
        epochs: int = 35,
        batch_size: int = 128,
        lr: float = 8e-4,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
        aux_weight: float = 0.5,
    ) -> None:
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.grad_clip = float(grad_clip)
        self.aux_weight = float(aux_weight)

    def evolve(
        self,
        model: BaseBackbone,
        replay_buffer: ReplayBuffer,
        new_samples: np.ndarray | torch.Tensor,
        new_pseudo_labels: np.ndarray | torch.Tensor,
        model_cfg: dict[str, Any],
        device: str | torch.device = "cuda",
        seed: int = 42,
    ) -> tuple[DERWrapperBackbone, EvolutionReport]:
        dev = torch.device(device)
        wrapper = model if isinstance(model, DERWrapperBackbone) else DERWrapperBackbone(model)
        x_new = torch.as_tensor(new_samples, dtype=torch.float32)
        y_local = torch.as_tensor(new_pseudo_labels, dtype=torch.long).reshape(-1)
        if len(x_new) == 0:
            raise ValueError("new_samples must not be empty")
        if int(y_local.min().item()) < 0:
            raise ValueError("Noise labels (-1) must be removed before evolve()")

        num_old = int(wrapper.num_classes)
        num_new = int(y_local.max().item()) + 1

        new_branch = build_backbone(copy.deepcopy(model_cfg), num_classes=num_new + 1).to(dev)
        # Warm-start new branch weights from latest branch when shapes match
        prev_branch = wrapper.branches[-1]
        prev_sd = prev_branch.state_dict()
        new_sd = new_branch.state_dict()
        compatible = {k: v for k, v in prev_sd.items() if k in new_sd and v.shape == new_sd[k].shape and "classifier" not in k}
        new_sd.update(compatible)
        new_branch.load_state_dict(new_sd)

        wrapper.add_branch(new_branch, num_new_classes=num_new)
        wrapper.to(dev)

        aux_head = nn.Linear(int(new_branch.feature_dim), num_new + 1).to(dev)
        y_global_new = y_local + num_old
        old_x, old_y = replay_buffer.as_tensors()
        x_all = torch.cat([old_x, x_new], dim=0) if len(old_x) > 0 else x_new
        y_all = torch.cat([old_y, y_global_new], dim=0) if len(old_y) > 0 else y_global_new
        loader = DataLoader(
            TensorDataset(x_all, y_all),
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(int(seed)),
        )

        train_params = list(wrapper.branches[-1].parameters()) + list(wrapper.classifier.parameters()) + list(aux_head.parameters())
        optimizer = torch.optim.AdamW(train_params, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, self.epochs), eta_min=1e-6)

        history: list[dict[str, float]] = []
        t0 = time.time()
        for epoch in range(1, self.epochs + 1):
            for idx, b in enumerate(wrapper.branches):
                b.train() if idx == len(wrapper.branches) - 1 else b.eval()
            wrapper.classifier.train()
            aux_head.train()
            ep_tot, ep_ce, ep_aux, steps = 0.0, 0.0, 0.0, 0
            for xb, yb in loader:
                xb = xb.to(dev, non_blocking=True)
                yb = yb.to(dev, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits, sup = wrapper(xb)
                ce_loss = F.cross_entropy(logits, yb)

                # Auxiliary target collapses all old classes into class 0 and new classes to 1..K_new
                aux_y = torch.where(yb >= num_old, yb - num_old + 1, torch.zeros_like(yb))
                last_dim = int(wrapper.branches[-1].feature_dim)
                aux_loss = F.cross_entropy(aux_head(sup[:, -last_dim:]), aux_y)
                loss = ce_loss + self.aux_weight * aux_loss
                loss.backward()
                if self.grad_clip > 0.0:
                    nn.utils.clip_grad_norm_(train_params, max_norm=self.grad_clip)
                optimizer.step()
                ep_tot += float(loss.item())
                ep_ce += float(ce_loss.item())
                ep_aux += float(aux_loss.item())
                steps += 1
            scheduler.step()
            denom = float(max(1, steps))
            history.append(
                {
                    "epoch": float(epoch),
                    "loss": round(ep_tot / denom, 5),
                    "ce_loss": round(ep_ce / denom, 5),
                    "feat_kd": round(ep_aux / denom, 5),
                    "logit_kd": 0.0,
                }
            )

        last = history[-1]
        return wrapper, EvolutionReport(
            num_old_classes=num_old,
            num_new_classes=num_new,
            epochs=self.epochs,
            final_loss=last["loss"],
            final_ce_loss=last["ce_loss"],
            final_feat_kd=last["feat_kd"],
            final_logit_kd=0.0,
            elapsed_sec=round(time.time() - t0, 2),
            history=history,
        )
