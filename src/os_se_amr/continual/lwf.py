"""Anti-forgetting incremental learner with Class-Balanced RF-Aug Replay, Old-Only Dual-LwF & Per-Head BiC."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from os_se_amr.continual.replay import ReplayBuffer
from os_se_amr.models.base import BaseBackbone


@dataclass
class EvolutionReport:
    """Summary statistics from one incremental evolution update."""

    num_old_classes: int
    num_new_classes: int
    epochs: int
    final_loss: float
    final_ce_loss: float
    final_feat_kd: float
    final_logit_kd: float
    elapsed_sec: float
    history: list[dict[str, float]]
    bic_gammas: list[float] = field(default_factory=list)
    wa_gamma: float = 1.0


def apply_weight_alignment(backbone: BaseBackbone, num_old_classes: int) -> float:
    """Apply Weight Aligning (Zhao et al., CVPR 2020) across all old vs newly added classifier heads."""
    if hasattr(backbone.classifier, "fc"):
        weight = backbone.classifier.fc.weight
        bias = backbone.classifier.fc.bias
    elif hasattr(backbone.classifier, "weight"):
        weight = backbone.classifier.weight
        bias = getattr(backbone.classifier, "bias", None)
    else:
        return 1.0

    total = int(weight.shape[0])
    if num_old_classes <= 0 or num_old_classes >= total:
        return 1.0

    with torch.no_grad():
        old_norm = float(weight[:num_old_classes].norm(dim=1).mean().clamp_min(1e-8).item())
        new_norm = float(weight[num_old_classes:].norm(dim=1).mean().clamp_min(1e-8).item())
        gamma = old_norm / max(new_norm, 1e-8)
        weight[num_old_classes:].mul_(gamma)
        if bias is not None:
            bias[num_old_classes:].mul_(gamma)
    return float(gamma)


def rf_phase_aligned_replay_ss(
    x: torch.Tensor,
    y: torch.Tensor,
    min_class_id: int,
    gen: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply phase-aligned half-frame SS within historical discovered classes (cid >= min_class_id)."""
    out = x.clone()
    frame_len = out.shape[-1]
    cut = frame_len // 2
    if frame_len < 4 or cut <= 0:
        return out

    for cls_id in torch.unique(y).tolist():
        if int(cls_id) < int(min_class_id):
            continue
        idx = torch.where(y == cls_id)[0]
        if len(idx) < 2:
            continue
        half = max(1, len(idx) // 2)
        perm = idx[torch.randperm(len(idx), device=x.device, generator=gen)]
        a_idx = idx[:half]
        b_idx = perm[:half]
        theta_a = torch.atan2(out[a_idx, 1, cut - 1], out[a_idx, 0, cut - 1])
        theta_b = torch.atan2(out[b_idx, 1, cut], out[b_idx, 0, cut])
        d_theta = (theta_a - theta_b).unsqueeze(-1)
        cos_d = torch.cos(d_theta)
        sin_d = torch.sin(d_theta)
        r_i = out[b_idx, 0, cut:].clone()
        r_q = out[b_idx, 1, cut:].clone()
        out[a_idx, 0, cut:] = cos_d * r_i - sin_d * r_q
        out[a_idx, 1, cut:] = sin_d * r_i + cos_d * r_q
    return out


def rf_augment_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    gen: torch.Generator | None = None,
    use_ss_aug: bool = True,
) -> torch.Tensor:
    """Apply exact electromagnetic signal invariances: 90-deg carrier rotation + optional intra-class SS."""
    out = x.clone()
    bsz = len(out)
    if bsz == 0:
        return out

    # 1. Discrete 90-degree carrier rotation on 50% of samples (preserves QAM constellation axes)
    rot_mask = torch.rand(bsz, device=x.device, generator=gen) < 0.5
    rot_k = torch.randint(0, 4, (bsz,), device=x.device, generator=gen)
    i_ch, q_ch = out[:, 0].clone(), out[:, 1].clone()
    for kv, (ci, si) in enumerate([(1, 0), (0, 1), (-1, 0), (0, -1)]):
        mk = rot_mask & (rot_k == kv)
        if mk.any():
            out[mk, 0] = ci * i_ch[mk] - si * q_ch[mk]
            out[mk, 1] = si * i_ch[mk] + ci * q_ch[mk]

    # 2. Symbol-aligned Segment Substitution at cut=64 (half frame) within each class
    if use_ss_aug:
        cut = out.shape[-1] // 2
        for cls_id in torch.unique(y).tolist():
            idx = torch.where(y == cls_id)[0]
            if len(idx) >= 2:
                half = len(idx) // 2
                perm = idx[torch.randperm(len(idx), device=x.device, generator=gen)]
                out[idx[:half], :, cut:] = out[perm[:half], :, cut:]

    return out


@torch.no_grad()
def calibrate_per_head_bias(
    backbone: BaseBackbone,
    val_known_loader: DataLoader,
    num_base_classes: int,
    max_steal_per_head: float = 0.042,
    device: str | torch.device = "cuda",
) -> np.ndarray:
    """Calibrate biases of novel heads (>= num_base_classes) strictly using Session 0 Val Known samples.

    Only penalizes novel heads that falsely steal more than `max_steal_per_head` fraction of
    Val Known samples from base heads; orthogonal novel heads receive zero shift.
    """
    dev = torch.device(device)
    backbone.eval()
    logits_list: list[np.ndarray] = []
    for batch in val_known_loader:
        xb = batch[0].to(dev, non_blocking=True)
        lg, _ = backbone(xb)
        logits_list.append(lg.cpu().numpy())

    val_logits = np.concatenate(logits_list, axis=0)
    num_total = val_logits.shape[1]
    if num_total <= num_base_classes:
        return np.empty(0, dtype=np.float64)

    old_max = val_logits[:, :num_base_classes].max(axis=1)
    gammas = np.zeros(num_total - num_base_classes, dtype=np.float64)
    for j, head in enumerate(range(num_base_classes, num_total)):
        diff = val_logits[:, head] - old_max
        steal_rate = float((diff > 0.0).mean())
        if steal_rate > max_steal_per_head:
            q = float(np.quantile(diff, 1.0 - max_steal_per_head))
            gammas[j] = -max(0.0, q)
    return gammas


class DualLwFLearner:
    """Incremental learner combining Class-Balanced RF-Aug Replay, Old-Only Dual-LwF & Per-Head BiC.

    Setting `use_replay=False, beta_feat=0.0, beta_logit=0.0` runs Naive Fine-Tune cleanly.
    """

    def __init__(
        self,
        epochs: int = 35,
        steps_per_epoch: int = 30,
        samples_per_class_per_step: int = 24,
        batch_size_new: int = 128,
        batch_size_old: int = 128,
        lr: float = 8e-4,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
        beta_feat: float = 0.05,
        beta_logit: float = 0.5,
        distill_temp: float = 2.0,
        use_replay: bool = True,
        use_rf_aug: bool = True,
        use_ss_aug: bool = True,
        bic_max_steal: float | None = 0.042,
        use_weight_align: bool = False,
        replay_ss_discovered: bool = False,
        use_amp: bool = False,
    ) -> None:
        self.epochs = int(epochs)
        self.steps_per_epoch = int(steps_per_epoch)
        self.samples_per_class_per_step = int(samples_per_class_per_step)
        self.batch_size_new = int(batch_size_new)
        self.batch_size_old = int(batch_size_old)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.grad_clip = float(grad_clip)
        self.beta_feat = float(beta_feat)
        self.beta_logit = float(beta_logit)
        self.distill_temp = float(distill_temp)
        self.use_replay = bool(use_replay)
        self.use_rf_aug = bool(use_rf_aug)
        self.use_ss_aug = bool(use_ss_aug)
        self.bic_max_steal = float(bic_max_steal) if bic_max_steal is not None else None
        self.use_weight_align = bool(use_weight_align)
        self.replay_ss_discovered = bool(replay_ss_discovered)
        self.use_amp = bool(use_amp)

    def _build_init_weights(
        self,
        backbone: BaseBackbone,
        num_old_classes: int,
        prototypes_64d: np.ndarray | torch.Tensor | None,
    ) -> torch.Tensor | None:
        if prototypes_64d is None:
            return None
        proto_t = torch.as_tensor(prototypes_64d, dtype=torch.float32)
        if proto_t.ndim != 2 or proto_t.shape[1] != backbone.feature_dim:
            raise ValueError(
                f"Expected prototypes shape (K, {backbone.feature_dim}), got {tuple(proto_t.shape)}"
            )
        proto_unit = F.normalize(proto_t, p=2, dim=-1)

        if hasattr(backbone.classifier, "fc"):
            old_w = backbone.classifier.fc.weight.detach()[:num_old_classes]
            mean_norm = float(old_w.norm(dim=1).mean().clamp_min(1e-4).item())
        elif hasattr(backbone.classifier, "weight"):
            old_w = backbone.classifier.weight.detach()[:num_old_classes]
            mean_norm = float(old_w.norm(dim=1).mean().clamp_min(1e-4).item())
        else:
            mean_norm = 1.0

        return proto_unit * mean_norm

    def evolve(
        self,
        backbone: BaseBackbone,
        replay_buffer: ReplayBuffer,
        new_samples: np.ndarray | torch.Tensor,
        new_pseudo_labels: np.ndarray | torch.Tensor,
        prototypes_64d: np.ndarray | torch.Tensor | None = None,
        sample_weights: np.ndarray | torch.Tensor | None = None,
        val_known_loader: DataLoader | None = None,
        num_base_classes: int | None = None,
        device: str | torch.device = "cuda",
        seed: int = 42,
    ) -> EvolutionReport:
        """Expand classifier for discovered clusters and fine-tune with anti-forgetting losses.

        `new_pseudo_labels` must use local cluster indices 0 .. K_hat - 1;
        this method automatically shifts them by `num_old_classes`.
        """
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
        num_new = int(len(prototypes_64d)) if prototypes_64d is not None else (max_lbl + 1)
        if max_lbl >= num_new:
            raise ValueError(f"Cluster label {max_lbl} exceeds num_new_classes={num_new}")

        if sample_weights is None:
            w_new = torch.ones(len(x_new), dtype=torch.float32)
        else:
            w_new = torch.as_tensor(sample_weights, dtype=torch.float32).reshape(-1)
            if len(w_new) != len(x_new):
                raise ValueError("sample_weights length mismatch")

        teacher = copy.deepcopy(backbone).to(dev).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        init_w = self._build_init_weights(backbone, num_old, prototypes_64d)
        backbone.expand_classifier(num_new, init_weights=init_w)
        backbone.to(dev)

        y_global_new = y_local + num_old
        old_x_all, old_y_all = replay_buffer.as_tensors()
        has_replay = self.use_replay and len(old_x_all) > 0

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

        if has_replay:
            # Class-balanced pools on target device
            pools: dict[int, tuple[torch.Tensor, torch.Tensor | None]] = {}
            for cid in replay_buffer.class_ids:
                mask = old_y_all == cid
                pools[cid] = (old_x_all[mask].to(dev), None)
            for k in range(num_new):
                cid = num_old + k
                mask = y_global_new == cid
                if mask.any():
                    pools[cid] = (x_new[mask].to(dev), w_new[mask].to(dev))

            old_cids = [c for c in replay_buffer.class_ids if c in pools]
            new_cids = [num_old + k for k in range(num_new) if (num_old + k) in pools]
            gen_dev = torch.Generator(device=dev).manual_seed(int(seed))
            k_per_cls = self.samples_per_class_per_step

            for epoch in range(1, self.epochs + 1):
                backbone.train()
                ep_total, ep_ce, ep_feat, ep_logit = 0.0, 0.0, 0.0, 0.0

                for _ in range(self.steps_per_epoch):
                    xo_list, yo_list, wo_list = [], [], []
                    for cid in old_cids:
                        px, _ = pools[cid]
                        idx = torch.randint(0, len(px), (k_per_cls,), generator=gen_dev, device=dev)
                        xo_list.append(px[idx])
                        yo_list.append(torch.full((k_per_cls,), cid, dtype=torch.long, device=dev))
                        wo_list.append(torch.ones(k_per_cls, dtype=torch.float32, device=dev))

                    xn_list, yn_list, wn_list = [], [], []
                    for cid in new_cids:
                        px, pw = pools[cid]
                        idx = torch.randint(0, len(px), (k_per_cls,), generator=gen_dev, device=dev)
                        xn_list.append(px[idx])
                        yn_list.append(torch.full((k_per_cls,), cid, dtype=torch.long, device=dev))
                        wn_list.append(
                            pw[idx] if pw is not None else torch.ones(k_per_cls, dtype=torch.float32, device=dev)
                        )

                    xo = torch.cat(xo_list, dim=0)
                    yo = torch.cat(yo_list, dim=0)
                    wo = torch.cat(wo_list, dim=0)
                    xn = torch.cat(xn_list, dim=0)
                    yn = torch.cat(yn_list, dim=0)
                    wn = torch.cat(wn_list, dim=0)

                    if self.use_rf_aug:
                        xo = rf_augment_batch(xo, yo, gen=gen_dev, use_ss_aug=self.use_ss_aug)
                        xn = rf_augment_batch(xn, yn, gen=gen_dev, use_ss_aug=self.use_ss_aug)
                    if self.replay_ss_discovered and num_base_classes is not None:
                        xo = rf_phase_aligned_replay_ss(xo, yo, min_class_id=int(num_base_classes), gen=gen_dev)

                    x_comb = torch.cat([xo, xn], dim=0)
                    y_comb = torch.cat([yo, yn], dim=0)
                    w_comb = torch.cat([wo, wn], dim=0)

                    amp_enabled = self.use_amp and dev.type == "cuda"
                    amp_dtype = torch.bfloat16 if (amp_enabled and torch.cuda.is_bf16_supported()) else torch.float16
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                        s_logits, s_feats = backbone(x_comb)

                        ce_vec = F.cross_entropy(s_logits, y_comb, reduction="none")
                        ce_loss = (ce_vec * w_comb).sum() / w_comb.sum().clamp_min(1e-8)

                        n_old_b = len(xo)
                        feat_kd = x_comb.new_zeros(())
                        logit_kd = x_comb.new_zeros(())
                        if self.beta_feat > 0.0 or self.beta_logit > 0.0:
                            with torch.no_grad():
                                t_logits_old, t_feats_old = teacher(xo)
                            if self.beta_feat > 0.0:
                                feat_kd = F.mse_loss(s_feats[:n_old_b], t_feats_old)
                            if self.beta_logit > 0.0:
                                p_t = F.softmax(t_logits_old / temp, dim=-1)
                                log_p_s = F.log_softmax(s_logits[:n_old_b, :num_old] / temp, dim=-1)
                                logit_kd = F.kl_div(log_p_s, p_t, reduction="batchmean") * (temp * temp)

                        loss = ce_loss + self.beta_feat * feat_kd + self.beta_logit * logit_kd
                    loss.backward()
                    if self.grad_clip > 0.0:
                        nn.utils.clip_grad_norm_(backbone.parameters(), max_norm=self.grad_clip)
                    optimizer.step()

                    ep_total += float(loss.item())
                    ep_ce += float(ce_loss.item())
                    ep_feat += float(feat_kd.item())
                    ep_logit += float(logit_kd.item())

                scheduler.step()
                denom = float(max(1, self.steps_per_epoch))
                history.append(
                    {
                        "epoch": float(epoch),
                        "loss": round(ep_total / denom, 5),
                        "ce_loss": round(ep_ce / denom, 5),
                        "feat_kd": round(ep_feat / denom, 5),
                        "logit_kd": round(ep_logit / denom, 5),
                    }
                )
        else:
            # Naive Fine-Tune path (new-class samples only, no replay or KD)
            new_ds = TensorDataset(x_new, y_global_new, w_new)
            gen_cpu = torch.Generator().manual_seed(int(seed))
            new_loader = DataLoader(new_ds, batch_size=self.batch_size_new, shuffle=True, generator=gen_cpu)

            for epoch in range(1, self.epochs + 1):
                backbone.train()
                ep_total, ep_ce, steps = 0.0, 0.0, 0
                for xb, yb, wb in new_loader:
                    xb = xb.to(dev, non_blocking=True)
                    yb = yb.to(dev, non_blocking=True)
                    wb = wb.to(dev, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)
                    s_logits, _ = backbone(xb)
                    ce_vec = F.cross_entropy(s_logits, yb, reduction="none")
                    loss = (ce_vec * wb).sum() / wb.sum().clamp_min(1e-8)
                    loss.backward()
                    if self.grad_clip > 0.0:
                        nn.utils.clip_grad_norm_(backbone.parameters(), max_norm=self.grad_clip)
                    optimizer.step()

                    ep_total += float(loss.item())
                    ep_ce += float(loss.item())
                    steps += 1

                scheduler.step()
                denom = float(max(1, steps))
                history.append(
                    {
                        "epoch": float(epoch),
                        "loss": round(ep_total / denom, 5),
                        "ce_loss": round(ep_ce / denom, 5),
                        "feat_kd": 0.0,
                        "logit_kd": 0.0,
                    }
                )

        # Optional Multi-Session Weight Aligning across ALL old heads vs current new heads
        wa_gamma = 1.0
        if has_replay and self.use_weight_align:
            wa_gamma = apply_weight_alignment(backbone, num_old)

        # Post-update Per-Head BiC strictly on Session-0 Val Known set
        bic_gammas: list[float] = []
        if has_replay and val_known_loader is not None and self.bic_max_steal is not None:
            base_n = num_old if num_base_classes is None else int(num_base_classes)
            gammas = calibrate_per_head_bias(
                backbone=backbone,
                val_known_loader=val_known_loader,
                num_base_classes=base_n,
                max_steal_per_head=self.bic_max_steal,
                device=dev,
            )
            if len(gammas) > 0 and hasattr(backbone.classifier, "fc") and backbone.classifier.fc.bias is not None:
                with torch.no_grad():
                    backbone.classifier.fc.bias.data[base_n:] += torch.from_numpy(gammas).to(
                        device=dev, dtype=backbone.classifier.fc.bias.dtype
                    )
            bic_gammas = [round(float(g), 4) for g in gammas]

        last = history[-1]
        return EvolutionReport(
            num_old_classes=num_old,
            num_new_classes=num_new,
            epochs=self.epochs,
            final_loss=last["loss"],
            final_ce_loss=last["ce_loss"],
            final_feat_kd=last["feat_kd"],
            final_logit_kd=last["logit_kd"],
            elapsed_sec=round(time.time() - t0, 2),
            history=history,
            bic_gammas=bic_gammas,
            wa_gamma=round(float(wa_gamma), 4),
        )
