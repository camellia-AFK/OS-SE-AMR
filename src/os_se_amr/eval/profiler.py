"""Computational complexity, parameter count, memory footprint, and online latency profiler."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from os_se_amr.models.base import BaseBackbone, BaseSNREstimator
from os_se_amr.osr.mahalanobis import MahalanobisPrototypeScorer


def estimate_conv_linear_macs(model: nn.Module, sample_input: torch.Tensor) -> int:
    """Count exact forward MACs for Conv1d/Conv2d/Linear layers via a single dummy trace."""
    total_macs = 0
    handles = []

    def _hook_conv(mod: nn.Module, _inp: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        nonlocal total_macs
        if isinstance(mod, nn.Conv1d):
            out_elems = int(out.shape[1] * out.shape[2])
            k_ops = (int(mod.in_channels) // int(mod.groups)) * int(mod.kernel_size[0])
            total_macs += out_elems * k_ops
        elif isinstance(mod, nn.Conv2d):
            out_elems = int(out.shape[1] * out.shape[2] * out.shape[3])
            k_ops = (int(mod.in_channels) // int(mod.groups)) * int(mod.kernel_size[0]) * int(mod.kernel_size[1])
            total_macs += out_elems * k_ops

    def _hook_linear(mod: nn.Linear, _inp: tuple[torch.Tensor, ...], _out: torch.Tensor) -> None:
        nonlocal total_macs
        total_macs += int(mod.in_features) * int(mod.out_features)

    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            handles.append(m.register_forward_hook(_hook_conv))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(_hook_linear))

    prev_mode = model.training
    model.eval()
    with torch.no_grad():
        model(sample_input[:1])
    model.train(prev_mode)
    for h in handles:
        h.remove()
    return int(total_macs)


def _measure_latency_ms(fn: Any, device: torch.device, warmup: int = 5, repeats: int = 30) -> float:
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return float((time.perf_counter() - t0) * 1000.0 / float(repeats))


def profile_system_overhead(
    model: BaseBackbone,
    scorer: MahalanobisPrototypeScorer | None = None,
    snr_estimator: BaseSNREstimator | None = None,
    frame_len: int = 128,
    replay_exemplars_per_class: int = 100,
    num_classes: int | None = None,
    batch_size: int = 64,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Measure model params, forward MACs/FLOPs, replay memory, and online per-frame latency."""
    dev = torch.device(device)
    model_eval = model.to(dev).eval()
    c_total = int(model_eval.num_classes if num_classes is None else num_classes)
    x_dummy = torch.randn(batch_size, 2, int(frame_len), device=dev, dtype=torch.float32)

    params_backbone = sum(p.numel() for p in model_eval.parameters())
    params_snr = sum(p.numel() for p in snr_estimator.parameters()) if isinstance(snr_estimator, nn.Module) else 0
    macs_backbone = estimate_conv_linear_macs(model_eval, x_dummy[:1])
    flops_backbone = 2 * macs_backbone

    @torch.no_grad()
    def _step_backbone() -> None:
        model_eval.extract_multiscale_features(x_dummy)

    @torch.no_grad()
    def _step_full_gate() -> None:
        ms = model_eval.extract_multiscale_features(x_dummy)
        ms_np = {k: v.cpu().numpy() for k, v in ms.items()}
        if scorer is not None and scorer._fitted:
            # Handle physics layer lazily if requested
            if "physics" in scorer.layers and "physics" not in ms_np:
                from os_se_amr.data.transforms import extract_rf_physics_features

                ms_np["physics"] = extract_rf_physics_features(x_dummy).cpu().numpy()
            scorer.score_multiscale(ms_np)
        if snr_estimator is not None:
            if isinstance(snr_estimator, nn.Module):
                snr_estimator.to(dev).eval()
            snr_estimator.predict_snr(x_dummy)

    lat_batch_backbone_ms = _measure_latency_ms(_step_backbone, device=dev)
    lat_batch_gate_ms = _measure_latency_ms(_step_full_gate, device=dev)

    replay_bytes = c_total * int(replay_exemplars_per_class) * 2 * int(frame_len) * 4
    scorer_bytes = 0
    if scorer is not None and scorer._fitted:
        for arr in scorer.means.values():
            scorer_bytes += arr.nbytes
        for arr in scorer.precisions.values():
            scorer_bytes += arr.nbytes

    return {
        "device": str(dev),
        "frame_len": int(frame_len),
        "num_classes": c_total,
        "batch_size": int(batch_size),
        "params_backbone": int(params_backbone),
        "params_snr_estimator": int(params_snr),
        "params_total_k": round((params_backbone + params_snr) / 1e3, 2),
        "macs_per_frame_m": round(macs_backbone / 1e6, 3),
        "flops_per_frame_m": round(flops_backbone / 1e6, 3),
        "backbone_batch_ms": round(lat_batch_backbone_ms, 3),
        "backbone_per_frame_ms": round(lat_batch_backbone_ms / float(batch_size), 4),
        "full_online_gate_batch_ms": round(lat_batch_gate_ms, 3),
        "full_online_gate_per_frame_ms": round(lat_batch_gate_ms / float(batch_size), 4),
        "replay_memory_mb": round(replay_bytes / (1024.0 * 1024.0), 3),
        "scorer_memory_mb": round(scorer_bytes / (1024.0 * 1024.0), 3),
    }
