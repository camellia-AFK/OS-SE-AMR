"""Differentiable electromagnetic signal representations from raw I/Q frames."""

from __future__ import annotations

import math

import numpy as np
import torch


def normalize_iq_rms(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize complex baseband I/Q frames (..., 2, L) to unit RMS envelope energy.

    Preserves exact instantaneous phase and relative symbol amplitude ratios
    (critical for QAM16/QAM64/PAM4 discrimination) while removing frame gain variance.
    """
    if x.shape[-2] != 2:
        raise ValueError(f"Expected channel dim == 2 for I/Q tensor, got shape {tuple(x.shape)}")
    power = x[..., 0, :] ** 2 + x[..., 1, :] ** 2
    rms = torch.sqrt(power.mean(dim=-1, keepdim=True) + eps).unsqueeze(-2)
    return x / rms


def iq_to_ap(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Convert raw I/Q frames (..., 2, L) to [Amplitude_norm, Phase_norm] in (..., 2, L).

    Amplitude is computed after per-frame RMS normalization; phase is normalized to [-1, 1].
    """
    x_norm = normalize_iq_rms(x, eps=eps)
    i_ch = x_norm[..., 0, :]
    q_ch = x_norm[..., 1, :]
    amp = torch.sqrt(i_ch**2 + q_ch**2 + eps)
    phase = torch.atan2(q_ch, i_ch) / math.pi
    return torch.stack((amp, phase), dim=-2)


def iq_to_iq_ap(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Convert raw I/Q frames (..., 2, L) to 4-channel [I_norm, Q_norm, A_norm, P_norm]."""
    x_norm = normalize_iq_rms(x, eps=eps)
    ap = iq_to_ap(x_norm, eps=eps)
    return torch.cat((x_norm, ap), dim=-2)


def iq_to_stft_mag(
    x: torch.Tensor,
    n_fft: int = 32,
    hop_length: int = 8,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute centered two-sided log-magnitude STFT spectrogram (B, 1, F, T) on x.device.

    Two-sided complex STFT preserves upper/lower sideband asymmetry (e.g., AM-SSB).
    """
    if x.ndim != 3 or x.shape[1] != 2:
        raise ValueError(f"Expected shape (B, 2, L), got {tuple(x.shape)}")

    x_norm = normalize_iq_rms(x, eps=eps)
    complex_sig = torch.complex(x_norm[:, 0, :], x_norm[:, 1, :])
    window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
    spec = torch.stft(
        complex_sig,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        onesided=False,
        return_complex=True,
    )
    spec = torch.fft.fftshift(spec, dim=-2)
    log_mag = torch.log1p(spec.abs() + eps)
    return log_mag.unsqueeze(1)


def extract_rf_physics_features(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute 10-d length-agnostic analytical RF cumulant, envelope, and spectral physics features."""
    if x.ndim != 3 or x.shape[1] != 2:
        raise ValueError(f"Expected shape (B, 2, L), got {tuple(x.shape)}")

    x_norm = normalize_iq_rms(x.to(dtype=torch.float32), eps=eps)
    i_ch = x_norm[:, 0, :]
    q_ch = x_norm[:, 1, :]
    z = torch.complex(i_ch, q_ch)
    frame_len = int(x_norm.shape[-1])

    p = i_ch * i_ch + q_ch * q_ch
    m20 = (z * z).mean(dim=-1)
    m21 = p.mean(dim=-1)
    z2 = z * z
    z4 = z2 * z2
    m40 = z4.mean(dim=-1)
    m41 = (z2 * z * torch.conj(z)).mean(dim=-1)
    m42 = (p * p).mean(dim=-1)
    m63 = (p * p * p).mean(dim=-1)

    abs_c20 = m20.abs()
    abs_c40 = (m40 - 3.0 * m20 * m20).abs()
    abs_c41 = (m41 - 3.0 * m20 * m21).abs()
    c42 = m42 - (abs_c20 * abs_c20) - 2.0 * (m21 * m21)
    c63 = (
        m63
        - 9.0 * m42 * m21
        + 12.0 * (abs_c20 * abs_c20) * m21
        + 12.0 * (m21 * m21 * m21)
        - 3.0 * (m40 * torch.conj(m20)).real
    )
    asinh_c63 = torch.asinh(c63)

    log_papr = torch.log(p.max(dim=-1).values / (m21 + eps) + eps)
    amp = torch.sqrt(p + eps)
    env_std = amp.std(dim=-1, unbiased=False)

    if frame_len >= 2:
        prod = z[:, 1:] * torch.conj(z[:, :-1])
        dphase = torch.atan2(prod.imag, prod.real) / math.pi
        inst_freq_std = dphase.std(dim=-1, unbiased=False)
    else:
        inst_freq_std = torch.zeros_like(m21)

    fft_c = torch.fft.fftshift(torch.fft.fft(z, dim=-1), dim=-1)
    psd = (fft_c.real * fft_c.real + fft_c.imag * fft_c.imag) / float(frame_len) + eps
    spec_flat = torch.exp(torch.log(psd).mean(dim=-1)) / (psd.mean(dim=-1) + eps)

    mid = max(1, frame_len // 2)
    p_neg = psd[:, :mid].mean(dim=-1)
    p_pos = psd[:, mid:].mean(dim=-1)
    sideband_ratio = torch.log((p_pos + eps) / (p_neg + eps))

    feats = torch.stack(
        [
            abs_c20,
            abs_c40,
            abs_c41,
            c42,
            asinh_c63,
            log_papr,
            env_std,
            inst_freq_std,
            spec_flat,
            sideband_ratio,
        ],
        dim=-1,
    )
    return torch.clamp(feats, -10.0, 10.0).to(dtype=torch.float32)


def apply_rf_channel_impairments(
    x: np.ndarray | torch.Tensor,
    cfo_norm: float = 0.0,
    phase_noise_std: float = 0.0,
    seed: int = 42,
) -> np.ndarray | torch.Tensor:
    """Apply normalized carrier frequency offset (cycles/sample) and Wiener phase noise to I/Q frames."""
    is_tensor = torch.is_tensor(x)
    arr = x.detach().cpu().numpy().astype(np.float32) if is_tensor else np.asarray(x, dtype=np.float32).copy()
    if arr.ndim != 3 or arr.shape[1] != 2:
        raise ValueError(f"Expected shape (N, 2, L), got {arr.shape}")
    if cfo_norm == 0.0 and phase_noise_std == 0.0:
        return x.clone() if is_tensor else arr

    n_samples, _, frame_len = arr.shape
    rng = np.random.default_rng(int(seed))
    t = np.arange(frame_len, dtype=np.float32)[None, :]
    theta = (2.0 * np.pi * float(cfo_norm)) * t
    if n_samples > 1:
        theta = np.repeat(theta, n_samples, axis=0)
    else:
        theta = theta.copy()

    if phase_noise_std > 0.0:
        steps = rng.normal(0.0, float(phase_noise_std), size=(n_samples, frame_len)).astype(np.float32)
        theta = theta + np.cumsum(steps, axis=1)

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    i_in = arr[:, 0, :]
    q_in = arr[:, 1, :]
    out = np.empty_like(arr)
    out[:, 0, :] = cos_t * i_in - sin_t * q_in
    out[:, 1, :] = sin_t * i_in + cos_t * q_in
    if is_tensor:
        return torch.from_numpy(out).to(device=x.device, dtype=x.dtype)
    return np.ascontiguousarray(out, dtype=np.float32)
