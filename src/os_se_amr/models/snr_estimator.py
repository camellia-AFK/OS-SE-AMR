"""Blind SNR estimators: M2M4 physical moment baseline and modulation-agnostic AuxSNRRegressor."""

from __future__ import annotations

import logging
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from os_se_amr.data.transforms import normalize_iq_rms
from os_se_amr.models.base import BaseSNREstimator


class M2M4SNREstimator(BaseSNREstimator):
    """Training-free higher-order moment SNR estimator with optional affine calibration."""

    def __init__(
        self,
        signal_kurtosis: float = 1.0,
        eps: float = 1e-8,
        clip_min_db: float = -20.0,
        clip_max_db: float = 25.0,
    ) -> None:
        self.signal_kurtosis = float(signal_kurtosis)
        self.eps = float(eps)
        self.clip_min_db = float(clip_min_db)
        self.clip_max_db = float(clip_max_db)
        self.scale: float = 1.0
        self.bias: float = 0.0

    @torch.no_grad()
    def raw_predict_snr(self, x_iq: torch.Tensor) -> torch.Tensor:
        """Compute uncalibrated M2M4 SNR estimate in dB from complex baseband I/Q (B, 2, L)."""
        if x_iq.ndim != 3 or x_iq.shape[1] != 2:
            raise ValueError(f"Expected shape (B, 2, L), got {tuple(x_iq.shape)}")
        x = x_iq.to(dtype=torch.float32)
        p = x[:, 0, :] ** 2 + x[:, 1, :] ** 2
        m2 = p.mean(dim=-1)
        m4 = (p * p).mean(dim=-1)
        denom = max(2.0 - self.signal_kurtosis, 1e-6)
        s_hat = torch.sqrt(torch.clamp_min((2.0 * m2 * m2 - m4) / denom, 0.0))
        n_hat = torch.clamp_min(m2 - s_hat, self.eps)
        snr_db = 10.0 * torch.log10(torch.clamp_min(s_hat, self.eps) / n_hat)
        return torch.clamp(snr_db, self.clip_min_db, self.clip_max_db)

    def fit_calibration(
        self,
        x_val_iq: np.ndarray | torch.Tensor,
        y_val_snr: np.ndarray | torch.Tensor,
        batch_size: int = 1024,
    ) -> None:
        """Fit linear affine bias correction strictly on Session 0 Val Known frames."""
        x_t = torch.as_tensor(x_val_iq, dtype=torch.float32)
        y_np = np.asarray(y_val_snr, dtype=np.float64).reshape(-1)
        if len(x_t) == 0 or len(x_t) != len(y_np):
            raise ValueError("x_val_iq and y_val_snr must be non-empty and have matching lengths")

        raw_parts = [self.raw_predict_snr(x_t[i : i + batch_size]).cpu().numpy() for i in range(0, len(x_t), batch_size)]
        raw = np.concatenate(raw_parts, axis=0).astype(np.float64)
        var_raw = float(np.var(raw))
        if var_raw > 1e-8:
            cov = float(np.mean((raw - raw.mean()) * (y_np - y_np.mean())))
            self.scale = cov / var_raw
            self.bias = float(y_np.mean() - self.scale * raw.mean())
        else:
            self.scale = 1.0
            self.bias = float(y_np.mean() - raw.mean())

    @torch.no_grad()
    def predict_snr(self, x_iq: torch.Tensor) -> torch.Tensor:
        raw = self.raw_predict_snr(x_iq)
        return torch.clamp(self.scale * raw + self.bias, self.clip_min_db, self.clip_max_db)

    def state_dict(self) -> dict[str, Any]:
        return {
            "signal_kurtosis": self.signal_kurtosis,
            "eps": self.eps,
            "clip_min_db": self.clip_min_db,
            "clip_max_db": self.clip_max_db,
            "scale": self.scale,
            "bias": self.bias,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.signal_kurtosis = float(state.get("signal_kurtosis", 1.0))
        self.eps = float(state.get("eps", 1e-8))
        self.clip_min_db = float(state.get("clip_min_db", -20.0))
        self.clip_max_db = float(state.get("clip_max_db", 25.0))
        self.scale = float(state.get("scale", 1.0))
        self.bias = float(state.get("bias", 0.0))


class AuxSNRRegressor(nn.Module, BaseSNREstimator):
    """Lightweight modulation-agnostic Time-Frequency 1D-CNN + physical stats SNR regressor."""

    def __init__(self) -> None:
        super().__init__()
        self.time_conv = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )
        self.freq_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )
        # 64 (time mean+std) + 64 (freq mean+std) + 8 (physical stats) = 136
        self.mlp = nn.Sequential(
            nn.Linear(136, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x_iq: torch.Tensor) -> torch.Tensor:
        """Predict SNR (dB) shape (B,) from raw I/Q shape (B, 2, L)."""
        if x_iq.ndim != 3 or x_iq.shape[1] != 2:
            raise ValueError(f"Expected shape (B, 2, L), got {tuple(x_iq.shape)}")

        x_norm = normalize_iq_rms(x_iq.to(dtype=torch.float32))
        i_ch = x_norm[:, 0, :]
        q_ch = x_norm[:, 1, :]
        amp = torch.sqrt(i_ch * i_ch + q_ch * q_ch + 1e-8)

        z = torch.complex(i_ch, q_ch)
        prod = z[:, 1:] * torch.conj(z[:, :-1])
        safe_real = torch.where(prod.abs() > 1e-8, prod.real, prod.real.new_full(prod.real.shape, 1e-8))
        dphase = torch.zeros_like(i_ch)
        dphase[:, 1:] = torch.atan2(prod.imag, safe_real) / math.pi

        t_in = torch.stack([i_ch, q_ch, amp, dphase], dim=1)  # (B, 4, L)

        frame_len = int(x_norm.shape[-1])
        fft_c = torch.fft.fftshift(torch.fft.fft(z, dim=-1), dim=-1)
        psd = (fft_c.real * fft_c.real + fft_c.imag * fft_c.imag) / float(frame_len)
        f_in = torch.log1p(psd).unsqueeze(1)  # (B, 1, L)

        # Modulation-agnostic physical features (8 dims)
        ac_feats = [
            torch.abs((z[:, k:] * torch.conj(z[:, :-k])).mean(dim=-1)) for k in (1, 2, 3, 4)
        ]
        psd_safe = psd + 1e-8
        spec_flat = torch.exp(torch.log(psd_safe).mean(dim=-1)) / (psd.mean(dim=-1) + 1e-8)
        p_f = psd_safe / psd_safe.sum(dim=-1, keepdim=True)
        spec_ent = -(p_f * torch.log(p_f)).sum(dim=-1) / math.log(float(frame_len))

        p_t = i_ch * i_ch + q_ch * q_ch
        m2 = p_t.mean(dim=-1)
        m4 = (p_t * p_t).mean(dim=-1)
        kurt = m4 / torch.clamp_min(m2 * m2, 1e-8)

        q_edge = max(1, frame_len // 8)
        hf_ratio = (psd[:, :q_edge].sum(dim=-1) + psd[:, -q_edge:].sum(dim=-1)) / torch.clamp_min(
            psd.sum(dim=-1), 1e-8
        )
        phys = torch.stack([*ac_feats, spec_flat, spec_ent, kurt, hf_ratio], dim=-1)  # (B, 8)

        t_h = self.time_conv(t_in)
        t_feat = torch.cat([t_h.mean(dim=-1), t_h.std(dim=-1, unbiased=False)], dim=-1)
        f_h = self.freq_conv(f_in)
        f_feat = torch.cat([f_h.mean(dim=-1), f_h.std(dim=-1, unbiased=False)], dim=-1)

        comb = torch.cat([t_feat, f_feat, phys], dim=-1)
        return self.mlp(comb).squeeze(-1)

    @torch.no_grad()
    def predict_snr(self, x_iq: torch.Tensor) -> torch.Tensor:
        dev = next(self.parameters()).device
        was_training = self.training
        self.eval()
        out = self.forward(x_iq.to(dev, non_blocking=True)).to(device=x_iq.device)
        if was_training:
            self.train()
        return out


def train_aux_snr_regressor(
    model: AuxSNRRegressor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str | torch.device = "cuda",
    epochs: int = 30,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    patience: int = 8,
    ckpt_path: str | Path | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Train AuxSNRRegressor strictly on Session 0 Known classes using Smooth-L1 loss."""
    dev = torch.device(device)
    model.to(dev)
    criterion = nn.SmoothL1Loss(beta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=1e-5)

    best_val_mae = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    no_improve = 0
    history: list[dict[str, float]] = []
    t0 = time.time()

    for epoch in range(1, int(epochs) + 1):
        model.train()
        tr_loss, tr_abs, tr_n = 0.0, 0.0, 0
        for batch in train_loader:
            xb = batch[0].to(dev, non_blocking=True)
            sb = batch[2].to(dev, dtype=torch.float32, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, sb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            bs = len(xb)
            tr_loss += float(loss.item()) * bs
            tr_abs += float((pred.detach() - sb).abs().sum().item())
            tr_n += bs
        scheduler.step()

        model.eval()
        val_abs, val_sq, val_n = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                xb = batch[0].to(dev, non_blocking=True)
                sb = batch[2].to(dev, dtype=torch.float32, non_blocking=True)
                pred = model(xb)
                err = pred - sb
                val_abs += float(err.abs().sum().item())
                val_sq += float((err * err).sum().item())
                val_n += len(xb)

        train_mae = tr_abs / max(1, tr_n)
        val_mae = val_abs / max(1, val_n)
        val_rmse = math.sqrt(val_sq / max(1, val_n))
        improved = val_mae < best_val_mae

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": round(tr_loss / max(1, tr_n), 5),
                "train_mae": round(train_mae, 4),
                "val_mae": round(val_mae, 4),
                "val_rmse": round(val_rmse, 4),
            }
        )

        if improved:
            best_val_mae = val_mae
            best_epoch = epoch
            no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if ckpt_path is not None:
                p = Path(ckpt_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "val_mae_db": val_mae,
                        "val_rmse_db": val_rmse,
                        "model_state_dict": best_state,
                    },
                    p,
                )
        else:
            no_improve += 1

        if logger is not None and (epoch == 1 or epoch % 5 == 0 or improved or epoch == epochs):
            logger.info(
                "[AuxSNR] Epoch %02d/%02d | tr_mae=%.3f dB | val_mae=%.3f dB (rmse=%.3f dB)%s",
                epoch,
                epochs,
                train_mae,
                val_mae,
                val_rmse,
                " [BEST]" if improved else "",
            )

        if no_improve >= int(patience):
            if logger is not None:
                logger.info("[AuxSNR] Early stopping at epoch %d (best=%d, val_mae=%.3f dB)", epoch, best_epoch, best_val_mae)
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})

    return {
        "best_epoch": best_epoch,
        "best_val_mae_db": round(best_val_mae, 4),
        "elapsed_sec": round(time.time() - t0, 2),
        "history": history,
    }


def _gate_binary_metrics(true_snrs: np.ndarray, pred_snrs: np.ndarray, threshold: float) -> dict[str, float]:
    y_true = true_snrs >= threshold
    y_pred = pred_snrs >= threshold
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum((~y_true) & y_pred))
    fn = int(np.sum(y_true & (~y_pred)))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = (2.0 * prec * rec) / max(1e-8, prec + rec)
    acc = float((y_true == y_pred).mean())
    return {
        "threshold_db": float(threshold),
        "precision_pct": round(prec * 100.0, 2),
        "recall_pct": round(rec * 100.0, 2),
        "f1_pct": round(f1 * 100.0, 2),
        "accuracy_pct": round(acc * 100.0, 2),
    }


def evaluate_snr_estimator(
    estimator: BaseSNREstimator,
    x_iq: np.ndarray | torch.Tensor,
    true_snrs: np.ndarray,
    labels: np.ndarray | None = None,
    idx_to_name: dict[int, str] | None = None,
    gate_thresholds: tuple[float, ...] = (0.0, 4.0),
    batch_size: int = 512,
    device: str | torch.device = "cuda",
) -> dict[str, Any]:
    """Evaluate regression MAE/RMSE and SNR gate binary F1/Precision/Recall."""
    dev = torch.device(device)
    x_t = torch.as_tensor(x_iq, dtype=torch.float32)
    y_s = np.asarray(true_snrs, dtype=np.float64).reshape(-1)
    if len(x_t) == 0 or len(x_t) != len(y_s):
        raise ValueError("x_iq and true_snrs must be non-empty and match in length")

    preds_l: list[np.ndarray] = []
    for start in range(0, len(x_t), batch_size):
        xb = x_t[start : start + batch_size].to(dev, non_blocking=True)
        preds_l.append(estimator.predict_snr(xb).detach().cpu().numpy())
    preds = np.concatenate(preds_l, axis=0).astype(np.float64)

    err = preds - y_s
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt(np.mean(err * err)))
    bias = float(err.mean())

    gates = {f"ge_{thr:g}db": _gate_binary_metrics(y_s, preds, float(thr)) for thr in gate_thresholds}

    per_class: dict[str, dict[str, Any]] = {}
    if labels is not None:
        lbl_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
        for cid in sorted(np.unique(lbl_arr).tolist()):
            m = lbl_arr == cid
            name = idx_to_name[cid] if (idx_to_name is not None and cid in idx_to_name) else str(cid)
            c_err = preds[m] - y_s[m]
            c_gates = {f"ge_{thr:g}db": _gate_binary_metrics(y_s[m], preds[m], float(thr)) for thr in gate_thresholds}
            per_class[name] = {
                "n": int(m.sum()),
                "mae_db": round(float(np.abs(c_err).mean()), 3),
                "rmse_db": round(float(np.sqrt(np.mean(c_err * c_err))), 3),
                "bias_db": round(float(c_err.mean()), 3),
                "gates": c_gates,
            }

    return {
        "n_samples": int(len(y_s)),
        "mae_db": round(mae, 3),
        "rmse_db": round(rmse, 3),
        "bias_db": round(bias, 3),
        "gates": gates,
        "per_class": per_class,
    }
