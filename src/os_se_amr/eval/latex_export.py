"""Publication-grade Overleaf LaTeX (booktabs) exporters & 300 DPI PDF/SVG vector plot helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render_booktabs_table(
    caption: str,
    label: str,
    headers: list[str],
    rows: list[list[str]],
    col_spec: str | None = None,
) -> str:
    """Format an IEEE double-column/single-column booktabs LaTeX table string."""
    if col_spec is None:
        col_spec = "l" + "c" * (len(headers) - 1)
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(" & ".join(str(x) for x in r) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    return "\n".join(lines) + "\n"


def save_latex_bundle(tables: dict[str, str], out_dir: str | Path) -> list[Path]:
    """Write individual `.tex` tables and one combined `all_tables.tex` file."""
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    combined: list[str] = ["% Auto-generated Overleaf booktabs tables for OS-SE-AMR (IEEE TWC/TCOM)", ""]
    for name, tex in tables.items():
        p = target / f"{name}.tex"
        p.write_text(tex, encoding="utf-8")
        written.append(p)
        combined.append(tex)
    all_p = target / "all_tables.tex"
    all_p.write_text("\n".join(combined), encoding="utf-8")
    written.append(all_p)
    return written


def plot_covariance_anisotropy_spectrum(
    eigvals_by_layer: dict[str, np.ndarray],
    out_stem: str | Path,
) -> list[Path]:
    """Plot normalized class-conditional covariance eigenvalue spectrum across representation layers."""
    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.9})
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for layer_name, eigs in eigvals_by_layer.items():
        arr = np.sort(np.maximum(np.asarray(eigs, dtype=np.float64), 1e-12))[::-1]
        arr = arr / max(float(arr.max()), 1e-12)
        ranks = np.arange(1, len(arr) + 1)
        ax.semilogy(ranks, arr, linewidth=2.0, label=layer_name)
    ax.set_xlabel("Covariance Eigenvalue Rank (Descending)")
    ax.set_ylabel("Normalized Eigenvalue (Log Scale)")
    ax.set_title("Anisotropic Neural Collapse Spectrum Across Backbone Layers")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(frameon=True, fontsize=8.5)
    fig.tight_layout()

    stem = Path(out_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for ext in ("pdf", "svg", "png"):
        out_p = stem.with_suffix(f".{ext}")
        fig.savefig(out_p, dpi=300, bbox_inches="tight")
        saved.append(out_p)
    plt.close(fig)
    return saved


def plot_phase_continuity_zoom(
    x_iq_stitched: np.ndarray,
    x_pa_stitched: np.ndarray,
    cut: int | None = None,
    out_stem: str | Path = "phase_continuity_zoom",
) -> list[Path]:
    """Zoomed comparison of SS-IQ vs SS-PhaseAligned seam waveform and instantaneous phase derivative."""
    arr_iq = np.asarray(x_iq_stitched, dtype=np.float32)
    arr_pa = np.asarray(x_pa_stitched, dtype=np.float32)
    c = (arr_iq.shape[-1] // 2) if cut is None else int(cut)
    lo, hi = max(0, c - 16), min(arr_iq.shape[-1], c + 16)
    t_idx = np.arange(lo, hi)

    def _dphase(arr: np.ndarray) -> np.ndarray:
        z = arr[0] + 1j * arr[1]
        dp = np.angle(z[1:] * np.conj(z[:-1])) / np.pi
        return np.concatenate([[dp[0]], dp])

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    axes[0].plot(t_idx, arr_iq[0, lo:hi], "--", linewidth=1.6, label="SS-IQ (I)")
    axes[0].plot(t_idx, arr_pa[0, lo:hi], "-", linewidth=2.0, label="SS-PhaseAligned (I)")
    axes[0].axvline(c - 0.5, color="red", linestyle=":", linewidth=1.2, label="Stitch Seam")
    axes[0].set_xlabel("Sample Index")
    axes[0].set_ylabel("In-Phase Amplitude")
    axes[0].set_title("Time-Domain Seam Continuity")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(fontsize=8)

    dp_iq = _dphase(arr_iq)[lo:hi]
    dp_pa = _dphase(arr_pa)[lo:hi]
    axes[1].plot(t_idx, dp_iq, "--o", markersize=3, linewidth=1.5, label="SS-IQ Inst. Freq")
    axes[1].plot(t_idx, dp_pa, "-s", markersize=3, linewidth=1.8, label="SS-PhaseAligned Inst. Freq")
    axes[1].axvline(c - 0.5, color="red", linestyle=":", linewidth=1.2, label="Stitch Seam")
    axes[1].set_xlabel("Sample Index")
    axes[1].set_ylabel("Normalized Freq (pi rad/sample)")
    axes[1].set_title("Instantaneous Frequency Jump Suppression")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend(fontsize=8)
    fig.tight_layout()

    stem = Path(out_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for ext in ("pdf", "svg", "png"):
        out_p = stem.with_suffix(f".{ext}")
        fig.savefig(out_p, dpi=300, bbox_inches="tight")
        saved.append(out_p)
    plt.close(fig)
    return saved


def plot_sensitivity_grid(
    panels: list[dict[str, Any]],
    out_stem: str | Path,
) -> list[Path]:
    """Render 2x2 publication hyperparameter sensitivity plot in PDF, SVG, and PNG."""
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2))
    flat_axes = axes.ravel()
    for idx, panel in enumerate(panels[:4]):
        ax = flat_axes[idx]
        xs = panel["xs"]
        for label, ys in panel["series"].items():
            ax.plot(xs, ys, marker="o", linewidth=1.9, label=label)
        ax.set_xlabel(panel["xlabel"])
        ax.set_ylabel(panel["ylabel"])
        ax.set_title(panel["title"], fontsize=9.5)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=8)
    for idx in range(len(panels), 4):
        flat_axes[idx].axis("off")
    fig.tight_layout()

    stem = Path(out_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for ext in ("pdf", "svg", "png"):
        out_p = stem.with_suffix(f".{ext}")
        fig.savefig(out_p, dpi=300, bbox_inches="tight")
        saved.append(out_p)
    plt.close(fig)
    return saved
