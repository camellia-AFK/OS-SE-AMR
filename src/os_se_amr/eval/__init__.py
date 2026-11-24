"""Evaluation metrics for open-set recognition, clustering, continual learning, and SNR drift."""

from os_se_amr.eval.cil_metrics import evaluate_cil_session
from os_se_amr.eval.hungarian import (
    apply_locked_mapping,
    evaluate_clustering_diagnostics,
    locked_known_hungarian_match,
)
from os_se_amr.eval.latex_export import (
    plot_covariance_anisotropy_spectrum,
    plot_phase_continuity_zoom,
    plot_sensitivity_grid,
    render_booktabs_table,
    save_latex_bundle,
)
from os_se_amr.eval.multiseed import (
    aggregate_multiseed_results,
    extract_closed_loop_metrics,
    format_mean_std,
    paired_significance_test,
    summarize_scalar,
)
from os_se_amr.eval.osr_metrics import (
    compute_auroc,
    compute_fpr_at_tpr,
    compute_oscr,
    evaluate_osr,
)
from os_se_amr.eval.profiler import estimate_conv_linear_macs, profile_system_overhead
from os_se_amr.eval.robustness import (
    DEFAULT_SNR_REGIMES,
    evaluate_snr_cil_drift,
    evaluate_snr_osr_drift,
)

__all__ = [
    "DEFAULT_SNR_REGIMES",
    "aggregate_multiseed_results",
    "apply_locked_mapping",
    "compute_auroc",
    "compute_fpr_at_tpr",
    "compute_oscr",
    "estimate_conv_linear_macs",
    "evaluate_cil_session",
    "evaluate_clustering_diagnostics",
    "evaluate_osr",
    "evaluate_snr_cil_drift",
    "evaluate_snr_osr_drift",
    "extract_closed_loop_metrics",
    "format_mean_std",
    "locked_known_hungarian_match",
    "paired_significance_test",
    "plot_covariance_anisotropy_spectrum",
    "plot_phase_continuity_zoom",
    "plot_sensitivity_grid",
    "profile_system_overhead",
    "render_booktabs_table",
    "save_latex_bundle",
    "summarize_scalar",
]
