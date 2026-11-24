"""Sample-level OSR confidence + Blind SNR Dual-Aware Evolution Gate (Stage 4)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VALID_GATE_MODES = ("osr_only", "m2m4", "aux_snr", "oracle_snr")


@dataclass
class GatingReport:
    """Audit metrics and sample selection mask emitted by DualAwareGate."""

    gate_mode: str
    total_samples: int
    passed_count: int
    osr_rejected_count: int
    snr_passed_count: int
    osr_threshold: float
    snr_threshold: float
    unknown_purity_pct: float
    main_unknown_recall_pct: float
    neg_snr_leakage_pct: float
    known_fp_ratio_pct: float
    passed_mask: np.ndarray


class DualAwareGate:
    """First-stage dual-aware sample gate: S_OSR(x) < tau_low AND SNR_hat(x) >= tau_snr."""

    def __init__(self, gate_mode: str = "aux_snr", snr_threshold: float = 4.0) -> None:
        if gate_mode not in VALID_GATE_MODES:
            raise ValueError(f"Unsupported gate_mode='{gate_mode}'; choose from {VALID_GATE_MODES}")
        self.gate_mode = str(gate_mode)
        self.snr_threshold = float(snr_threshold)

    def filter(
        self,
        osr_scores: np.ndarray,
        osr_threshold: float,
        snr_est: np.ndarray | None = None,
        true_labels: np.ndarray | None = None,
        true_snrs: np.ndarray | None = None,
        num_known_classes: int | None = None,
        novel_class_ids: list[int] | set[int] | None = None,
    ) -> GatingReport:
        """Select high-confidence unknown candidates while keeping offline metrics strictly decoupled."""
        osr_s = np.asarray(osr_scores, dtype=np.float64).reshape(-1)
        n = len(osr_s)
        thr_osr = float(osr_threshold)
        osr_mask = osr_s < thr_osr

        if self.gate_mode == "osr_only":
            snr_mask = np.ones(n, dtype=bool)
        elif self.gate_mode == "oracle_snr":
            if true_snrs is None:
                raise ValueError("true_snrs must be provided when gate_mode='oracle_snr'")
            snr_arr = np.asarray(true_snrs, dtype=np.float64).reshape(-1)
            if len(snr_arr) != n:
                raise ValueError(f"Length mismatch: osr_scores={n}, true_snrs={len(snr_arr)}")
            snr_mask = snr_arr >= self.snr_threshold
        else:
            if snr_est is None:
                raise ValueError(f"snr_est must be provided when gate_mode='{self.gate_mode}'")
            snr_arr = np.asarray(snr_est, dtype=np.float64).reshape(-1)
            if len(snr_arr) != n:
                raise ValueError(f"Length mismatch: osr_scores={n}, snr_est={len(snr_arr)}")
            snr_mask = snr_arr >= self.snr_threshold

        passed_mask = osr_mask & snr_mask

        unknown_purity_pct = 0.0
        known_fp_ratio_pct = 0.0
        main_unknown_recall_pct = 0.0
        neg_snr_leakage_pct = 0.0

        if true_labels is not None and (novel_class_ids is not None or num_known_classes is not None) and n > 0:
            lbls = np.asarray(true_labels, dtype=np.int64).reshape(-1)
            if novel_class_ids is not None:
                is_unk = np.isin(lbls, sorted(int(c) for c in novel_class_ids))
            else:
                assert num_known_classes is not None
                is_unk = lbls >= int(num_known_classes)
            is_known = ~is_unk
            if passed_mask.any():
                unknown_purity_pct = float(is_unk[passed_mask].mean() * 100.0)
                known_fp_ratio_pct = float(is_known[passed_mask].mean() * 100.0)

            if true_snrs is not None:
                snrs_diag = np.asarray(true_snrs, dtype=np.float64).reshape(-1)
                main_unk = is_unk & (snrs_diag >= 0.0)
                if main_unk.any():
                    main_unknown_recall_pct = float(passed_mask[main_unk].mean() * 100.0)
                if passed_mask.any():
                    neg_snr_leakage_pct = float((snrs_diag[passed_mask] < 0.0).mean() * 100.0)
            elif is_unk.any():
                main_unknown_recall_pct = float(passed_mask[is_unk].mean() * 100.0)

        return GatingReport(
            gate_mode=self.gate_mode,
            total_samples=n,
            passed_count=int(passed_mask.sum()),
            osr_rejected_count=int(osr_mask.sum()),
            snr_passed_count=int(snr_mask.sum()),
            osr_threshold=thr_osr,
            snr_threshold=self.snr_threshold,
            unknown_purity_pct=round(unknown_purity_pct, 2),
            main_unknown_recall_pct=round(main_unknown_recall_pct, 2),
            neg_snr_leakage_pct=round(neg_snr_leakage_pct, 2),
            known_fp_ratio_pct=round(known_fp_ratio_pct, 2),
            passed_mask=passed_mask,
        )
