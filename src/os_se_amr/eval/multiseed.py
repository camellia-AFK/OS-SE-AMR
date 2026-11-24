"""Statistical aggregation helpers for multi-seed evaluation (mean +/- sample std)."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


def paired_significance_test(
    values_a: list[float],
    values_b: list[float],
) -> dict[str, Any]:
    """Run two-sided paired t-test and Wilcoxon sign-rank summary between two aligned seed lists."""
    a = np.asarray([float(v) for v in values_a], dtype=np.float64)
    b = np.asarray([float(v) for v in values_b], dtype=np.float64)
    if len(a) != len(b):
        raise ValueError(f"Aligned seed lists required, got len(a)={len(a)}, len(b)={len(b)}")
    diff = a - b
    mean_diff = float(np.mean(diff)) if len(diff) > 0 else 0.0
    if len(a) < 2 or float(np.std(diff)) < 1e-12:
        p_val = 0.0 if (len(a) >= 2 and abs(mean_diff) > 1e-9) else 1.0
        t_stat = 0.0
    else:
        res = stats.ttest_rel(a, b)
        t_stat = float(res.statistic)
        p_val = float(res.pvalue)
    return {
        "n": int(len(a)),
        "mean_diff": mean_diff,
        "t_stat": t_stat,
        "p_value": p_val,
        "significant_05": bool(p_val < 0.05),
        "significant_01": bool(p_val < 0.01),
    }


def summarize_scalar(values: list[float]) -> dict[str, Any]:
    """Compute count, arithmetic mean, and unbiased sample standard deviation (ddof=1)."""
    clean = [float(v) for v in values]
    n = len(clean)
    if n == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "values": []}
    arr = np.asarray(clean, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n >= 2 else 0.0
    return {"n": n, "mean": mean, "std": std, "values": clean}


def format_mean_std(stat: dict[str, Any], precision: int = 2) -> str:
    """Format a scalar stat dict as ``mean +/- std`` in LaTeX/Markdown."""
    return f"{float(stat['mean']):.{precision}f} ± {float(stat['std']):.{precision}f}"


def _extract_snr_metrics(block: dict[str, Any]) -> dict[str, float]:
    f1_val = block.get("f1_4db_pct")
    if f1_val is None:
        gates = block.get("gates", {})
        g4 = gates.get("ge_4db", {})
        f1_val = g4.get("f1_pct", 0.0)
    return {
        "mae_db": float(block["mae_db"]),
        "rmse_db": float(block["rmse_db"]),
        "f1_4db_pct": float(f1_val),
    }


def extract_closed_loop_metrics(run: dict[str, Any], include_clean10_extra: bool = False) -> dict[str, float]:
    """Flatten S0/S1/S2 closed-loop trajectory summary metrics into scalar floats."""
    sessions = run["sessions"]
    s0, s1, s2 = sessions[0], sessions[1], sessions[2]
    out: dict[str, float] = {
        "s0_a_all_pct": float(s0["a_all_pct"]),
        "s1_a_all_pct": float(s1["a_all_pct"]),
        "s1_a_old_pct": float(s1["a_old_pct"]),
        "s1_a_new_pct": float(s1["a_new_pct"] or 0.0),
        "s2_a_all_pct": float(s2["a_all_pct"]),
        "s2_a_old_pct": float(s2["a_old_pct"]),
        "s2_a_new_pct": float(s2["a_new_pct"] or 0.0),
        "delta_a_old_pct": float(s2["delta_a_old_pct"]),
        "fm_cls_pct": float(s2["fm_cls_pct"]),
    }
    if include_clean10_extra:
        out["s1_k_hat"] = float(s1.get("discovered_clusters", 0))
        out["s2_k_hat"] = float(s2.get("discovered_clusters", 0))
        out["s2_core_purity_pct"] = float(s2.get("core_purity_pct", 0.0))
    return out


def _aggregate_records(records: list[dict[str, float]], keys: tuple[str, ...]) -> dict[str, Any]:
    return {k: summarize_scalar([float(rec[k]) for rec in records]) for k in keys}


def aggregate_multiseed_results(seed_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate normalized per-seed summaries into mean +/- sample std (ddof=1) tables."""
    if not seed_summaries:
        raise ValueError("seed_summaries must not be empty")

    seeds = [int(s["seed"]) for s in seed_summaries]

    # 1. Stage 0 closed-set backbone
    stage0_stats = {
        "tldnn_test_acc_pct": summarize_scalar([float(s["stage0"]["tldnn_test_acc_pct"]) for s in seed_summaries])
    }

    # 2. Stage 1 static OSR methods
    s1_methods = list(seed_summaries[0].get("stage1_osr", {}).keys())
    s1_keys = ("known_acc_pct", "auroc_pct", "fpr_at_tpr95_pct", "oscr_pct")
    stage1_stats: dict[str, Any] = {}
    for m in s1_methods:
        recs = [s["stage1_osr"][m] for s in seed_summaries if m in s.get("stage1_osr", {})]
        if recs:
            stage1_stats[m] = _aggregate_records(recs, s1_keys)

    # 3. Blind SNR estimator generalization
    snr_keys = ("mae_db", "rmse_db", "f1_4db_pct")
    snr_stats: dict[str, Any] = {}
    for est in ("m2m4", "aux_snr"):
        snr_stats[est] = {}
        for ds_key in ("known_test", "valid_unseen_unknown_test"):
            recs = [
                _extract_snr_metrics(s["blind_snr"][est][ds_key])
                for s in seed_summaries
                if est in s.get("blind_snr", {}) and ds_key in s["blind_snr"][est]
            ]
            if recs:
                snr_stats[est][ds_key] = _aggregate_records(recs, snr_keys)

    # 4. Table 2 closed-loop trajectories across Order A & Order B
    t2_keys = (
        "s0_a_all_pct",
        "s1_a_all_pct",
        "s1_a_old_pct",
        "s1_a_new_pct",
        "s2_a_all_pct",
        "s2_a_old_pct",
        "s2_a_new_pct",
        "delta_a_old_pct",
        "fm_cls_pct",
    )
    table2_stats: dict[str, Any] = {}
    first_t2 = seed_summaries[0].get("table2_closed_loop", [])
    order_method_pairs: list[tuple[str, str]] = []
    for r in first_t2:
        pair = (str(r["order"]), str(r["method"]))
        if pair not in order_method_pairs:
            order_method_pairs.append(pair)

    for order_name, method_name in order_method_pairs:
        recs: list[dict[str, float]] = []
        for s in seed_summaries:
            match = next(
                (r for r in s.get("table2_closed_loop", []) if r["order"] == order_name and r["method"] == method_name),
                None,
            )
            if match is not None:
                recs.append(extract_closed_loop_metrics(match, include_clean10_extra=False))
        if recs:
            table2_stats.setdefault(order_name, {})[method_name] = _aggregate_records(recs, t2_keys)

    # 5. Table 2B Clean10 closed-loop benchmark
    c10_keys = t2_keys + ("s1_k_hat", "s2_k_hat", "s2_core_purity_pct")
    c10_runs_first = seed_summaries[0].get("clean10_closed_loop", {}).get("runs", [])
    c10_methods = [str(r["method"]) for r in c10_runs_first]
    c10_stats: dict[str, Any] = {}
    for method_name in c10_methods:
        recs = []
        for s in seed_summaries:
            runs = s.get("clean10_closed_loop", {}).get("runs", [])
            match = next((r for r in runs if r["method"] == method_name), None)
            if match is not None:
                recs.append(extract_closed_loop_metrics(match, include_clean10_extra=True))
        if recs:
            c10_stats[method_name] = _aggregate_records(recs, c10_keys)

    clean10_out = {
        "openness_pct": float(seed_summaries[0].get("clean10_closed_loop", {}).get("openness_pct", 9.25)),
        "methods": c10_stats,
    }

    # 6. Order B prototype synchronization ablation (S2)
    sync_keys = (
        "s2_known_frr_pct",
        "s2_unk_purity_pct",
        "s2_a_all_pct",
        "s2_a_old_pct",
        "s2_delta_a_old_pct",
    )
    sync_first = seed_summaries[0].get("sync_ablation_order_b", [])
    sync_methods = [str(r["method"]) for r in sync_first]
    sync_stats: dict[str, Any] = {}
    for method_name in sync_methods:
        recs = []
        for s in seed_summaries:
            match = next((r for r in s.get("sync_ablation_order_b", []) if r["method"] == method_name), None)
            if match is not None:
                s2 = match["sessions"][2]
                g2 = s2["gating"]
                recs.append(
                    {
                        "s2_known_frr_pct": float(g2["known_frr_on_stream_pct"]),
                        "s2_unk_purity_pct": float(g2["unknown_purity_pct"]),
                        "s2_a_all_pct": float(s2["a_all_pct"]),
                        "s2_a_old_pct": float(s2["a_old_pct"]),
                        "s2_delta_a_old_pct": float(s2["delta_a_old_pct"]),
                    }
                )
        if recs:
            sync_stats[method_name] = _aggregate_records(recs, sync_keys)

    return {
        "seeds": seeds,
        "n_seeds": len(seeds),
        "std_ddof": 1,
        "stage0": stage0_stats,
        "stage1_osr": stage1_stats,
        "blind_snr": snr_stats,
        "table2_closed_loop": table2_stats,
        "clean10_closed_loop": clean10_out,
        "sync_ablation_order_b": sync_stats,
    }
