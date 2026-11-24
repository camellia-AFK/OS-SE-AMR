"""Unified cross-stage benchmark aggregator for OS-SE-AMR (Stages 0 through 4)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from os_se_amr.eval.multiseed import format_mean_std
from os_se_amr.utils import resolve_path, save_json


def _load_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    out_dir = resolve_path("outputs")
    s0_tldnn = _load_if_exists(out_dir / "stage0" / "tldnn_config_a_seed42" / "metrics.json")
    s0_iq = _load_if_exists(out_dir / "stage0" / "iqformer_config_a_seed42" / "metrics.json")
    s1 = _load_if_exists(out_dir / "stage1" / "summary_table1.json")
    s2 = _load_if_exists(out_dir / "stage2" / "summary_stage2.json")
    s3 = _load_if_exists(out_dir / "stage3" / "summary_stage3.json")
    s4 = _load_if_exists(out_dir / "stage4" / "summary_stage4.json")
    ms = _load_if_exists(out_dir / "multiseed" / "summary_multiseed.json")

    report = {
        "stage0": {
            "tldnn_test_acc_pct": round(s0_tldnn["test_metrics"]["overall_acc"] * 100.0, 2) if s0_tldnn else None,
            "iqformer_test_acc_pct": round(s0_iq["test_metrics"]["overall_acc"] * 100.0, 2) if s0_iq else None,
        },
        "stage1_table1": s1.get("table1", []) if s1 else [],
        "stage2_decoupled": s2 if s2 else {},
        "stage3_baselines_and_drift": s3 if s3 else {},
        "stage4_closed_loop": s4 if s4 else {},
        "multiseed_stats": ms.get("aggregated", {}) if ms else {},
    }
    save_json(report, out_dir / "final_benchmark_report.json")

    lines = ["# OS-SE-AMR 全五阶段最终基准测试总报告", ""]
    if s0_tldnn and s0_iq:
        lines.extend(
            [
                "## 阶段 0：封闭集骨干基准（配置 A，SNR >= 0 dB）",
                "",
                f"- `TLDNN` Test Acc: **{report['stage0']['tldnn_test_acc_pct']:.2f}%**",
                f"- `IQFormer` Test Acc: **{report['stage0']['iqformer_test_acc_pct']:.2f}%**",
                "",
            ]
        )

    if s3 and s3.get("static_osr"):
        lines.extend(["## 阶段 1 & 3：静态 OSR 对齐（配置 A Openness=11.81% / 配置 B Openness=20.94%）", ""])
        for key in ("config_a", "config_b"):
            rows = s3["static_osr"].get(key, [])
            if not rows:
                continue
            lines.extend(
                [
                    f"### {key}",
                    "| 骨干 | 算法 | Known Acc (%) | AUROC (%) | FPR@TPR95 (%) | OSCR (%) |",
                    "| :--- | :--- | :---: | :---: | :---: | :---: |",
                ]
            )
            for r in rows:
                lines.append(
                    f"| {r['backbone']} | {r['algorithm']} | {r['known_acc_pct']:.2f} | "
                    f"{r['auroc_pct']:.2f} | {r['fpr_at_tpr95_pct']:.2f} | {r['oscr_pct']:.2f} |"
                )
            lines.append("")

    if s4 and s4.get("table2_closed_loop"):
        lines.extend(
            [
                "## 阶段 4：端到端真实混流自进化闭环（Table 2）",
                "",
                "| 协议 | 方法 | S0 `A_all` | S1 `A_all / A_old / A_new` | S2 `A_all / A_old / A_new` | 终态 `ΔA_old` |",
                "| :--- | :--- | :---: | :---: | :---: | :---: |",
            ]
        )
        for r in s4["table2_closed_loop"]:
            s_0, s_1, s_2 = r["sessions"][0], r["sessions"][1], r["sessions"][2]
            lines.append(
                f"| {r['order']} | {r['method']} | {s_0['a_all_pct']:.2f}% | "
                f"{s_1['a_all_pct']:.2f}% / {s_1['a_old_pct']:.2f}% / {s_1['a_new_pct']:.2f}% | "
                f"{s_2['a_all_pct']:.2f}% / {s_2['a_old_pct']:.2f}% / {s_2['a_new_pct']:.2f}% | "
                f"{s_2['delta_a_old_pct']:.2f}% |"
            )
        lines.append("")

    c10 = (s4 or {}).get("clean10_closed_loop", {})
    if c10 and c10.get("runs"):
        op = float(c10.get("openness_pct", 9.25))
        lines.extend(
            [
                f"### Clean10 无缺陷 10 类闭环对比（7 -> 9 -> 10，Openness = {op:.2f}%）",
                "",
                "| 方法 | S0 `A_all` | S1 `A_all / A_old / A_new` | S2 `A_all / A_old / A_new` | 终态 `ΔA_old` | S1/S2 簇数 `K_hat` |",
                "| :--- | :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        for r in c10["runs"]:
            s_0, s_1, s_2 = r["sessions"][0], r["sessions"][1], r["sessions"][2]
            k_str = (
                f"S1:{s_1.get('raw_discovered_clusters', s_1.get('discovered_clusters', 0))}->{s_1.get('discovered_clusters', 0)}, "
                f"S2:{s_2.get('raw_discovered_clusters', s_2.get('discovered_clusters', 0))}->{s_2.get('discovered_clusters', 0)}"
            )
            lines.append(
                f"| {r['method']} | {s_0['a_all_pct']:.2f}% | "
                f"{s_1['a_all_pct']:.2f}% / {s_1['a_old_pct']:.2f}% / {s_1['a_new_pct']:.2f}% | "
                f"{s_2['a_all_pct']:.2f}% / {s_2['a_old_pct']:.2f}% / {s_2['a_new_pct']:.2f}% | "
                f"{s_2['delta_a_old_pct']:.2f}% | {k_str} |"
            )
        lines.append("")

    agg = report.get("multiseed_stats") or {}
    if agg:
        seeds_str = ", ".join(str(x) for x in agg.get("seeds", []))
        lines.extend(
            [
                f"## 多种子统计稳定性验证（`seed ∈ {{{seeds_str}}}`，`mean ± std`）",
                "",
                "### 静态 OSR 与盲 SNR 估计",
                "| 模块 / 方法 | 核心指标 | `mean ± std` |",
                "| :--- | :--- | :---: |",
                f"| Stage 0 `TLDNN` | Test Acc (%) | {format_mean_std(agg['stage0']['tldnn_test_acc_pct'])} |",
            ]
        )
        for m_key, m_name in [
            ("tldnn_softmax", "Stage 1 `Softmax`"),
            ("tldnn_mls", "Stage 1 `MLS`"),
            ("tldnn_ms_maha", "Stage 1 `MS-MahaProto (Ours)`"),
        ]:
            if m_key in agg.get("stage1_osr", {}):
                d = agg["stage1_osr"][m_key]
                lines.append(
                    f"| {m_name} | AUROC / OSCR (%) | {format_mean_std(d['auroc_pct'])} / {format_mean_std(d['oscr_pct'])} |"
                )
        lines.extend(
            [
                "",
                "### 闭环演化核心对比（`Table 2 Order A/B` + `Clean10`）",
                "| 基准 | 方法 | S1 `A_all / A_new` (%) | S2 `A_all` (%) | S2 `A_old` (%) | 终态 `ΔA_old` (%) |",
                "| :--- | :--- | :---: | :---: | :---: | :---: |",
            ]
        )
        for order_name, meth_map in agg.get("table2_closed_loop", {}).items():
            for meth in ("Static Baseline", "Naive Fine-Tune", "iCaRL (m=100)", "Ours (W/O SS Aug)", "Ours (Full Model + SS)"):
                if meth in meth_map:
                    d = meth_map[meth]
                    lines.append(
                        f"| {order_name} | {meth} | {format_mean_std(d['s1_a_all_pct'])} / {format_mean_std(d['s1_a_new_pct'])} | "
                        f"{format_mean_std(d['s2_a_all_pct'])} | {format_mean_std(d['s2_a_old_pct'])} | {format_mean_std(d['delta_a_old_pct'])} |"
                    )
        c10_agg = agg.get("clean10_closed_loop", {}).get("methods", {})
        for meth in ("Static Baseline", "Naive Fine-Tune", "iCaRL (m=100)", "Ours (W/O SS Aug)", "Ours (Full Model + SS)"):
            if meth in c10_agg:
                d = c10_agg[meth]
                lines.append(
                    f"| Clean10 | {meth} | {format_mean_std(d['s1_a_all_pct'])} / {format_mean_std(d['s1_a_new_pct'])} | "
                    f"{format_mean_std(d['s2_a_all_pct'])} | {format_mean_std(d['s2_a_old_pct'])} | {format_mean_std(d['delta_a_old_pct'])} |"
                )
        lines.append("")

    md_path = out_dir / "final_benchmark_report.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Final benchmark report saved to {md_path}")


if __name__ == "__main__":
    main()
