"""End-to-end closed-loop streaming self-evolution scheduler (Stage 4)."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from os_se_amr.continual import (
    DERLearner,
    DualLwFLearner,
    EWCLearner,
    NaiveFineTuneLearner,
    ReplayBuffer,
    WALearner,
    iCaRLLearner,
    sync_prototypes_and_mahalanobis,
)
from os_se_amr.data.splitter import SplitBundle
from os_se_amr.data.stream import StreamBatch, _sample_balanced_from_classes
from os_se_amr.data.transforms import extract_rf_physics_features
from os_se_amr.discovery import (
    DualAwareGate,
    HDBSCANDiscoverer,
    RejectionBuffer,
    augment_core_clusters,
)
from os_se_amr.eval import evaluate_cil_session, evaluate_clustering_diagnostics
from os_se_amr.models import BaseBackbone, BaseSNREstimator, build_backbone
from os_se_amr.osr import MahalanobisPrototypeScorer, calibrate_threshold
from os_se_amr.utils import load_checkpoint, seed_everything


@dataclass
class ClosedLoopConfig:
    """Configuration for one closed-loop evolution trajectory."""

    method_key: str  # "static", "naive", "icarl", "ewc", "wa", "der", "ours"
    gate_mode: str = "aux_snr"  # "osr_only", "m2m4", "aux_snr", "oracle_snr"
    snr_threshold: float = 4.0
    target_tpr: float = 0.90
    ss_mode: str = "ss_phase_aligned"  # "none", "ss_iq", "ss_phase_aligned"
    ss_expand_factor: int = 4
    ss_max_per_cluster: int = 2000
    sync_prototypes: bool = True
    exemplars_per_class: int = 100
    min_cluster_size: int = 100
    min_samples: int = 20
    silhouette_threshold: float = 0.40
    core_prob_threshold: float = 0.75
    core_top_ratio: float = 0.60
    allow_single_cluster: bool = True
    merge_cosine_threshold: float | None = 0.80
    min_cluster_silhouette: float | None = 0.40
    min_cluster_cohesion: float | None = 0.85
    ss_min_cluster_silhouette: float | None = 0.45
    ss_min_cluster_cohesion: float | None = 0.90
    ss_max_core_size: int | None = 450
    include_physics: bool = False
    use_weight_align: bool = False
    replay_ss_discovered: bool = False
    use_amp: bool = False
    seed: int = 42


def build_closed_loop_mixed_sessions(
    bundle: SplitBundle,
    order_name: str = "order_a",
    samples_per_class_main: int = 2000,
    samples_per_class_neg: int = 600,
    known_ratio: float = 0.5,
    seed: int = 42,
) -> list[StreamBatch]:
    """Build realistic mixed streaming batches (known + main unknown + neg-SNR unknown) from Train split."""
    orders = bundle.split_cfg.get("arrival_orders", {})
    if order_name not in orders:
        raise KeyError(f"Arrival order '{order_name}' not found in split_cfg['arrival_orders']")
    if not (0.0 < known_ratio < 1.0):
        raise ValueError(f"known_ratio must be in (0, 1), got {known_ratio}")

    rng = np.random.default_rng(int(seed))
    seen_classes = list(bundle.known_classes)
    sessions: list[StreamBatch] = []

    ratio_scale = known_ratio / (1.0 - known_ratio)
    has_neg = samples_per_class_neg > 0 and len(bundle.stress_snrs) > 0

    for sess_idx, arriving in enumerate(orders[order_name], start=1):
        arriving_list = [str(c) for c in arriving]
        xs, ys, ss = [], [], []

        if samples_per_class_main > 0:
            x_um, y_um, s_um = _sample_balanced_from_classes(
                bundle=bundle,
                classes=arriving_list,
                total_per_class=samples_per_class_main,
                rng=rng,
                snrs=bundle.main_snrs,
            )
            per_k_main = max(1, int(round(len(x_um) * ratio_scale)) // len(seen_classes))
            x_km, y_km, s_km = _sample_balanced_from_classes(
                bundle=bundle,
                classes=seen_classes,
                total_per_class=per_k_main,
                rng=rng,
                snrs=bundle.main_snrs,
            )
            xs.extend([x_km, x_um])
            ys.extend([y_km, y_um])
            ss.extend([s_km, s_um])

        if has_neg:
            x_un, y_un, s_un = _sample_balanced_from_classes(
                bundle=bundle,
                classes=arriving_list,
                total_per_class=samples_per_class_neg,
                rng=rng,
                snrs=bundle.stress_snrs,
            )
            per_k_neg = max(1, int(round(len(x_un) * ratio_scale)) // len(seen_classes))
            x_kn, y_kn, s_kn = _sample_balanced_from_classes(
                bundle=bundle,
                classes=seen_classes,
                total_per_class=per_k_neg,
                rng=rng,
                snrs=bundle.stress_snrs,
            )
            xs.extend([x_kn, x_un])
            ys.extend([y_kn, y_un])
            ss.extend([s_kn, s_un])

        x_all = np.concatenate(xs, axis=0)
        y_all = np.concatenate(ys, axis=0)
        s_all = np.concatenate(ss, axis=0)
        perm = rng.permutation(len(x_all))

        sessions.append(
            StreamBatch(
                x_iq=np.ascontiguousarray(x_all[perm], dtype=np.float32),
                true_labels=np.ascontiguousarray(y_all[perm], dtype=np.int64),
                snrs=np.ascontiguousarray(s_all[perm], dtype=np.int64),
                session_id=sess_idx,
                arriving_classes=arriving_list,
                mode="mixed",
            )
        )
        seen_classes.extend(arriving_list)

    return sessions


def build_zero_novelty_session(
    bundle: SplitBundle,
    seen_classes: list[str] | None = None,
    samples_per_class_main: int = 500,
    samples_per_class_neg: int = 150,
    session_id: int = 1,
    seed: int = 42,
) -> StreamBatch:
    """Build a pure known-class + negative-SNR stream session with zero novel classes (anti-hallucination test)."""
    classes = list(bundle.known_classes) if seen_classes is None else [str(c) for c in seen_classes]
    rng = np.random.default_rng(int(seed))
    xs, ys, ss = [], [], []

    if samples_per_class_main > 0:
        xm, ym, sm = _sample_balanced_from_classes(
            bundle=bundle,
            classes=classes,
            total_per_class=samples_per_class_main,
            rng=rng,
            snrs=bundle.main_snrs,
        )
        xs.append(xm)
        ys.append(ym)
        ss.append(sm)

    if samples_per_class_neg > 0 and len(bundle.stress_snrs) > 0:
        xn, yn, sn = _sample_balanced_from_classes(
            bundle=bundle,
            classes=classes,
            total_per_class=samples_per_class_neg,
            rng=rng,
            snrs=bundle.stress_snrs,
        )
        xs.append(xn)
        ys.append(yn)
        ss.append(sn)

    x_all = np.concatenate(xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    s_all = np.concatenate(ss, axis=0)
    perm = rng.permutation(len(x_all))
    return StreamBatch(
        x_iq=np.ascontiguousarray(x_all[perm], dtype=np.float32),
        true_labels=np.ascontiguousarray(y_all[perm], dtype=np.int64),
        snrs=np.ascontiguousarray(s_all[perm], dtype=np.int64),
        session_id=int(session_id),
        arriving_classes=[],
        mode="zero_novelty",
    )


@torch.no_grad()
def _extract_ms_np(
    backbone: BaseBackbone,
    x_iq: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
    include_physics: bool = False,
) -> dict[str, np.ndarray]:
    backbone.eval()
    buf: dict[str, list[np.ndarray]] = {}
    for start in range(0, len(x_iq), batch_size):
        xb = torch.from_numpy(x_iq[start : start + batch_size]).to(device, non_blocking=True)
        ms = backbone.extract_multiscale_features(xb)
        for k, v in ms.items():
            buf.setdefault(k, []).append(v.cpu().numpy())
        if include_physics:
            buf.setdefault("physics", []).append(extract_rf_physics_features(xb).cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in buf.items()}


@torch.no_grad()
def _estimate_snr_np(
    estimator: BaseSNREstimator,
    x_iq: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    out: list[np.ndarray] = []
    for start in range(0, len(x_iq), batch_size):
        xb = torch.from_numpy(x_iq[start : start + batch_size]).to(device, non_blocking=True)
        out.append(estimator.predict_snr(xb).detach().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


@torch.no_grad()
def _predict_heads(
    backbone: BaseBackbone,
    ds: Any,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    backbone.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    targets, preds = [], []
    for x, y, _ in loader:
        lg, _ = backbone(x.to(device, non_blocking=True))
        preds.append(lg.argmax(dim=1).cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(targets, axis=0), np.concatenate(preds, axis=0)


class ClosedLoopEvolver:
    """End-to-end streaming closed-loop self-evolution engine (Mode 2)."""

    def _calibrate_osr_threshold(
        self,
        model: BaseBackbone,
        scorer: MahalanobisPrototypeScorer,
        val_known_x: np.ndarray,
        val_known_snrs: np.ndarray,
        snr_estimator: BaseSNREstimator | None,
        cl_cfg: ClosedLoopConfig,
        device: torch.device,
        extra_core_x: np.ndarray | None = None,
    ) -> float:
        val_ms = _extract_ms_np(model, val_known_x, device=device, include_physics=cl_cfg.include_physics)
        val_scores = scorer.score_multiscale(val_ms)

        if cl_cfg.gate_mode == "osr_only":
            pool_scores = val_scores
        else:
            if cl_cfg.gate_mode == "oracle_snr":
                val_snr_vals = np.asarray(val_known_snrs, dtype=np.float64)
            else:
                assert snr_estimator is not None
                val_snr_vals = _estimate_snr_np(snr_estimator, val_known_x, device=device)
            snr_mask = val_snr_vals >= cl_cfg.snr_threshold
            pool_scores = val_scores[snr_mask] if int(snr_mask.sum()) >= 100 else val_scores

        if extra_core_x is not None and len(extra_core_x) > 0:
            core_ms = _extract_ms_np(model, extra_core_x, device=device, include_physics=cl_cfg.include_physics)
            core_scores = scorer.score_multiscale(core_ms)
            pool_scores = np.concatenate([pool_scores, core_scores], axis=0)

        return calibrate_threshold(pool_scores, target_tpr=cl_cfg.target_tpr)

    def run_trajectory(
        self,
        method_label: str,
        order_name: str,
        cl_cfg: ClosedLoopConfig,
        sessions: list[StreamBatch],
        base_ckpt_path: Path,
        base_scorer: MahalanobisPrototypeScorer,
        snr_estimator: BaseSNREstimator | None,
        bundle: SplitBundle,
        model_cfg: dict[str, Any],
        cil_cfg: dict[str, Any],
        device: torch.device,
        logger: Any,
        save_final_ckpt: Path | None = None,
    ) -> dict[str, Any]:
        """Execute a complete closed-loop streaming trajectory (S0 -> S1 -> S2)."""
        seed = int(cl_cfg.seed)
        seed_everything(seed, deterministic=True)
        num_base = len(bundle.known_classes)
        idx_to_name = {v: k for k, v in bundle.all_to_idx.items()}

        model = build_backbone(copy.deepcopy(model_cfg), num_classes=num_base).to(device)
        load_checkpoint(base_ckpt_path, model, device=device)

        train_ds, val_known_ds, known_test = bundle.get_closed_set_datasets()
        val_known_loader = DataLoader(val_known_ds, batch_size=256, shuffle=False, num_workers=0)
        ms_tr = _extract_ms_np(model, train_ds.x, device=device, include_physics=cl_cfg.include_physics)

        replay_buf = ReplayBuffer(
            exemplars_per_class=cl_cfg.exemplars_per_class,
            selection_strategy=str(cil_cfg.get("selection_strategy", "herding")),
        )
        icarl: iCaRLLearner | None = None
        lwf_learner: DualLwFLearner | NaiveFineTuneLearner | None = None
        extra_learner: EWCLearner | WALearner | DERLearner | None = None

        if cl_cfg.method_key == "icarl":
            for cid in range(num_base):
                m = train_ds.y == cid
                replay_buf.add_class_exemplars(cid, train_ds.x[m], features=ms_tr["pre_ln"][m], snrs=None, seed=seed)
            ic_cfg = cil_cfg.get("icarl", {})
            icarl = iCaRLLearner(
                epochs=int(ic_cfg.get("epochs", 35)),
                batch_size=int(ic_cfg.get("batch_size", 128)),
                lr=float(ic_cfg.get("lr", 8e-4)),
                weight_decay=float(cil_cfg.get("weight_decay", 1e-4)),
                grad_clip=float(cil_cfg.get("grad_clip", 1.0)),
                lambda_kd=float(ic_cfg.get("lambda_kd", 1.0)),
                distill_temp=float(ic_cfg.get("distill_temp", 2.0)),
            )
            icarl.update_class_means(model, replay_buf, device=device)
            y_t0, y_p0 = icarl.predict_nme(model, known_test, device=device)
        elif cl_cfg.method_key in ("ewc", "wa", "der"):
            for cid in range(num_base):
                m = train_ds.y == cid
                replay_buf.add_class_exemplars(cid, train_ds.x[m], features=ms_tr["pre_ln"][m], snrs=None, seed=seed)
            if cl_cfg.method_key == "ewc":
                extra_learner = EWCLearner(
                    epochs=int(cil_cfg.get("epochs", 35)),
                    batch_size=int(cil_cfg.get("batch_size_new", 128)),
                    lr=float(cil_cfg.get("lr", 8e-4)),
                    weight_decay=float(cil_cfg.get("weight_decay", 1e-4)),
                    grad_clip=float(cil_cfg.get("grad_clip", 1.0)),
                    lambda_ewc=float(cil_cfg.get("lambda_ewc", 400.0)),
                )
            elif cl_cfg.method_key == "wa":
                extra_learner = WALearner(
                    epochs=int(cil_cfg.get("epochs", 35)),
                    batch_size=int(cil_cfg.get("batch_size_new", 128)),
                    lr=float(cil_cfg.get("lr", 8e-4)),
                    weight_decay=float(cil_cfg.get("weight_decay", 1e-4)),
                    grad_clip=float(cil_cfg.get("grad_clip", 1.0)),
                    lambda_kd=float(cil_cfg.get("beta_logit", 1.0)),
                    distill_temp=float(cil_cfg.get("distill_temp", 2.0)),
                )
            else:
                extra_learner = DERLearner(
                    epochs=int(cil_cfg.get("epochs", 35)),
                    batch_size=int(cil_cfg.get("batch_size_new", 128)),
                    lr=float(cil_cfg.get("lr", 8e-4)),
                    weight_decay=float(cil_cfg.get("weight_decay", 1e-4)),
                    grad_clip=float(cil_cfg.get("grad_clip", 1.0)),
                    aux_weight=float(cil_cfg.get("der_aux_weight", 0.5)),
                )
            y_t0, y_p0 = _predict_heads(model, known_test, device=device)
        elif cl_cfg.method_key == "ours":
            for cid in range(num_base):
                m = train_ds.y == cid
                replay_buf.add_class_exemplars(
                    cid, train_ds.x[m], features=ms_tr["pre_ln"][m], snrs=train_ds.snrs[m], seed=seed
                )
            lwf_learner = DualLwFLearner(
                epochs=int(cil_cfg.get("epochs", 35)),
                steps_per_epoch=int(cil_cfg.get("steps_per_epoch", 30)),
                samples_per_class_per_step=int(cil_cfg.get("samples_per_class_per_step", 24)),
                batch_size_new=int(cil_cfg.get("batch_size_new", 128)),
                batch_size_old=int(cil_cfg.get("batch_size_old", 128)),
                lr=float(cil_cfg.get("lr", 8e-4)),
                weight_decay=float(cil_cfg.get("weight_decay", 1e-4)),
                grad_clip=float(cil_cfg.get("grad_clip", 1.0)),
                beta_feat=float(cil_cfg.get("beta_feat", 0.05)),
                beta_logit=float(cil_cfg.get("beta_logit", 0.5)),
                distill_temp=float(cil_cfg.get("distill_temp", 2.0)),
                use_replay=True,
                use_rf_aug=bool(cil_cfg.get("use_rf_aug", True)),
                use_ss_aug=False,  # Controlled exclusively via discovery/ss_augment.py in Stage 4
                bic_max_steal=float(cil_cfg.get("bic_max_steal", 0.046)),
                use_weight_align=cl_cfg.use_weight_align,
                replay_ss_discovered=cl_cfg.replay_ss_discovered,
                use_amp=cl_cfg.use_amp,
            )
            y_t0, y_p0 = _predict_heads(model, known_test, device=device)
        elif cl_cfg.method_key == "naive":
            lwf_learner = NaiveFineTuneLearner(
                epochs=int(cil_cfg.get("epochs", 35)),
                batch_size_new=int(cil_cfg.get("batch_size_new", 128)),
                lr=float(cil_cfg.get("lr", 8e-4)),
                weight_decay=float(cil_cfg.get("weight_decay", 1e-4)),
                grad_clip=float(cil_cfg.get("grad_clip", 1.0)),
            )
            y_t0, y_p0 = _predict_heads(model, known_test, device=device)
        else:  # static
            y_t0, y_p0 = _predict_heads(model, known_test, device=device)

        s0_eval = evaluate_cil_session(
            y_true=y_t0, y_pred=y_p0, num_base_classes=num_base, current_true_classes=list(range(num_base))
        )
        base_acc_0 = s0_eval["a_old"]
        prev_per_class_accs = [s0_eval["per_class_acc"]]

        active_scorer: MahalanobisPrototypeScorer = copy.deepcopy(base_scorer)
        tau_low = self._calibrate_osr_threshold(
            model=model,
            scorer=active_scorer,
            val_known_x=val_known_ds.x,
            val_known_snrs=val_known_ds.snrs,
            snr_estimator=snr_estimator,
            cl_cfg=cl_cfg,
            device=device,
        )

        gate = DualAwareGate(gate_mode=cl_cfg.gate_mode, snr_threshold=cl_cfg.snr_threshold)
        discoverer = HDBSCANDiscoverer(
            min_cluster_size=cl_cfg.min_cluster_size,
            min_samples=cl_cfg.min_samples,
            silhouette_threshold=cl_cfg.silhouette_threshold,
            core_prob_threshold=cl_cfg.core_prob_threshold,
            core_top_ratio=cl_cfg.core_top_ratio,
            feature_mode="stem_preln",
            allow_single_cluster=cl_cfg.allow_single_cluster,
            merge_cosine_threshold=cl_cfg.merge_cosine_threshold,
            min_cluster_silhouette=cl_cfg.min_cluster_silhouette,
            min_cluster_cohesion=cl_cfg.min_cluster_cohesion,
        )
        buf = RejectionBuffer()
        seen_classes = list(bundle.known_classes)

        sessions_out: list[dict[str, Any]] = [
            {
                "session": 0,
                "num_classes": num_base,
                "model_heads": num_base,
                "arriving_classes": [],
                "discovery_accepted": True,
                "a_all_pct": round(s0_eval["a_all"] * 100.0, 2),
                "a_old_pct": round(s0_eval["a_old"] * 100.0, 2),
                "a_new_pct": None,
                "delta_a_old_pct": 0.0,
                "fm_cls_pct": 0.0,
                "per_class_acc_pct": {idx_to_name[c]: round(a * 100.0, 2) for c, a in s0_eval["per_class_acc"].items()},
            }
        ]

        for batch in sessions:
            prior_seen_true_ids = [bundle.all_to_idx[c] for c in seen_classes]
            arriving_true_ids = [bundle.all_to_idx[c] for c in batch.arriving_classes]
            seen_classes.extend(batch.arriving_classes)
            seen_true_ids = [bundle.all_to_idx[c] for c in seen_classes]

            # 1. OSR scoring & Blind SNR Dual-Aware Gating
            ms_batch = _extract_ms_np(model, batch.x_iq, device=device, include_physics=cl_cfg.include_physics)
            osr_scores = active_scorer.score_multiscale(ms_batch)
            snr_est: np.ndarray | None = None
            if cl_cfg.gate_mode in ("m2m4", "aux_snr"):
                assert snr_estimator is not None
                snr_est = _estimate_snr_np(snr_estimator, batch.x_iq, device=device)

            gate_rep = gate.filter(
                osr_scores=osr_scores,
                osr_threshold=tau_low,
                snr_est=snr_est,
                true_labels=batch.true_labels,
                true_snrs=batch.snrs,
                novel_class_ids=arriving_true_ids,
            )

            # Monitor Main-SNR prior-known False Rejection Rate on incoming stream
            prior_main_mask = np.isin(batch.true_labels, prior_seen_true_ids) & (batch.snrs >= 0)
            known_frr_stream_pct = (
                float((osr_scores[prior_main_mask] < tau_low).mean() * 100.0) if prior_main_mask.any() else 0.0
            )

            test_ds = bundle.get_dataset("test", seen_classes, bundle.main_snrs, bundle.all_to_idx)

            if cl_cfg.method_key == "static":
                yt, yp = _predict_heads(model, test_ds, device=device)
                cil_eval = evaluate_cil_session(
                    y_true=yt,
                    y_pred=yp,
                    num_base_classes=num_base,
                    current_true_classes=seen_true_ids,
                    prev_per_class_accs=prev_per_class_accs,
                    base_acc_0=base_acc_0,
                    head_to_true={},
                )
                prev_per_class_accs.append(cil_eval["per_class_acc"])
                rec = {
                    "session": batch.session_id,
                    "num_classes": len(seen_classes),
                    "model_heads": model.num_classes,
                    "arriving_classes": batch.arriving_classes,
                    "discovery_accepted": True,
                    "discovered_clusters": 0,
                    "delta_k": len(batch.arriving_classes),
                    "gating": {
                        "gate_mode": gate_rep.gate_mode,
                        "passed_count": gate_rep.passed_count,
                        "unknown_purity_pct": gate_rep.unknown_purity_pct,
                        "main_unknown_recall_pct": gate_rep.main_unknown_recall_pct,
                        "neg_snr_leakage_pct": gate_rep.neg_snr_leakage_pct,
                        "known_fp_ratio_pct": gate_rep.known_fp_ratio_pct,
                        "known_frr_on_stream_pct": round(known_frr_stream_pct, 2),
                    },
                    "a_all_pct": round(cil_eval["a_all"] * 100.0, 2),
                    "a_old_pct": round(cil_eval["a_old"] * 100.0, 2),
                    "a_new_pct": 0.0,
                    "delta_a_old_pct": round(cil_eval["delta_a_old"] * 100.0, 2),
                    "fm_cls_pct": round(cil_eval["fm_cls"] * 100.0, 2),
                    "per_class_acc_pct": {
                        idx_to_name[c]: round(a * 100.0, 2) for c, a in cil_eval["per_class_acc"].items()
                    },
                    "head_to_true": {},
                    "elapsed_sec": 0.0,
                }
                sessions_out.append(rec)
                logger.info(
                    "[ClosedLoop | %-7s | %-26s] S%d (C=%2d) | A_all=%.2f%% | A_old=%.2f%% | A_new=0.00%%",
                    order_name,
                    method_label,
                    batch.session_id,
                    len(seen_classes),
                    rec["a_all_pct"],
                    rec["a_old_pct"],
                )
                continue

            # 2. Buffer gated candidates & run HDBSCAN discovery
            if gate_rep.passed_count > 0:
                buf.add(
                    x_iq=batch.x_iq[gate_rep.passed_mask],
                    snr_est=snr_est[gate_rep.passed_mask] if snr_est is not None else None,
                    true_labels=batch.true_labels[gate_rep.passed_mask],
                )

            x_buf = buf.get_waveforms()
            y_buf = buf.get_true_labels()
            if len(x_buf) > 0:
                ms_buf = _extract_ms_np(model, x_buf, device=device, include_physics=cl_cfg.include_physics)
                disc = discoverer.discover(ms_buf)
                accepted = bool(disc.accepted and disc.k_hat > 0 and disc.core_mask.any())
            else:
                from os_se_amr.discovery.hdbscan_cluster import DiscoveryResult

                disc = DiscoveryResult(
                    accepted=False,
                    reason="empty_buffer",
                    cluster_ids=np.empty(0, dtype=np.int64),
                    membership_probs=np.empty(0, dtype=np.float32),
                    k_hat=0,
                    noise_ratio=1.0,
                    silhouette=-1.0,
                    per_cluster_silhouette={},
                    core_mask=np.empty(0, dtype=bool),
                    prototypes_64d=np.empty((0, model.feature_dim), dtype=np.float32),
                    cluster_sizes={},
                    cluster_ids_unique=[],
                )
                accepted = False

            core_purity_pct = 0.0
            core_raw_count = 0
            train_new_count = 0
            elapsed = 0.0
            old_proto_shift_pre_ln = None
            old_proto_shift_stem = None

            if accepted:
                assert y_buf is not None
                old_num_classes = model.num_classes
                x_core_raw = x_buf[disc.core_mask]
                y_core_local = disc.cluster_ids[disc.core_mask]
                y_core_true = y_buf[disc.core_mask]
                core_raw_count = int(len(x_core_raw))

                # Core purity wrt arriving + true classes
                core_diag = evaluate_clustering_diagnostics(y_core_true, y_core_local)
                core_purity_pct = float(core_diag["many_to_1_purity"] * 100.0)

                # 3. Intra-cluster TLDNN-SS expansion (quality/size-adaptive inside each discovered cluster)
                x_train_new, y_train_new = augment_core_clusters(
                    x_core=x_core_raw,
                    cluster_labels=y_core_local,
                    mode=cl_cfg.ss_mode,
                    expand_factor=cl_cfg.ss_expand_factor,
                    max_samples_per_cluster=cl_cfg.ss_max_per_cluster,
                    seed=seed + batch.session_id,
                    per_cluster_silhouette=disc.per_cluster_silhouette,
                    per_cluster_cohesion=disc.cluster_cohesions,
                    min_cluster_silhouette=cl_cfg.ss_min_cluster_silhouette,
                    min_cluster_cohesion=cl_cfg.ss_min_cluster_cohesion,
                    max_core_size_for_ss=cl_cfg.ss_max_core_size,
                )
                train_new_count = int(len(x_train_new))

                # 4. Continual Learner Evolution
                if cl_cfg.method_key == "icarl":
                    assert icarl is not None
                    evo = icarl.evolve(
                        model, replay_buf, x_train_new, y_train_new, device=device, seed=seed + batch.session_id
                    )
                    elapsed = evo.elapsed_sec
                    ms_upd = _extract_ms_np(model, x_core_raw, device=device)
                    for cid in disc.cluster_ids_unique:
                        c_mask = y_core_local == cid
                        replay_buf.add_class_exemplars(
                            class_id=old_num_classes + cid,
                            x_iq=x_core_raw[c_mask],
                            features=ms_upd["pre_ln"][c_mask],
                            snrs=None,
                            seed=seed + batch.session_id,
                        )
                    icarl.update_class_means(model, replay_buf, device=device)
                elif cl_cfg.method_key in ("ewc", "wa", "der", "ours"):
                    if cl_cfg.method_key == "ewc":
                        assert isinstance(extra_learner, EWCLearner)
                        evo = extra_learner.evolve(
                            model, replay_buf, x_train_new, y_train_new, device=device, seed=seed + batch.session_id
                        )
                    elif cl_cfg.method_key == "wa":
                        assert isinstance(extra_learner, WALearner)
                        evo = extra_learner.evolve(
                            model, replay_buf, x_train_new, y_train_new, device=device, seed=seed + batch.session_id
                        )
                    elif cl_cfg.method_key == "der":
                        assert isinstance(extra_learner, DERLearner)
                        model, evo = extra_learner.evolve(
                            model,
                            replay_buf,
                            x_train_new,
                            y_train_new,
                            model_cfg=model_cfg,
                            device=device,
                            seed=seed + batch.session_id,
                        )
                    else:
                        assert isinstance(lwf_learner, DualLwFLearner)
                        evo = lwf_learner.evolve(
                            backbone=model,
                            replay_buffer=replay_buf,
                            new_samples=x_train_new,
                            new_pseudo_labels=y_train_new,
                            prototypes_64d=None,
                            sample_weights=None,
                            val_known_loader=val_known_loader,
                            num_base_classes=num_base,
                            device=device,
                            seed=seed + batch.session_id,
                        )
                    elapsed = evo.elapsed_sec
                    ms_upd = _extract_ms_np(model, x_core_raw, device=device, include_physics=cl_cfg.include_physics)
                    for cid in disc.cluster_ids_unique:
                        c_mask = y_core_local == cid
                        replay_buf.add_class_exemplars(
                            class_id=old_num_classes + cid,
                            x_iq=x_core_raw[c_mask],
                            features=ms_upd["pre_ln"][c_mask],
                            snrs=None,  # Strict no-oracle-SNR rule
                            seed=seed + batch.session_id,
                        )
                    if cl_cfg.sync_prototypes:
                        sync_info = sync_prototypes_and_mahalanobis(
                            backbone=model,
                            replay_buffer=replay_buf,
                            new_core_x=x_core_raw,
                            new_core_labels=y_core_local,
                            prev_scorer=active_scorer,
                            device=device,
                            num_old_classes=old_num_classes,
                            include_physics=cl_cfg.include_physics,
                        )
                        active_scorer = sync_info["scorer"]
                        old_proto_shift_pre_ln = round(float(sync_info["old_proto_shift_pre_ln"]), 4)
                        old_proto_shift_stem = round(float(sync_info["old_proto_shift_stem"]), 4)
                        tau_low = self._calibrate_osr_threshold(
                            model=model,
                            scorer=active_scorer,
                            val_known_x=val_known_ds.x,
                            val_known_snrs=val_known_ds.snrs,
                            snr_estimator=snr_estimator,
                            cl_cfg=cl_cfg,
                            device=device,
                            extra_core_x=x_core_raw,
                        )
                else:  # naive
                    assert lwf_learner is not None
                    evo = lwf_learner.evolve(
                        backbone=model,
                        replay_buffer=replay_buf,
                        new_samples=x_train_new,
                        new_pseudo_labels=y_train_new,
                        prototypes_64d=None,
                        sample_weights=None,
                        val_known_loader=None,
                        num_base_classes=num_base,
                        device=device,
                        seed=seed + batch.session_id,
                    )
                    elapsed = evo.elapsed_sec

                buf.clear()
            else:
                logger.warning(
                    "[ClosedLoop | %-7s | %-26s] S%d discovery rejected (%s, K_hat=%d, buf=%d) -> holding model & buffer",
                    order_name,
                    method_label,
                    batch.session_id,
                    disc.reason,
                    disc.k_hat,
                    len(x_buf),
                )

            if cl_cfg.method_key == "icarl":
                assert icarl is not None
                yt, yp = icarl.predict_nme(model, test_ds, device=device)
            else:
                yt, yp = _predict_heads(model, test_ds, device=device)

            cil_eval = evaluate_cil_session(
                y_true=yt,
                y_pred=yp,
                num_base_classes=num_base,
                current_true_classes=seen_true_ids,
                prev_per_class_accs=prev_per_class_accs,
                base_acc_0=base_acc_0,
            )
            prev_per_class_accs.append(cil_eval["per_class_acc"])
            per_cls_named = {idx_to_name[c]: round(a * 100.0, 2) for c, a in cil_eval["per_class_acc"].items()}

            is_zero_novelty = len(batch.arriving_classes) == 0
            rec = {
                "session": batch.session_id,
                "num_classes": len(seen_classes),
                "model_heads": model.num_classes,
                "arriving_classes": batch.arriving_classes,
                "discovery_accepted": accepted,
                "discovery_reason": disc.reason,
                "false_discovery_triggered": bool(accepted and is_zero_novelty),
                "false_heads_added": int(disc.k_hat if (accepted and is_zero_novelty) else 0),
                "raw_discovered_clusters": int(disc.raw_k_hat),
                "discovered_clusters": disc.k_hat,
                "delta_k": abs(int(disc.k_hat) - len(batch.arriving_classes)),
                "discovery_silhouette": round(float(disc.silhouette), 4),
                "cluster_cohesions": {str(k): round(float(v), 4) for k, v in disc.cluster_cohesions.items()},
                "core_samples_raw": core_raw_count,
                "train_samples_augmented": train_new_count,
                "core_purity_pct": round(core_purity_pct, 2),
                "gating": {
                    "gate_mode": gate_rep.gate_mode,
                    "passed_count": gate_rep.passed_count,
                    "unknown_purity_pct": gate_rep.unknown_purity_pct,
                    "main_unknown_recall_pct": gate_rep.main_unknown_recall_pct,
                    "neg_snr_leakage_pct": gate_rep.neg_snr_leakage_pct,
                    "known_fp_ratio_pct": gate_rep.known_fp_ratio_pct,
                    "known_frr_on_stream_pct": round(known_frr_stream_pct, 2),
                },
                "a_all_pct": round(cil_eval["a_all"] * 100.0, 2),
                "a_old_pct": round(cil_eval["a_old"] * 100.0, 2),
                "a_new_pct": round((cil_eval["a_new"] or 0.0) * 100.0, 2),
                "delta_a_old_pct": round(cil_eval["delta_a_old"] * 100.0, 2),
                "fm_cls_pct": round(cil_eval["fm_cls"] * 100.0, 2),
                "per_class_acc_pct": per_cls_named,
                "head_to_true": cil_eval["head_to_true"],
                "old_proto_shift_pre_ln": old_proto_shift_pre_ln,
                "old_proto_shift_stem": old_proto_shift_stem,
                "elapsed_sec": elapsed,
            }
            sessions_out.append(rec)
            logger.info(
                "[ClosedLoop | %-7s | %-26s] S%d (C=%2d) | pass=%d (pur=%.1f%%, neg=%.1f%%, FRR=%.1f%%) | acc=%s K=%d(raw=%d) | A_all=%.2f%% A_old=%.2f%% A_new=%.2f%% dA=%.2f%%",
                order_name,
                method_label,
                batch.session_id,
                len(seen_classes),
                gate_rep.passed_count,
                gate_rep.unknown_purity_pct,
                gate_rep.neg_snr_leakage_pct,
                known_frr_stream_pct,
                accepted,
                disc.k_hat,
                disc.raw_k_hat,
                rec["a_all_pct"],
                rec["a_old_pct"],
                rec["a_new_pct"],
                rec["delta_a_old_pct"],
            )

        if save_final_ckpt is not None:
            save_final_ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": model.num_classes,
                    "scorer_state": active_scorer.state_dict(),
                    "tau_low": float(tau_low),
                    "sessions": sessions_out,
                },
                save_final_ckpt,
            )

        return {
            "order": order_name,
            "method": method_label,
            "config": {
                "method_key": cl_cfg.method_key,
                "gate_mode": cl_cfg.gate_mode,
                "ss_mode": cl_cfg.ss_mode,
                "sync_prototypes": cl_cfg.sync_prototypes,
                "merge_cosine_threshold": cl_cfg.merge_cosine_threshold,
                "ss_max_core_size": cl_cfg.ss_max_core_size,
            },
            "sessions": sessions_out,
        }
