"""Stage 0 entry point: closed-set baseline training & feature validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from os_se_amr.data.splitter import SplitBundle
from os_se_amr.engine.trainer import ClosedSetTrainer
from os_se_amr.models import build_backbone
from os_se_amr.utils import (
    load_stage_config,
    load_yaml,
    resolve_path,
    save_json,
    save_yaml,
    seed_everything,
    setup_logger,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 0: Closed-Set Baseline Training & Validation")
    parser.add_argument("--config", type=str, default="configs/stage0_base.yaml", help="Path to Stage 0 YAML")
    parser.add_argument("--model", type=str, default=None, choices=["tldnn", "iqformer"], help="Override backbone")
    parser.add_argument("--split-config", type=str, default=None, help="Override split YAML path")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--device", type=str, default=None, help="Override device (cuda or cpu)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_stage_config(args.config)

    if args.split_config is not None:
        cfg["split_config"] = args.split_config
        cfg["split"] = load_yaml(args.split_config)
    if args.model is not None:
        cfg["model"]["name"] = args.model
    if args.epochs is not None:
        cfg["train"]["epochs"] = int(args.epochs)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
        cfg["split"]["seed"] = int(args.seed)
    if args.device is not None:
        cfg["device"] = args.device

    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    req_device = str(cfg.get("device", "cuda"))
    if req_device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(req_device)

    model_name = str(cfg["model"]["name"]).lower()
    split_name = str(cfg["split"].get("name", "custom"))
    exp_id = f"{model_name}_{split_name}_seed{seed}"
    out_dir = resolve_path(cfg.get("output_dir", "outputs/stage0")) / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(f"stage0.{exp_id}", log_file=out_dir / "train.log")
    logger.info("Starting Stage 0 experiment '%s' on device=%s", exp_id, device)
    if device.type == "cuda":
        logger.info("GPU device: %s", torch.cuda.get_device_name(device))

    save_yaml(cfg, out_dir / "config.yaml")

    bundle = SplitBundle.from_config(cfg["split"])
    bundle.verify_disjoint()
    split_summary = bundle.summary()
    logger.info(
        "Split summary: name=%s | C_train=%d | C_target=%d | Openness=%.2f%%",
        split_summary["name"],
        split_summary["c_train"],
        split_summary["c_target"],
        split_summary["openness_pct"],
    )

    train_ds, val_ds, test_ds = bundle.get_closed_set_datasets()
    batch_size = int(cfg["train"].get("batch_size", 256))
    num_workers = int(cfg["train"].get("num_workers", 0))
    pin_mem = device.type == "cuda"

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_mem
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem
    )
    logger.info("Dataset sizes: train=%d, val=%d, test=%d", len(train_ds), len(val_ds), len(test_ds))

    model = build_backbone(cfg["model"], num_classes=len(bundle.known_classes))
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model=%s | trainable_params=%d | feature_dim=%d", model_name, num_params, model.feature_dim)

    trainer = ClosedSetTrainer(
        model=model,
        device=device,
        train_cfg=cfg["train"],
        output_dir=out_dir,
        logger=logger,
    )
    train_summary = trainer.fit(train_loader, val_loader, bundle.known_classes)
    val_metrics = trainer.evaluate(val_loader, bundle.known_classes)
    test_metrics = trainer.evaluate(test_loader, bundle.known_classes)

    trainer.save_figures(test_metrics, bundle.known_classes, title_prefix=f"{model_name.upper()} ({split_name})")

    logger.info(
        "Final Test Acc: %.2f%% | Feature dim=%d | NaN=%s Inf=%s | MeanNorm=%.3f (min_std=%.4f)",
        test_metrics["overall_acc"] * 100.0,
        test_metrics["feature_stats"]["dim"],
        test_metrics["feature_stats"]["has_nan"],
        test_metrics["feature_stats"]["has_inf"],
        test_metrics["feature_stats"]["mean_norm"],
        test_metrics["feature_stats"]["min_dim_std"],
    )
    for cls_name, acc in test_metrics["per_class_acc"].items():
        logger.info("  Class %-6s : %.2f%%", cls_name, acc * 100.0)

    metrics_payload = {
        "experiment_id": exp_id,
        "model_name": model_name,
        "num_params": num_params,
        "device": str(device),
        "split": split_summary,
        "dataset_sizes": {
            "train": len(train_ds),
            "val": len(val_ds),
            "test": len(test_ds),
        },
        "train_summary": train_summary,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    save_json(metrics_payload, out_dir / "metrics.json")
    logger.info("Saved artifacts to %s", out_dir)


if __name__ == "__main__":
    main()
