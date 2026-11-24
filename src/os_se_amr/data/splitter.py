"""Leakage-free stratified open-set dataset splitter and openness calculations."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from os_se_amr.data.rml2016 import RMLDataset
from os_se_amr.data.rml2018 import load_dataset_dict


def compute_openness(c_train: int, c_target: int) -> float:
    """Compute Scheirer's Openness score: 1 - sqrt(2 * C_train / (C_target + C_train))."""
    if c_train <= 0:
        raise ValueError(f"c_train must be positive, got {c_train}")
    if c_target < c_train:
        raise ValueError(f"c_target ({c_target}) must be >= c_train ({c_train})")
    return 1.0 - math.sqrt((2.0 * c_train) / (c_target + c_train))


class SplitBundle:
    """Deterministic, zero-leakage train/val/test partition over (mod, snr) cells.

    Locked-known indexing invariant:
    - Known classes occupy indices 0 .. C_train - 1 in both known_to_idx and all_to_idx.
    - Unknown classes occupy indices C_train .. C_target - 1 in all_to_idx.
    """

    def __init__(
        self,
        raw_dict: dict[tuple[str, int], np.ndarray],
        split_cfg: dict[str, Any],
    ) -> None:
        self.raw_dict = raw_dict
        self.split_cfg = split_cfg
        self.name = str(split_cfg.get("name", "custom"))
        self.seed = int(split_cfg.get("seed", 42))

        self.known_classes = [str(c) for c in split_cfg["known_classes"]]
        self.unknown_classes = [str(c) for c in split_cfg["unknown_classes"]]
        self.all_classes = self.known_classes + self.unknown_classes

        overlap = set(self.known_classes) & set(self.unknown_classes)
        if overlap:
            raise ValueError(f"Known and unknown classes must be disjoint, overlap={overlap}")
        if len(set(self.all_classes)) != len(self.all_classes):
            raise ValueError("Duplicate class names in known_classes or unknown_classes")

        available_mods = {mod for mod, _ in raw_dict.keys()}
        missing = set(self.all_classes) - available_mods
        if missing:
            raise ValueError(f"Classes missing from dataset: {sorted(missing)}")

        self.known_to_idx = {cls: i for i, cls in enumerate(self.known_classes)}
        self.all_to_idx = {cls: i for i, cls in enumerate(self.all_classes)}

        all_snrs = sorted({snr for _, snr in raw_dict.keys()})
        snr_min = int(split_cfg.get("snr_min", 0))
        snr_max = int(split_cfg.get("snr_max", 18))
        self.main_snrs = [s for s in all_snrs if snr_min <= s <= snr_max]
        self.stress_snrs = [int(s) for s in split_cfg.get("stress_snrs", []) if s in all_snrs]
        if not self.main_snrs:
            raise ValueError(f"No SNRs found in [{snr_min}, {snr_max}] among {all_snrs}")

        ratios = split_cfg.get("ratios", {"train": 0.6, "val": 0.2, "test": 0.2})
        r_train = float(ratios["train"])
        r_val = float(ratios["val"])
        r_test = float(ratios["test"])
        if not math.isclose(r_train + r_val + r_test, 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError(f"Split ratios must sum to 1.0, got {r_train + r_val + r_test}")
        self.ratios = {"train": r_train, "val": r_val, "test": r_test}

        self.openness = compute_openness(len(self.known_classes), len(self.all_classes))
        self.indices: dict[tuple[str, int], dict[str, np.ndarray]] = {}
        self._build_indices()

    @classmethod
    def from_config(cls, split_cfg: dict[str, Any]) -> "SplitBundle":
        """Load dataset (.pkl/.dat or .h5/.hdf5) from split_cfg['dataset_path'] and construct SplitBundle."""
        max_per_cell = split_cfg.get("max_per_cell")
        seed = int(split_cfg.get("seed", 42))
        raw_dict = load_dataset_dict(
            split_cfg["dataset_path"],
            max_per_cell=int(max_per_cell) if max_per_cell is not None else None,
            seed=seed,
        )
        return cls(raw_dict=raw_dict, split_cfg=split_cfg)

    def _build_indices(self) -> None:
        rng = np.random.default_rng(self.seed)
        for key in sorted(self.raw_dict.keys()):
            n = self.raw_dict[key].shape[0]
            perm = rng.permutation(n)
            n_train = int(round(n * self.ratios["train"]))
            n_val = int(round(n * self.ratios["val"]))
            idx_train = np.sort(perm[:n_train])
            idx_val = np.sort(perm[n_train : n_train + n_val])
            idx_test = np.sort(perm[n_train + n_val :])
            self.indices[key] = {
                "train": idx_train,
                "val": idx_val,
                "test": idx_test,
            }

    def verify_disjoint(self) -> bool:
        """Assert zero index overlap and full sample coverage in every (mod, snr) cell."""
        for key, arr in self.raw_dict.items():
            splits = self.indices[key]
            s_train = set(splits["train"].tolist())
            s_val = set(splits["val"].tolist())
            s_test = set(splits["test"].tolist())
            if (s_train & s_val) or (s_train & s_test) or (s_val & s_test):
                raise AssertionError(f"Data leakage detected in cell {key}")
            if len(s_train) + len(s_val) + len(s_test) != arr.shape[0]:
                raise AssertionError(f"Incomplete split coverage in cell {key}")
        return True

    def get_dataset(
        self,
        split: str,
        classes: list[str],
        snrs: list[int] | None = None,
        label_map: dict[str, int] | None = None,
    ) -> RMLDataset:
        """Build an RMLDataset for a given partition ('train'|'val'|'test'), classes, and SNRs."""
        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split '{split}'; expected 'train', 'val', or 'test'")
        target_snrs = self.main_snrs if snrs is None else [int(s) for s in snrs]
        mapping = {c: i for i, c in enumerate(classes)} if label_map is None else label_map

        xs, ys, ss = [], [], []
        for mod in classes:
            if mod not in mapping:
                raise KeyError(f"Class '{mod}' missing in label_map")
            label_id = mapping[mod]
            for snr in target_snrs:
                key = (mod, snr)
                if key not in self.raw_dict:
                    continue
                idx = self.indices[key][split]
                if len(idx) == 0:
                    continue
                xs.append(self.raw_dict[key][idx])
                ys.append(np.full(len(idx), label_id, dtype=np.int64))
                ss.append(np.full(len(idx), snr, dtype=np.int64))

        if not xs:
            raise ValueError(f"Empty dataset slice for split={split}, classes={classes}, snrs={target_snrs}")

        return RMLDataset(
            x=np.concatenate(xs, axis=0),
            y=np.concatenate(ys, axis=0),
            snrs=np.concatenate(ss, axis=0),
            class_names=classes,
        )

    def get_closed_set_datasets(self) -> tuple[RMLDataset, RMLDataset, RMLDataset]:
        """Return (train_ds, val_ds, test_ds) over known classes and main_snrs."""
        return (
            self.get_dataset("train", self.known_classes, self.main_snrs, self.known_to_idx),
            self.get_dataset("val", self.known_classes, self.main_snrs, self.known_to_idx),
            self.get_dataset("test", self.known_classes, self.main_snrs, self.known_to_idx),
        )

    def get_open_set_test_datasets(self) -> tuple[RMLDataset, RMLDataset]:
        """Return (known_test_ds, unknown_test_ds) using locked-known all_to_idx labels."""
        known_test = self.get_dataset("test", self.known_classes, self.main_snrs, self.all_to_idx)
        unknown_test = self.get_dataset("test", self.unknown_classes, self.main_snrs, self.all_to_idx)
        return known_test, unknown_test

    def get_stress_test_dataset(self, classes: list[str] | None = None) -> RMLDataset:
        """Return held-out low/negative-SNR test split for robustness evaluation."""
        if not self.stress_snrs:
            raise ValueError("No stress_snrs configured in split config")
        target_classes = self.known_classes if classes is None else classes
        label_map = self.known_to_idx if target_classes == self.known_classes else self.all_to_idx
        return self.get_dataset("test", target_classes, self.stress_snrs, label_map)

    def summary(self) -> dict[str, Any]:
        """Return concise metadata summary for logging and artifacts."""
        return {
            "name": self.name,
            "seed": self.seed,
            "c_train": len(self.known_classes),
            "c_unknown": len(self.unknown_classes),
            "c_target": len(self.all_classes),
            "openness": round(self.openness, 6),
            "openness_pct": round(self.openness * 100.0, 2),
            "known_classes": self.known_classes,
            "unknown_classes": self.unknown_classes,
            "main_snrs": self.main_snrs,
            "stress_snrs": self.stress_snrs,
            "ratios": self.ratios,
        }
