"""RadioML2018.01a HDF5 loader, compact stratified cache builder, and unified dataset dispatcher."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from os_se_amr.data.rml2016 import load_rml2016_dict
from os_se_amr.utils.config import resolve_path

RML2018_CLASSES: tuple[str, ...] = (
    "OOK",
    "4ASK",
    "8ASK",
    "BPSK",
    "QPSK",
    "8PSK",
    "16PSK",
    "32PSK",
    "16APSK",
    "32APSK",
    "64APSK",
    "128APSK",
    "16QAM",
    "32QAM",
    "64QAM",
    "128QAM",
    "256QAM",
    "AM-SSB-WC",
    "AM-SSB-SC",
    "AM-DSB-WC",
    "AM-DSB-SC",
    "FM",
    "GMSK",
    "OQPSK",
)


def _require_h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required to read/write RML2018.01a HDF5 datasets (`pip install h5py`).") from exc
    return h5py


def _read_selected_rows(dset: Any, sorted_indices: np.ndarray, block_size: int = 8192) -> np.ndarray:
    """Memory-safe HDF5 read supporting contiguous slices cleanly with chunk fallback."""
    if len(sorted_indices) == 0:
        return np.empty((0, 2, dset.shape[1]), dtype=np.float32)

    lo = int(sorted_indices[0])
    hi = int(sorted_indices[-1])
    span = hi - lo + 1
    if span == len(sorted_indices):
        raw = np.asarray(dset[lo : hi + 1])
        return np.ascontiguousarray(np.transpose(raw, (0, 2, 1)), dtype=np.float32)
    if span <= max(16384, 4 * len(sorted_indices)):
        raw = np.asarray(dset[lo : hi + 1])[sorted_indices - lo]
        return np.ascontiguousarray(np.transpose(raw, (0, 2, 1)), dtype=np.float32)

    parts: list[np.ndarray] = []
    for start in range(0, len(sorted_indices), block_size):
        idx_chunk = sorted_indices[start : start + block_size]
        raw = np.asarray(dset[idx_chunk])
        parts.append(np.transpose(raw, (0, 2, 1)))
    return np.ascontiguousarray(np.concatenate(parts, axis=0), dtype=np.float32)


def load_rml2018_hdf5(
    path: str | Path,
    max_per_cell: int | None = 1000,
    snrs: list[int] | None = None,
    seed: int = 42,
    classes: tuple[str, ...] | list[str] = RML2018_CLASSES,
) -> dict[tuple[str, int], np.ndarray]:
    """Load RML2018.01a from raw DeepSig HDF5 or pre-built compact `(mod, snr)` cache into float32 (N, 2, L)."""
    h5py = _require_h5py()
    resolved = resolve_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"HDF5 dataset not found: {resolved}")

    target_snrs = None if snrs is None else {int(s) for s in snrs}
    rng = np.random.default_rng(int(seed))
    out: dict[tuple[str, int], np.ndarray] = {}

    with h5py.File(resolved, "r") as f:
        if "cells" in f:
            cells_grp = f["cells"]
            for mod in sorted(cells_grp.keys()):
                mod_grp = cells_grp[mod]
                for snr_str in sorted(mod_grp.keys(), key=lambda s: int(s)):
                    snr_val = int(snr_str)
                    if target_snrs is not None and snr_val not in target_snrs:
                        continue
                    arr = np.asarray(mod_grp[snr_str], dtype=np.float32)
                    if max_per_cell is not None and len(arr) > int(max_per_cell):
                        chosen = np.sort(rng.choice(len(arr), size=int(max_per_cell), replace=False))
                        arr = arr[chosen]
                    if arr.ndim != 3 or arr.shape[1] != 2:
                        raise ValueError(f"Unexpected shape {arr.shape} in cache cell {(mod, snr_val)}")
                    out[(str(mod), snr_val)] = np.ascontiguousarray(arr, dtype=np.float32)
            return out

        if "X" not in f or "Y" not in f or "Z" not in f:
            raise KeyError(f"Expected ('X','Y','Z') or 'cells' group in {resolved}, got {list(f.keys())}")

        x_dset, y_dset, z_dset = f["X"], f["Y"], f["Z"]
        n_total = int(x_dset.shape[0])
        num_cls = int(y_dset.shape[1])
        cls_names = [str(c) for c in classes]
        if len(cls_names) != num_cls:
            raise ValueError(f"Expected {num_cls} class names, got {len(cls_names)}")

        # Pass 1: index lightweight labels & SNRs without touching 21 GB waveform tensor
        cell_indices: dict[tuple[int, int], list[np.ndarray]] = {}
        scan_blk = 65536
        for start in range(0, n_total, scan_blk):
            end = min(n_total, start + scan_blk)
            y_blk = np.argmax(np.asarray(y_dset[start:end]), axis=1)
            z_blk = np.asarray(z_dset[start:end]).reshape(-1).astype(np.int64)
            global_idx = np.arange(start, end, dtype=np.int64)
            pairs = np.stack([y_blk, z_blk], axis=1)
            uniq_pairs, inv = np.unique(pairs, axis=0, return_inverse=True)
            for p_idx, (c_id, snr_v) in enumerate(uniq_pairs):
                snr_int = int(snr_v)
                if target_snrs is not None and snr_int not in target_snrs:
                    continue
                cell_indices.setdefault((int(c_id), snr_int), []).append(global_idx[inv == p_idx])

        # Pass 2: stratified subsample per cell & read waveforms in contiguous slices when possible
        for (c_id, snr_int) in sorted(cell_indices.keys()):
            idx_all = np.concatenate(cell_indices[(c_id, snr_int)], axis=0)
            if max_per_cell is not None and len(idx_all) > int(max_per_cell):
                chosen_pos = np.sort(rng.choice(len(idx_all), size=int(max_per_cell), replace=False))
                idx_all = idx_all[chosen_pos]
            out[(cls_names[c_id], snr_int)] = _read_selected_rows(x_dset, idx_all)

    return out


def save_rml2018_cache(
    raw_dict: dict[tuple[str, int], np.ndarray],
    out_path: str | Path,
    dtype: str = "float16",
) -> Path:
    """Save stratified `(mod, snr) -> (N, 2, L)` dict into compact HDF5 cache for Colab/Drive."""
    h5py = _require_h5py()
    target = resolve_path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np_dtype = np.float16 if dtype == "float16" else np.float32

    with h5py.File(target, "w") as f:
        cells = f.create_group("cells")
        for (mod, snr), arr in sorted(raw_dict.items()):
            grp = cells.require_group(str(mod))
            grp.create_dataset(str(int(snr)), data=np.asarray(arr, dtype=np_dtype), compression="lzf")
        f.attrs["format"] = "os_se_amr_cells_v1"
        f.attrs["dtype"] = dtype
    return target


def load_dataset_dict(
    path: str | Path,
    max_per_cell: int | None = None,
    snrs: list[int] | None = None,
    seed: int = 42,
) -> dict[tuple[str, int], np.ndarray]:
    """Unified loader dispatching `.h5`/`.hdf5` (RML2018) and `.pkl`/`.dat` (RML2016.10a/b)."""
    resolved = resolve_path(path)
    suffix = resolved.suffix.lower()
    if suffix in (".h5", ".hdf5"):
        return load_rml2018_hdf5(resolved, max_per_cell=max_per_cell, snrs=snrs, seed=seed)

    raw_dict = load_rml2016_dict(resolved)
    target_snrs = None if snrs is None else {int(s) for s in snrs}
    if target_snrs is None and max_per_cell is None:
        return raw_dict

    rng = np.random.default_rng(int(seed))
    filtered: dict[tuple[str, int], np.ndarray] = {}
    for key in sorted(raw_dict.keys()):
        mod, snr = key
        if target_snrs is not None and snr not in target_snrs:
            continue
        arr = raw_dict[key]
        if max_per_cell is not None and len(arr) > int(max_per_cell):
            chosen = np.sort(rng.choice(len(arr), size=int(max_per_cell), replace=False))
            arr = arr[chosen]
        filtered[key] = np.ascontiguousarray(arr, dtype=np.float32)
    return filtered
