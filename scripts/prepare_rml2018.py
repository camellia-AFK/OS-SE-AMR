"""Convert 21 GB raw RML2018.01a HDF5 into compact float16 stratified cache for Colab/Drive."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from os_se_amr.data.rml2018 import load_rml2018_hdf5, save_rml2018_cache
from os_se_amr.utils import resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compact float16 stratified RML2018.01a cache")
    parser.add_argument(
        "--input",
        type=str,
        default="data/GOLD_XYZ_OSC.0001_1024.hdf5",
        help="Path to raw DeepSig GOLD_XYZ_OSC.0001_1024.hdf5",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/rml2018_cache_1000.h5",
        help="Destination compact HDF5 cache path",
    )
    parser.add_argument("--max-per-cell", type=int, default=1000, help="Samples per (mod, snr) cell")
    parser.add_argument("--seed", type=int, default=42, help="Stratified sampling seed")
    parser.add_argument("--min-snr", type=int, default=-10, help="Minimum SNR to include")
    parser.add_argument("--max-snr", type=int, default=30, help="Maximum SNR to include")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snrs = [s for s in range(int(args.min_snr), int(args.max_snr) + 1, 2)]
    print(f"Scanning & subsampling {args.input} (max_per_cell={args.max_per_cell}, snrs={snrs[0]}..{snrs[-1]})...", flush=True)
    raw_dict = load_rml2018_hdf5(
        resolve_path(args.input),
        max_per_cell=int(args.max_per_cell),
        snrs=snrs,
        seed=int(args.seed),
    )
    out_path = save_rml2018_cache(raw_dict, resolve_path(args.output), dtype="float16")
    size_gb = out_path.stat().st_size / 1e9
    total_frames = sum(len(v) for v in raw_dict.values())
    print(f"Saved {len(raw_dict)} cells ({total_frames} frames) to {out_path} ({size_gb:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
