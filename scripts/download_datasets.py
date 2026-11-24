"""Resumable downloader for RadioML2016.10b and RadioML2018.01a supporting curl and Python backends."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request

MIRRORS = {
    "hf-mirror": "https://hf-mirror.com/datasets/FlowVortex/RML/resolve/main",
    "hf": "https://huggingface.co/datasets/FlowVortex/RML/resolve/main",
}

DATASETS = {
    "rml2016b": "RML2016.10b.dat",
    "rml2018": "GOLD_XYZ_OSC.0001_1024.hdf5",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download RML2016.10b and RML2018.01a with auto-resume")
    parser.add_argument(
        "--datasets",
        type=str,
        default="rml2016b,rml2018",
        help="Comma-separated subset of {rml2016b,rml2018}",
    )
    parser.add_argument(
        "--mirror",
        type=str,
        choices=sorted(MIRRORS.keys()),
        default="hf-mirror",
        help="Hugging Face mirror base URL",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["auto", "curl", "python"],
        default="auto",
        help="Transfer backend ('curl' recommended on Windows/Colab; 'auto' picks curl if present)",
    )
    parser.add_argument("--output-dir", type=str, default="data", help="Destination directory")
    parser.add_argument("--workers", type=int, default=2, help="Concurrent chunk workers for python backend")
    parser.add_argument("--chunk-mb", type=int, default=4, help="Chunk size in MiB for python backend")
    parser.add_argument("--retries", type=int, default=10, help="Max retries per transfer/chunk")
    return parser.parse_args()


def _probe_remote_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "OS-SE-AMR-Downloader/1.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        length = resp.headers.get("Content-Length")
    if not length or not length.isdigit():
        raise RuntimeError(f"Missing Content-Length header from {url}: {length!r}")
    return int(length)


def _fetch_range(url: str, start: int, end: int, retries: int) -> bytes:
    expected = end - start + 1
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "OS-SE-AMR-Downloader/1.0",
                    "Range": f"bytes={start}-{end}",
                },
            )
            with urllib.request.urlopen(req, timeout=45) as resp:
                if resp.status not in (200, 206):
                    raise RuntimeError(f"Unexpected HTTP status {resp.status}")
                buf = bytearray()
                while len(buf) < expected:
                    block = resp.read(min(1024 * 1024, expected - len(buf)))
                    if not block:
                        break
                    buf.extend(block)
            if len(buf) != expected:
                raise RuntimeError(f"Short read {len(buf)} != {expected} bytes for {start}-{end}")
            return bytes(buf)
        except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as exc:
            last_err = exc
            if attempt + 1 < retries:
                time.sleep(min(20.0, 1.5 * (2**attempt)))
    raise RuntimeError(f"Failed chunk {start}-{end} after {retries} retries: {last_err}")


def _verify_file_signature(key: str, path: Path) -> None:
    with path.open("rb") as f:
        header = f.read(8)
    if key == "rml2018":
        if header != b"\x89HDF\r\n\x1a\n":
            raise RuntimeError(f"Invalid HDF5 signature in {path}: {header!r}")
    elif key == "rml2016b":
        if not header or header[0] not in (0x80, ord("("), ord("{")):
            raise RuntimeError(f"Unexpected pickle header in {path}: {header!r}")


def _save_meta(meta_path: Path, total_bytes: int, chunk_size: int, completed: set[int]) -> None:
    payload = json.dumps(
        {
            "total_bytes": total_bytes,
            "chunk_size": chunk_size,
            "completed_chunks": sorted(completed),
        }
    )
    for attempt in range(5):
        try:
            meta_path.write_text(payload, encoding="utf-8")
            return
        except PermissionError:
            if attempt == 4:
                return
            time.sleep(0.15 * (attempt + 1))


def _download_with_curl(
    curl_exe: str,
    filename: str,
    url: str,
    part_path: Path,
    total_bytes: int,
    retries: int,
) -> None:
    if part_path.exists() and part_path.stat().st_size > total_bytes:
        part_path.unlink()

    cmd = [
        curl_exe,
        "-L",
        "-C",
        "-",
        "--retry",
        "10",
        "--retry-delay",
        "5",
        "--speed-time",
        "30",
        "--speed-limit",
        "20000",
        "--connect-timeout",
        "30",
        "--silent",
        "--show-error",
        "-o",
        str(part_path),
        url,
    ]
    print(f"[{filename}] Starting resumable curl transfer ({total_bytes / 1e9:.2f} GB) -> {part_path.name}", flush=True)
    t_start = time.time()
    start_bytes = part_path.stat().st_size if part_path.exists() else 0
    max_rounds = max(retries, 50)
    for round_idx in range(1, max_rounds + 1):
        proc = subprocess.Popen(cmd)
        while proc.poll() is None:
            time.sleep(10.0)
            cur_bytes = part_path.stat().st_size if part_path.exists() else 0
            elapsed = max(time.time() - t_start, 1e-3)
            delta_mb = max(cur_bytes - start_bytes, 0) / (1024 * 1024)
            speed = delta_mb / elapsed
            pct = 100.0 * cur_bytes / max(total_bytes, 1)
            eta_min = (
                ((total_bytes - cur_bytes) / max((cur_bytes - start_bytes) / elapsed, 1.0)) / 60.0
                if cur_bytes > start_bytes
                else 0.0
            )
            print(
                f"[{filename}] {cur_bytes / 1e9:.2f}/{total_bytes / 1e9:.2f} GB ({pct:.1f}%) | {speed:.2f} MB/s | ETA {eta_min:.1f} min",
                flush=True,
            )
        cur_bytes = part_path.stat().st_size if part_path.exists() else 0
        if proc.returncode == 0 and cur_bytes == total_bytes:
            return
        if round_idx == max_rounds:
            raise RuntimeError(
                f"curl exited with code {proc.returncode} after {max_rounds} resume rounds "
                f"({cur_bytes}/{total_bytes} bytes) while fetching {url}"
            )
        wait_s = min(30.0, 2.0 * round_idx)
        print(
            f"[{filename}] curl interrupted (code={proc.returncode}, {cur_bytes / 1e9:.2f}/{total_bytes / 1e9:.2f} GB); "
            f"auto-resuming round {round_idx + 1}/{max_rounds} in {wait_s:.0f}s...",
            flush=True,
        )
        time.sleep(wait_s)


def _download_with_python(
    filename: str,
    url: str,
    part_path: Path,
    meta_path: Path,
    total_bytes: int,
    workers: int,
    chunk_size: int,
    retries: int,
) -> None:
    n_chunks = (total_bytes + chunk_size - 1) // chunk_size
    completed: set[int] = set()
    if meta_path.exists() and part_path.exists() and part_path.stat().st_size == total_bytes:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("total_bytes") == total_bytes and meta.get("chunk_size") == chunk_size:
                completed = {int(c) for c in meta.get("completed_chunks", []) if 0 <= int(c) < n_chunks}
        except (OSError, ValueError):
            completed = set()

    if not part_path.exists() or part_path.stat().st_size != total_bytes:
        completed = set()
        with part_path.open("wb") as f:
            if total_bytes > 0:
                f.seek(total_bytes - 1)
                f.write(b"\x00")

    pending = [cid for cid in range(n_chunks) if cid not in completed]
    already_bytes = sum(min(chunk_size, total_bytes - cid * chunk_size) for cid in completed)
    print(
        f"[{filename}] Python backend Total={total_bytes / 1e9:.2f} GB | chunks={n_chunks} | resumed={len(completed)} | remaining={len(pending)}",
        flush=True,
    )
    if not pending:
        return

    io_lock = threading.Lock()
    t_start = time.time()
    session_bytes = 0
    last_report = t_start
    chunks_since_meta = 0

    def _worker(cid: int) -> tuple[int, int]:
        start = cid * chunk_size
        end = min(total_bytes - 1, start + chunk_size - 1)
        data = _fetch_range(url, start, end, retries=retries)
        with io_lock:
            with part_path.open("r+b") as fh:
                fh.seek(start)
                fh.write(data)
        return cid, len(data)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(_worker, cid): cid for cid in pending}
        for fut in as_completed(futures):
            cid, n_bytes = fut.result()
            with io_lock:
                completed.add(cid)
                session_bytes += n_bytes
                chunks_since_meta += 1
                now = time.time()
                if chunks_since_meta >= 8 or len(completed) == n_chunks:
                    _save_meta(meta_path, total_bytes, chunk_size, completed)
                    chunks_since_meta = 0
                if now - last_report >= 5.0 or len(completed) == n_chunks:
                    total_done = already_bytes + session_bytes
                    elapsed = max(now - t_start, 1e-3)
                    speed_mb = session_bytes / (1024 * 1024) / elapsed
                    eta_s = max(total_bytes - total_done, 0) / max(session_bytes / elapsed, 1.0)
                    pct = 100.0 * total_done / total_bytes
                    print(
                        f"[{filename}] {total_done / 1e9:.2f}/{total_bytes / 1e9:.2f} GB "
                        f"({pct:.1f}%) | {speed_mb:.2f} MB/s | ETA {eta_s / 60.0:.1f} min",
                        flush=True,
                    )
                    last_report = now


def download_one(
    key: str,
    mirror_url: str,
    out_dir: Path,
    backend: str,
    workers: int,
    chunk_size: int,
    retries: int,
) -> Path:
    filename = DATASETS[key]
    url = f"{mirror_url}/{filename}"
    final_path = out_dir / filename
    part_path = out_dir / f"{filename}.part"
    meta_path = out_dir / f"{filename}.meta.json"

    total_bytes = _probe_remote_size(url)
    if final_path.exists() and final_path.stat().st_size == total_bytes:
        _verify_file_signature(key, final_path)
        print(f"[{filename}] Already complete ({total_bytes / 1e9:.2f} GB) at {final_path}", flush=True)
        return final_path

    curl_exe = shutil.which("curl.exe") or shutil.which("curl")
    use_curl = (backend == "curl") or (backend == "auto" and curl_exe is not None)
    if use_curl:
        if curl_exe is None:
            raise RuntimeError("curl backend requested but curl executable not found in PATH")
        # Wait briefly if a non-compatible Python sparse .part file already existed without curl progress
        if meta_path.exists() and part_path.exists() and part_path.stat().st_size == total_bytes:
            # Sparse preallocated python file cannot be resumed by curl -C - directly -> remove old preallocated placeholder
            part_path.unlink()
            meta_path.unlink(missing_ok=True)
        _download_with_curl(curl_exe, filename, url, part_path, total_bytes, retries=retries)
    else:
        _download_with_python(filename, url, part_path, meta_path, total_bytes, workers, chunk_size, retries)

    if part_path.stat().st_size != total_bytes:
        raise RuntimeError(f"Size mismatch for {part_path}: {part_path.stat().st_size} != {total_bytes}")

    _verify_file_signature(key, part_path)
    part_path.replace(final_path)
    if meta_path.exists():
        meta_path.unlink()
    print(f"[{filename}] Verified & saved to {final_path}", flush=True)
    return final_path


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    mirror_url = MIRRORS[args.mirror]
    chunk_size = int(args.chunk_mb) * 1024 * 1024

    keys = [k.strip().lower() for k in args.datasets.split(",") if k.strip()]
    for k in keys:
        if k not in DATASETS:
            raise ValueError(f"Unknown dataset '{k}'; choose from {sorted(DATASETS.keys())}")

    for k in keys:
        download_one(
            key=k,
            mirror_url=mirror_url,
            out_dir=out_dir,
            backend=args.backend,
            workers=int(args.workers),
            chunk_size=chunk_size,
            retries=int(args.retries),
        )


if __name__ == "__main__":
    main()
