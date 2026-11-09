"""CPU JPEG decode throughput probe, to size the training DataLoader.

Samples a deterministic, kind-balanced set of files from the manifest, then
times full RGB decode to a uint8 array with Pillow and (if installed) OpenCV,
single-process and with a process pool. The sample is read once beforehand so
the timings measure decode from page cache, not first-touch disk reads; a
separate cold-ish read pass is recorded for reference.

    uv run --group probe python -m tools.readiness.decode_probe \
        --data-dir ~/data/kaggle/alaska2-image-steganalysis/raw/extracted \
        --manifest ~/data/kaggle/alaska2-image-steganalysis/manifest/files_manifest.parquet \
        --out evidence/phase1/decode_probe.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image

try:
    import cv2

    cv2.setNumThreads(1)
except ImportError:  # optional backend
    cv2 = None

_ROOT = ""


def _init(root: str) -> None:
    global _ROOT
    _ROOT = root
    if cv2 is not None:
        cv2.setNumThreads(1)


def decode_pil(rel: str) -> int:
    with Image.open(os.path.join(_ROOT, rel)) as im:
        arr = np.asarray(im.convert("RGB"))
    return int(arr.shape[0] * arr.shape[1])


def decode_cv2(rel: str) -> int:
    arr = cv2.imread(os.path.join(_ROOT, rel), cv2.IMREAD_COLOR)
    return int(arr.shape[0] * arr.shape[1])


BACKENDS = {"pillow": decode_pil, "opencv": decode_cv2}


def sample(manifest: Path, n: int, seed: int) -> list[str]:
    m = pl.read_parquet(manifest, columns=["relpath", "kind"]).sort("relpath")
    kinds = sorted(m["kind"].unique())
    per = n // len(kinds)
    parts = [m.filter(pl.col("kind") == k).sample(per, seed=seed, shuffle=True) for k in kinds]
    return sorted(pl.concat(parts)["relpath"].to_list())


def timed(fn, rels: list[str], root: str, workers: int) -> dict:
    t0 = time.perf_counter()
    if workers == 1:
        _init(root)
        px = [fn(r) for r in rels]
    else:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_init, initargs=(root,), mp_context=mp.get_context("forkserver")
        ) as ex:
            px = list(ex.map(fn, rels, chunksize=16))
    wall = time.perf_counter() - t0
    return {
        "workers": workers,
        "images": len(rels),
        "wall_s": round(wall, 3),
        "images_per_s": round(len(rels) / wall, 1),
        "pixels_checked": int(sum(px)),
    }


def _read_len(path: str) -> int:
    with open(path, "rb") as f:
        return len(f.read())


def read_probe(root: str, rels: list[str], threads: int) -> dict:
    """Raw file-read throughput (no decode) on a sample disjoint from the decode sample."""
    paths = [os.path.join(root, r) for r in rels]
    out = {"threads": threads, "files": len(paths)}
    for name in ("first_pass", "second_pass"):
        t0 = time.perf_counter()
        with ThreadPoolExecutor(threads) as ex:
            n = sum(ex.map(_read_len, paths))
        wall = time.perf_counter() - t0
        out[f"{name}_files_per_s"] = round(len(paths) / wall, 1)
        out[f"{name}_mb_per_s"] = round(n / 2**20 / wall, 1)
    return out


def io_pressure() -> str | None:
    try:
        return Path("/proc/pressure/io").read_text().splitlines()[0]
    except OSError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261109)
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8, 16])
    a = ap.parse_args()
    if max(a.workers) > 16:
        raise SystemExit("worker cap is 16 for this host")

    root = str(a.data_dir)
    rels = sample(a.manifest, a.n, a.seed)
    t0 = time.perf_counter()
    nbytes = sum(len(Path(root, r).read_bytes()) for r in rels)
    first_read_s = time.perf_counter() - t0

    decode_set = set(rels)
    read_rels = [r for r in sample(a.manifest, a.n + len(rels), a.seed + 1) if r not in decode_set][: a.n]
    io_before = io_pressure()
    reads = [read_probe(root, read_rels[: a.n // 2], 1), read_probe(root, read_rels[a.n // 2 :], 16)]

    results: dict = {}
    for name, fn in BACKENDS.items():
        if name == "opencv" and cv2 is None:
            results[name] = {"available": False}
            continue
        _init(root)
        fn(rels[0])  # warm-up in the parent process
        results[name] = {"available": True, "runs": [timed(fn, rels, root, w) for w in a.workers]}

    out = {
        "probe": "phase1_cpu_jpeg_decode",
        "captured_at_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sample": {
            "n_images": len(rels),
            "kind_balanced": True,
            "seed": a.seed,
            "bytes": nbytes,
            "first_read_s": round(first_read_s, 3),
            "first_read_mb_per_s": round(nbytes / 2**20 / first_read_s, 1),
            "note": "first read may already be page-cached after extraction; decode timings are warm-cache",
        },
        "decode": "full decode to HxWx3 uint8 (Pillow convert('RGB'); OpenCV imread BGR); 1 thread/process",
        "versions": {
            "pillow": Image.__version__,
            "opencv": cv2.__version__ if cv2 is not None else None,
            "numpy": np.__version__,
        },
        "read_probe": {
            "runs": reads,
            "note": "disjoint sample per thread count; 1st pass may be partly cached, 2nd is cached",
            "io_pressure_before": io_before,
        },
        "cpu_threads_visible": os.cpu_count(),
        "io_pressure_at_end": io_pressure(),
        "load_avg_1m_at_end": round(os.getloadavg()[0], 1),
        "backends": results,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
