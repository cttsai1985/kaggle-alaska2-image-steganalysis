"""Identities recorded with every run: source, inputs, split, config, weights and runtime versions."""

from __future__ import annotations

import hashlib
import platform
import subprocess
from pathlib import Path

import polars as pl


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def manifest_content_sha256(manifest: pl.DataFrame) -> str:
    """Same definition as phase 1 (tools/readiness/build_manifest.py): sorted ``relpath\\tsha256`` lines."""
    rows = manifest.select("relpath", "sha256").sort("relpath").iter_rows()
    return hashlib.sha256("\n".join(f"{r}\t{s}" for r, s in rows).encode()).hexdigest()


def folds_content_sha256(folds: pl.DataFrame) -> str:
    """Same definition as phase 1 (tools/readiness/legacy_split.py): sorted ``family_id\\tfold`` lines."""
    rows = folds.select("family_id", "fold").sort("family_id").iter_rows()
    return hashlib.sha256("\n".join(f"{f}\t{k}" for f, k in rows).encode()).hexdigest()


def git_source(repo: Path | None = None) -> dict:
    repo = repo or Path(__file__).resolve().parents[2]

    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    sha = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=no")
    return {"git_sha": sha, "dirty": bool(status) if status is not None else None}


def runtime_versions() -> dict:
    import cv2
    import numpy
    import PIL
    import timm
    import torch
    import torchvision

    out = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "numpy": numpy.__version__,
        "opencv": cv2.__version__,
        "pillow": PIL.__version__,
        "polars": pl.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        out["cuda_device"] = torch.cuda.get_device_name(0)
        out["cuda_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
    return out
