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
    return {
        "git_sha": sha,
        "dirty": bool(status) if status is not None else None,
        "package_sha256": package_source_sha256(),
    }


def package_source_sha256(pkg: Path | None = None) -> str:
    """Content hash of the code that runs: sorted ``relpath\tsha256`` of every ``alaska2/*.py`` file.

    Catches edits in a dirty tree, which ``git_sha`` alone does not.
    """
    pkg = pkg or Path(__file__).resolve().parent
    files = sorted(p for p in pkg.rglob("*.py") if "__pycache__" not in p.parts)
    lines = [f"{p.relative_to(pkg).as_posix()}\t{sha256_file(p)}" for p in files]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def run_provenance() -> dict:
    """Source and runtime identity of the current process; stored in every checkpoint, compared on resume."""

    def plain(v):  # e.g. torch.__version__ is a str subclass that weights_only checkpoint loading rejects
        return v if v is None or type(v) in (bool, int, float) else str(v)

    return {
        "source": {k: plain(v) for k, v in git_source().items()},
        "versions": {k: plain(v) for k, v in runtime_versions().items()},
    }


def provenance_diff(saved: dict | None, current: dict) -> list[str]:
    """Dotted keys whose value differs between a checkpoint's provenance and the current process."""
    if not saved:
        return ["provenance (absent from checkpoint)"]
    out = []
    for sect in ("source", "versions"):
        a, b = saved.get(sect) or {}, current.get(sect) or {}
        out += [f"{sect}.{k}" for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
    return out


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
