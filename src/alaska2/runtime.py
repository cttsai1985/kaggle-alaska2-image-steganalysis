"""Device, precision, loaders, JSON and checkpoint I/O shared by train/evaluate/verify."""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from alaska2.data import EpochShuffleSampler, ImageDataset, worker_init

AMP_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}

# Set by the CLI SIGTERM handler; a run that fails for any reason after SIGTERM was received was stopped
# from outside (the wall-clock cap), so its manifest says interrupted, not failed.
TERMINATION = {"signal": None}


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_device(name: str) -> torch.device:
    """Explicit device; asking for CUDA without a visible GPU is an error, never a silent CPU fallback."""
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda requested but torch.cuda.is_available() is False")
        return torch.device("cuda", 0)
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unknown device {name!r}")


def autocast(device: torch.device, amp: str):
    dtype = AMP_DTYPES[amp]
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def make_loader(ds: ImageDataset, cfg: dict, batch_size: int, train: bool, epoch: int = 0) -> DataLoader:
    t = cfg["train"]
    workers = int(t["num_workers"])
    sampler = EpochShuffleSampler(len(ds), seed=int(t["seed"]), epoch=epoch, shuffle=train)
    # Dedicated generator: each new loader iterator draws its worker base seed from ``generator``; with
    # ``generator=None`` that draw consumes the global CPU RNG. A persistent loader draws only on its first
    # iteration, so a resumed run would take one extra draw after ``set_rng_state`` and shift later CPU
    # dropout masks. Augmentation never reads worker RNG (it is a pure function of seed, epoch, row).
    gen = torch.Generator().manual_seed(int(t["seed"]) + int(epoch))
    return DataLoader(
        ds,
        generator=gen,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=train,
        num_workers=workers,
        pin_memory=bool(t.get("pin_memory", False)) and torch.cuda.is_available(),
        persistent_workers=bool(t.get("persistent_workers", False)) and workers > 0,
        worker_init_fn=worker_init if workers > 0 else None,
        prefetch_factor=int(t.get("prefetch_factor", 2)) if workers > 0 else None,
    )


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def save_atomic(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def rng_state() -> dict:
    """RNG state as tensors / plain containers so checkpoints load with ``weights_only=True``."""
    np_state = np.random.get_state()
    out = {
        "python": random.getstate(),
        "numpy": {
            "name": np_state[0],
            "key": torch.from_numpy(np_state[1].astype(np.int64)),
            "pos": int(np_state[2]),
            "has_gauss": int(np_state[3]),
            "cached": float(np_state[4]),
        },
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        out["cuda"] = torch.cuda.get_rng_state_all()
    return out


def set_rng_state(state: dict) -> None:
    py = state["python"]
    random.setstate((py[0], tuple(py[1]), py[2]))
    n = state["numpy"]
    np.random.set_state(
        (n["name"], n["key"].numpy().astype(np.uint32), n["pos"], n["has_gauss"], n["cached"])
    )
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
