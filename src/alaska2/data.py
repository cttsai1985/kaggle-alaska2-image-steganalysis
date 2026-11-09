"""Manifest-driven records, OpenCV/Pillow decoding, pixel-preserving D4 transforms and the Dataset.

Decoding: OpenCV ``imread`` (BGR -> RGB), chosen from the phase-1 decode probe (491 vs 192 img/s single
process). Images are 512x512 native (all 305,000 files, phase-1 manifest), so there is no resize.
Normalization: ImageNet mean/std with max pixel 255, the same math as the legacy albumentations
``Normalize`` defaults and the timm ``efficientnet_b0.ra_in1k`` pretrained config.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset, Sampler

from alaska2.constants import CLASSES, TEST_KIND

RECORD_COLUMNS = ["relpath", "image", "family_id", "kind", "label", "quality", "sha256", "width", "height"]


def load_manifest(path: str | Path) -> pl.DataFrame:
    m = pl.read_parquet(path)
    required = (set(RECORD_COLUMNS) - {"quality"}) | {"q_from_table"}
    missing = required - set(m.columns)
    if missing:
        raise ValueError(f"manifest lacks columns {sorted(missing)}")
    return m.with_columns(pl.col("q_from_table").alias("quality"))


def train_records(manifest: pl.DataFrame) -> pl.DataFrame:
    """Labeled rows in deterministic order (family_id, label); label is the legacy class index."""
    df = manifest.filter(pl.col("kind").is_in(list(CLASSES))).select(RECORD_COLUMNS)
    expect = pl.col("kind").replace_strict({k: i for i, k in enumerate(CLASSES)}, return_dtype=pl.Int64)
    bad = df.filter(pl.col("label") != expect)
    if bad.height:
        raise ValueError(f"{bad.height} manifest rows have a label that disagrees with the class order")
    return df.sort("family_id", "label")


def test_records(manifest: pl.DataFrame) -> pl.DataFrame:
    df = manifest.filter(pl.col("kind") == TEST_KIND).select(RECORD_COLUMNS)
    return df.with_columns(pl.lit(-1, dtype=pl.Int64).alias("label")).sort("image")


def check_image_size(records: pl.DataFrame, size: int) -> None:
    bad = records.filter((pl.col("width") != size) | (pl.col("height") != size))
    if bad.height:
        raise ValueError(
            f"{bad.height} images are not {size}x{size}; resizing would destroy the stego signal"
        )


# --- decoding -------------------------------------------------------------------------------------------


def decode_opencv(path: str) -> np.ndarray:
    import cv2

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(f"cannot decode {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def decode_pillow(path: str) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


DECODERS = {"opencv": decode_opencv, "pillow": decode_pillow}


# --- transforms -----------------------------------------------------------------------------------------


def apply_d4(img: np.ndarray, ops: Sequence[dict], rng: np.random.Generator) -> np.ndarray:
    """Legacy recipe order: HorizontalFlip(p), VerticalFlip(p), RandomRotate90(p, k in 0..3).

    Each op only permutes pixel positions (no interpolation, no value change). On a 512x512 image the 8x8
    JPEG block grid maps onto itself, so the embedding signal is kept intact.
    """
    for t in ops:
        p = float(t.get("p", 1.0))
        if rng.random() >= p:
            continue
        if t["op"] == "hflip":
            img = img[:, ::-1]
        elif t["op"] == "vflip":
            img = img[::-1]
        elif t["op"] == "rot90":
            img = np.rot90(img, int(rng.integers(0, 4)), axes=(0, 1))
        else:
            raise ValueError(f"unsupported transform {t['op']!r}")
    return np.ascontiguousarray(img)


def normalize_to_tensor(
    img: np.ndarray, mean: Sequence[float], std: Sequence[float], max_pixel: float
) -> torch.Tensor:
    x = img.astype(np.float32) / np.float32(max_pixel)
    x = (x - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


class ImageDataset(Dataset):
    """Index keys are ``(row, epoch)`` so per-sample augmentation is a pure function of (seed, epoch, row),
    independent of worker count, worker order and resume point."""

    def __init__(self, records: pl.DataFrame, images_dir: str | Path, cfg: dict, train: bool):
        self.paths = [str(Path(images_dir) / r) for r in records["relpath"].to_list()]
        self.labels = records["label"].to_numpy().astype(np.int64)
        self.size = int(cfg["data"]["image_size"])
        self.decode = DECODERS[cfg["data"]["decoder"]]
        tr = cfg["transforms"]
        self.ops = tr["train"] if train else tr["valid"]
        self.mean, self.std, self.max_pixel = tr["normalize_mean"], tr["normalize_std"], tr["max_pixel_value"]
        self.seed = int(cfg["train"]["seed"])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, key: int | tuple[int, int]):
        row, epoch = key if isinstance(key, tuple) else (key, 0)
        img = self.decode(self.paths[row])
        if img.shape != (self.size, self.size, 3):
            raise ValueError(f"{self.paths[row]}: decoded shape {img.shape} != ({self.size}, {self.size}, 3)")
        if self.ops:
            img = apply_d4(img, self.ops, np.random.default_rng([self.seed, epoch, row]))
        return normalize_to_tensor(img, self.mean, self.std, self.max_pixel), int(self.labels[row]), row


class EpochShuffleSampler(Sampler):
    """Permutation from torch.Generator(seed + epoch); yields ``(row, epoch)`` keys. Resumable by epoch."""

    def __init__(self, n: int, seed: int, epoch: int = 0, shuffle: bool = True):
        self.n, self.seed, self.epoch, self.shuffle = n, seed, epoch, shuffle

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.n

    def __iter__(self) -> Iterator[tuple[int, int]]:
        if self.shuffle:
            g = torch.Generator().manual_seed(self.seed + self.epoch)
            order = torch.randperm(self.n, generator=g).tolist()
        else:
            order = range(self.n)
        return iter((int(i), self.epoch) for i in order)


def worker_init(_: int) -> None:
    """One decode thread per loader worker, as in the phase-1 probe."""
    import cv2

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
