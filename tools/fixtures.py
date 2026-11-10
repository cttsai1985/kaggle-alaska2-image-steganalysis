"""Synthetic images for CPU contracts/walkthroughs; never a real-data model result."""

from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image

from tools.readiness.build_manifest import inspect_file, list_files
from tools.readiness.legacy_split import legacy_family_folds

KINDS = ("Cover", "JMiPOD", "JUNIWARD", "UERD")
FIXTURE_SIZE = 64
N_FAMILIES = 20


def _save(arr: np.ndarray, path: Path, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8), "RGB").save(path, "JPEG", quality=quality)


def build_fixture(root: Path, n_families: int = N_FAMILIES, size: int = FIXTURE_SIZE, seed: int = 0) -> dict:
    """Families with non-contiguous ids; each stego variant = Cover + a kind-specific +-1 pattern."""
    rng = np.random.default_rng(seed)
    images = root / "raw" / "extracted"
    qualities = (75, 90, 95)
    for i in range(n_families):
        fam = f"{i * 3 + 1:05d}"  # gaps in the id range, as in the real data
        q = qualities[i % 3]
        base = rng.integers(40, 216, size=(size, size, 3)).astype(np.int16)
        _save(base, images / "Cover" / f"{fam}.jpg", q)
        for k, kind in enumerate(KINDS[1:], 1):
            noise = (rng.random((size, size, 3)) < 0.05 * k) * rng.choice([-1, 1], size=(size, size, 3))
            _save(np.clip(base + 8 * noise, 0, 255), images / kind / f"{fam}.jpg", q)
    for j in range(5):
        _save(rng.integers(0, 256, size=(size, size, 3)), images / "Test" / f"{j:04d}.jpg", 90)
    rows = [inspect_file((str(images), r)) for r in list_files(images)]
    manifest = pl.DataFrame(rows, infer_schema_length=None).sort("relpath")
    (root / "manifest").mkdir(parents=True, exist_ok=True)
    manifest.write_parquet(root / "manifest" / "files_manifest.parquet")
    train = manifest.filter(pl.col("split") == "train").with_columns(pl.col("q_from_table").alias("quality"))
    folds = legacy_family_folds(train).sort("family_id")
    folds.write_parquet(root / "manifest" / "legacy_family_folds.parquet")
    return {"root": root, "manifest": manifest, "folds": folds}
