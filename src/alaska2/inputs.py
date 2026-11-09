"""Resolve and verify every input identity before any fit, evaluation or verification."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from alaska2.config import data_paths
from alaska2.data import check_image_size, load_manifest, test_records, train_records
from alaska2.identity import manifest_content_sha256, sha256_file
from alaska2.splits import IdentityError, assign_roles, attach_folds, audit_roles, load_folds


@dataclass
class Inputs:
    records: pl.DataFrame  # labeled rows with fold and role, deterministic order
    test: pl.DataFrame
    images_dir: Path
    identities: dict
    audit: dict


def population_sha256(df: pl.DataFrame) -> str:
    rows = df.select("relpath", "role").sort("relpath").iter_rows()
    return hashlib.sha256("\n".join(f"{r}\t{o}" for r, o in rows).encode()).hexdigest()


def prepare_inputs(cfg: dict) -> Inputs:
    paths = data_paths(cfg)
    d, s = cfg["data"], cfg["split"]
    manifest = load_manifest(paths["manifest"])
    m_hash = manifest_content_sha256(manifest)
    if d.get("expected_manifest_sha256") and m_hash != d["expected_manifest_sha256"]:
        raise IdentityError(f"input manifest hash {m_hash} != expected {d['expected_manifest_sha256']}")
    folds, f_hash = load_folds(paths["folds"], d.get("expected_folds_sha256"))
    labeled = train_records(manifest)
    check_image_size(labeled, int(d["image_size"]))
    recs = assign_roles(
        attach_folds(labeled, folds),
        holdout_fold=int(s["holdout_fold"]),
        max_fit_families=int(s.get("max_fit_families", 0)),
        max_valid_families=int(s.get("max_valid_families", 0)),
        seed=int(cfg["train"]["seed"]),
    )
    subset = int(s.get("max_fit_families", 0)) > 0 or int(s.get("max_valid_families", 0)) > 0
    identities = {
        "input_manifest_sha256": m_hash,
        "split_folds_sha256": f_hash,
        "population_sha256": population_sha256(recs),
        "holdout_fold": int(s["holdout_fold"]),
        "population": "explicit_family_subset" if subset else "full_data_single_holdout",
        "manifest_parquet_sha256": sha256_file(paths["manifest"]),
    }
    return Inputs(recs, test_records(manifest), paths["images_dir"], identities, audit_roles(recs))


def rehash_sample(inputs: Inputs, n: int, seed: int) -> dict:
    """Re-read n random image files and compare their bytes with the manifest sha256."""
    rows = inputs.records.select("relpath", "sha256").rows()
    pick = random.Random(seed).sample(rows, min(n, len(rows))) if n > 0 else []
    bad = [r for r, h in pick if sha256_file(inputs.images_dir / r) != h]
    return {"files_rehashed": len(pick), "mismatched": len(bad)}


def decode_sample(inputs: Inputs, cfg: dict, n: int, seed: int) -> dict:
    """Decode n random fit rows through the training Dataset path (decode, D4, normalize)."""
    import torch

    from alaska2.data import ImageDataset

    if n <= 0:
        return {"images": 0}
    fit = inputs.records.filter(pl.col("role") == "fit")
    ds = ImageDataset(fit, inputs.images_dir, cfg, train=True)
    rows = random.Random(seed).sample(range(len(ds)), min(n, len(ds)))
    xs = [ds[(r, 0)][0] for r in rows]
    x = torch.stack(xs)
    return {
        "images": len(xs),
        "tensor_shape": list(x.shape[1:]),
        "dtype": str(x.dtype),
        "finite": bool(torch.isfinite(x).all()),
        "mean": round(float(x.mean()), 4),
        "std": round(float(x.std()), 4),
    }


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
