"""Reproduce the legacy fold assignment and audit family separation.

Legacy behavior (``alaska_utils/data_utils.py::split_train_valid_data``,
``script/trainer.py``): train rows sorted by image name, deduplicated to one row
per base image, then ``StratifiedKFold(n_splits=5)`` (no shuffle) stratified on
JPEG ``quality``; ``groups`` is passed but ignored by StratifiedKFold. Each fold's
families are expanded back to all four variants. Legacy training uses fold 1.

The fold-assignment parquet stays outside Git; only aggregates are committed.
A row-level split is computed only as a labeled leakage diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedKFold

N_SPLITS = 5


def legacy_family_folds(train: pl.DataFrame) -> pl.DataFrame:
    """train: one row per file with family_id, image, label, quality. Returns family_id -> fold (1-based)."""
    # Checked on all variants *before* dedup: after dedup each image has one row and nothing can differ.
    if train.group_by("image").agg(pl.col("quality").n_unique()).filter(pl.col("quality") > 1).height:
        raise ValueError("quality differs within a family; legacy dedup would be order-dependent")
    fam = train.sort("image").unique("image", keep="first", maintain_order=True)
    skf = StratifiedKFold(n_splits=N_SPLITS)
    fold = np.zeros(fam.height, dtype=np.int32)
    for k, (_, valid_idx) in enumerate(skf.split(X=fam["label"].to_numpy(), y=fam["quality"].to_numpy()), 1):
        fold[valid_idx] = k
    return fam.select("family_id").with_columns(pl.Series("fold", fold))


def audit_folds(train: pl.DataFrame, folds: pl.DataFrame) -> dict:
    rows = train.join(folds, on="family_id", how="left")
    out: dict = {"rows_without_fold": int(rows["fold"].is_null().sum())}
    out["families_in_multiple_folds"] = int(
        rows.group_by("family_id").agg(pl.col("fold").n_unique()).filter(pl.col("fold") > 1).height
    )
    per = rows.group_by("fold").agg(
        pl.col("family_id").n_unique().alias("families"),
        pl.len().alias("rows"),
        *[(pl.col("kind") == k).sum().alias(k) for k in ("Cover", "JMiPOD", "JUNIWARD", "UERD")],
        *[(pl.col("quality") == q).sum().alias(f"q{q}") for q in (75, 90, 95)],
    )
    out["per_fold"] = per.sort("fold").to_dicts()
    return out


def content_linked_families(train: pl.DataFrame, folds: pl.DataFrame) -> dict:
    """Byte-identical files under different family IDs, and whether the folds separate them."""
    rows = train.select("family_id", "sha256").join(folds, on="family_id", how="left")
    grp = rows.group_by("sha256").agg(
        pl.col("family_id").n_unique().alias("n_families"), pl.col("fold").n_unique().alias("n_folds")
    )
    linked = grp.filter(pl.col("n_families") > 1)
    return {
        "byte_identical_groups_spanning_families": linked.height,
        "of_which_split_across_folds": int((linked["n_folds"] > 1).sum()),
        "note": "family = filename stem; identical bytes under two stems are invisible to the legacy split",
    }


def row_level_leak_diagnostic(train: pl.DataFrame, seed: int = 0) -> dict:
    """DIAGNOSTIC ONLY: what a row-level split would allow. Never a reported evaluation."""
    t = train.sort("relpath")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    _, valid_idx = next(skf.split(np.zeros(t.height), t["label"].to_numpy()))
    is_valid = np.zeros(t.height, dtype=bool)
    is_valid[valid_idx] = True
    t = t.with_columns(pl.Series("is_valid", is_valid))
    train_fams = t.filter(~pl.col("is_valid"))["family_id"].implode()
    v = t.filter(pl.col("is_valid"))
    leaked = v.filter(pl.col("family_id").is_in(train_fams))
    stego_valid = v.filter(pl.col("kind") != "Cover")
    cover_in_train = t.filter(~pl.col("is_valid") & (pl.col("kind") == "Cover"))["family_id"].implode()
    return {
        "label": "DIAGNOSTIC_ONLY_row_level_split_not_an_evaluation",
        "split": f"StratifiedKFold(n_splits={N_SPLITS}, shuffle=True, random_state={seed}) on label, fold 1",
        "valid_rows": v.height,
        "valid_rows_with_sibling_in_train": leaked.height,
        "valid_stego_rows_whose_own_cover_is_in_train": int(
            stego_valid["family_id"].is_in(cover_in_train).sum()
        ),
        "valid_stego_rows": stego_valid.height,
    }


def join_quality(manifest: pl.DataFrame, quality: pl.DataFrame) -> pl.DataFrame:
    """Train manifest rows with their CSV quality; strictly one quality row per (image, kind).

    Duplicate metadata keys would silently multiply image rows (and still pass the null and
    family checks), so cardinality and row count are enforced before any fold is produced.
    """
    train = manifest.filter(pl.col("split") == "train")
    try:
        out = train.join(quality, on=["image", "kind"], how="left", validate="1:1")
    except pl.exceptions.ComputeError as exc:
        raise ValueError(f"quality join is not 1:1 on (image, kind): {exc}") from exc
    if out.height != train.height:
        raise ValueError(f"quality join changed row count: {train.height} -> {out.height}")
    if out["quality"].is_null().any():
        raise ValueError("quality missing for some train rows; legacy stratification cannot be reproduced")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--quality-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--audit-json", type=Path, required=True)
    a = ap.parse_args()

    m = pl.read_parquet(a.manifest)
    q = pl.read_csv(a.quality_csv)
    try:
        train = join_quality(m, q)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    folds = legacy_family_folds(train)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    folds.sort("family_id").write_parquet(a.out_dir / "legacy_family_folds.parquet")
    folds_hash = hashlib.sha256(
        "\n".join(f"{f}\t{k}" for f, k in folds.sort("family_id").iter_rows()).encode()
    ).hexdigest()

    result = {
        "policy": "legacy: sort by image, dedup to base image, StratifiedKFold(5, no shuffle) on quality; "
        "groups argument ignored by StratifiedKFold; families expanded to all variants",
        "fold_assignment_content_sha256": folds_hash,
        "families": folds.height,
        "quality_join": {"validate": "1:1", "train_rows": train.height},
        **audit_folds(train, folds),
        "content_linked_families": content_linked_families(train, folds),
        "row_level_diagnostic": row_level_leak_diagnostic(train),
    }
    a.audit_json.parent.mkdir(parents=True, exist_ok=True)
    a.audit_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
