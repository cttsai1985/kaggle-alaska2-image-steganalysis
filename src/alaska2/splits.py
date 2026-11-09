"""Frozen family fold assignment (phase 1) and fit/valid separation checks.

The independence unit is the source-image family (Cover + its three stego variants). Folds are loaded,
never recomputed, and their content hash must equal the phase-1 hash before any use.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl

from alaska2.constants import CLASSES
from alaska2.identity import folds_content_sha256


class IdentityError(RuntimeError):
    """An input identity (hash, membership) differs from the declared one; the work must be rejected."""


def load_folds(path: str | Path, expected_sha256: str | None) -> tuple[pl.DataFrame, str]:
    folds = pl.read_parquet(path).select(pl.col("family_id").cast(pl.String), pl.col("fold").cast(pl.Int32))
    if folds["family_id"].n_unique() != folds.height:
        raise IdentityError("fold table has duplicate family_id rows")
    got = folds_content_sha256(folds)
    if expected_sha256 and got != expected_sha256:
        raise IdentityError(f"fold assignment hash {got} != expected {expected_sha256}")
    return folds, got


def attach_folds(train: pl.DataFrame, folds: pl.DataFrame) -> pl.DataFrame:
    """Many image rows to one fold per family; every train row must get exactly one fold."""
    out = train.join(folds, on="family_id", how="left", validate="m:1")
    if out.height != train.height:
        raise IdentityError(f"fold join changed row count {train.height} -> {out.height}")
    if out["fold"].is_null().any():
        raise IdentityError(f"{int(out['fold'].is_null().sum())} train rows have no fold")
    return out


def subset_families(families: pl.Series, n: int, seed: int) -> set[str]:
    """Deterministic family subset: order by sha256(seed:family_id), take the first n. n <= 0 keeps all."""
    fams = sorted(set(families.to_list()))
    if n <= 0 or n >= len(fams):
        return set(fams)
    key = {f: hashlib.sha256(f"{seed}:{f}".encode()).hexdigest() for f in fams}
    return set(sorted(fams, key=key.__getitem__)[:n])


def assign_roles(
    records: pl.DataFrame,
    holdout_fold: int,
    max_fit_families: int = 0,
    max_valid_families: int = 0,
    seed: int = 0,
) -> pl.DataFrame:
    """role = valid for the holdout fold, fit for the other folds; optional explicit family subsets."""
    df = records.with_columns(
        pl.when(pl.col("fold") == holdout_fold).then(pl.lit("valid")).otherwise(pl.lit("fit")).alias("role")
    )
    keep_fit = subset_families(df.filter(pl.col("role") == "fit")["family_id"], max_fit_families, seed)
    keep_valid = subset_families(df.filter(pl.col("role") == "valid")["family_id"], max_valid_families, seed)
    df = df.filter(pl.col("family_id").is_in(list(keep_fit | keep_valid)))
    assert_family_separation(df)
    return df


def assert_family_separation(df: pl.DataFrame) -> dict:
    """No family and no byte-identical content may appear in both fit and valid."""
    crossing = df.group_by("family_id").agg(pl.col("role").n_unique().alias("n")).filter(pl.col("n") > 1)
    if crossing.height:
        raise IdentityError(f"{crossing.height} families cross fit/valid, e.g. {crossing['family_id'][0]}")
    linked = df.group_by("sha256").agg(pl.col("role").n_unique().alias("n")).filter(pl.col("n") > 1)
    if linked.height:
        raise IdentityError(f"{linked.height} byte-identical files cross fit/valid (content-linked families)")
    return {"families_crossing_roles": 0, "content_groups_crossing_roles": 0}


def audit_roles(df: pl.DataFrame) -> dict:
    """Per-role counts by class and quality; used to trace a score back to its population."""
    out: dict = {}
    for role in ("fit", "valid"):
        r = df.filter(pl.col("role") == role)
        out[role] = {
            "rows": r.height,
            "families": r["family_id"].n_unique(),
            "by_kind": {k: int((r["kind"] == k).sum()) for k in CLASSES},
            "by_quality": {
                str(q): int(n) for q, n in r.group_by("quality").len().sort("quality").iter_rows()
            },
            "complete_families": int(
                r.group_by("family_id").len().filter(pl.col("len") == len(CLASSES)).height
            ),
        }
    return out
