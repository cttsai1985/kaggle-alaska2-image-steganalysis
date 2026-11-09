"""Build a deterministic manifest of the real ALASKA2 files and audit membership.

The manifest (one row per file, with hashes) stays outside Git. Only the
aggregate audit JSON is meant to be committed.

    uv run python -m tools.readiness.build_manifest \
        --data-dir ~/data/kaggle/alaska2-image-steganalysis/raw/extracted \
        --quality-csv ~/data/kaggle/alaska2-image-steganalysis/quality/image_quality.csv \
        --sample-submission ~/data/kaggle/alaska2-image-steganalysis/raw/extracted/sample_submission.csv \
        --out-dir ~/data/kaggle/alaska2-image-steganalysis/manifest \
        --audit-json evidence/phase1/dataset_audit.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl
from PIL import Image

from tools.readiness.jpeg_quality import quality_from_table

TRAIN_KINDS = ("Cover", "JMiPOD", "JUNIWARD", "UERD")  # legacy class order: label = index
TEST_KIND = "Test"
KINDS = (*TRAIN_KINDS, TEST_KIND)


def inspect_file(args: tuple[str, str]) -> dict:
    root, rel = args
    path = os.path.join(root, rel)
    data = Path(path).read_bytes()
    kind, name = rel.split("/", 1)
    row = {
        "relpath": rel,
        "kind": kind,
        "image": name,
        "family_id": Path(name).stem,
        "split": "test" if kind == TEST_KIND else "train",
        "label": TRAIN_KINDS.index(kind) if kind in TRAIN_KINDS else None,
        "n_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "width": None,
        "height": None,
        "mode": None,
        "format": None,
        "q_from_table": None,
        "n_qtables": None,
        "open_error": None,
    }
    try:
        with Image.open(path) as im:
            row.update(width=im.width, height=im.height, mode=im.mode, format=im.format)
            tables = getattr(im, "quantization", None) or {}
            row["n_qtables"] = len(tables)
            if 0 in tables:
                row["q_from_table"] = quality_from_table(list(tables[0]))
    except Exception as exc:  # retain the failure in the manifest rather than dropping the row
        row["open_error"] = f"{type(exc).__name__}: {exc}"
    return row


def list_files(root: Path) -> list[str]:
    rels = []
    for kind in sorted(p.name for p in root.iterdir() if p.is_dir()):
        for f in (root / kind).iterdir():
            if f.is_file():
                rels.append(f"{kind}/{f.name}")
    return sorted(rels)


def audit(df: pl.DataFrame, quality: pl.DataFrame | None) -> dict:
    """Aggregate-only checks; no image rows or IDs are returned."""
    out: dict = {"n_files": df.height}
    out["files_per_kind"] = {r["kind"]: r["len"] for r in df.group_by("kind").len().sort("kind").to_dicts()}
    out["unexpected_kinds"] = sorted(set(df["kind"].unique()) - set(KINDS))
    out["open_errors"] = int(df["open_error"].is_not_null().sum())
    out["non_jpeg"] = int((df["format"] != "JPEG").sum())
    dims = df.group_by("width", "height", "mode").len().sort("len", descending=True)
    out["dimension_mode_counts"] = [
        {"width": r["width"], "height": r["height"], "mode": r["mode"], "n": r["len"]}
        for r in dims.to_dicts()
    ]
    out["total_bytes"] = int(df["n_bytes"].sum())

    train = df.filter(pl.col("split") == "train")
    test = df.filter(pl.col("split") == "test")
    fam = train.group_by("family_id").agg(
        pl.col("kind").n_unique().alias("n_kinds"), pl.len().alias("n"), pl.col("kind").sort().alias("kinds")
    )
    out["train_families"] = fam.height
    out["families_complete_4_variants"] = int(((fam["n_kinds"] == 4) & (fam["n"] == 4)).sum())
    out["families_incomplete"] = int(((fam["n_kinds"] != 4) | (fam["n"] != 4)).sum())
    out["test_images"] = test.height
    out["test_names_overlapping_train_families"] = int(
        test["family_id"].is_in(train["family_id"].implode()).sum()
    )

    # Numeric ranges: is a contiguous range a valid assumption?
    ids = fam["family_id"].cast(pl.Int64, strict=False)
    out["train_family_id_non_numeric"] = int(ids.is_null().sum())
    if ids.is_not_null().any():
        lo, hi = int(ids.min()), int(ids.max())
        out["train_family_id_range"] = [lo, hi]
        out["train_family_id_gaps_in_range"] = (hi - lo + 1) - int(ids.n_unique())

    # Byte-identical files
    dup = df.group_by("sha256").agg(
        pl.len().alias("n"),
        pl.col("kind").sort().alias("kinds"),
        pl.col("family_id").n_unique().alias("n_families"),
    )
    dup = dup.filter(pl.col("n") > 1)
    out["byte_duplicate_groups"] = dup.height
    out["byte_duplicate_kind_patterns"] = {
        k: n
        for k, n in dup.with_columns(pl.col("kinds").list.join("+").alias("k"))
        .group_by("k")
        .len()
        .sort("k")
        .iter_rows()
    }
    # Identical bytes under *different* family IDs: a filename-based family split cannot see these.
    xfam = dup.filter(pl.col("n_families") > 1)
    out["cross_family_byte_duplicate_groups"] = xfam.height
    out["cross_family_byte_duplicate_kind_patterns"] = {
        k: n
        for k, n in xfam.with_columns(pl.col("kinds").list.join("+").alias("k"))
        .group_by("k")
        .len()
        .sort("k")
        .iter_rows()
    }
    # Stego file byte-identical to its own Cover (no embedding change at the byte level)
    cover = train.filter(pl.col("kind") == "Cover").select("family_id", pl.col("sha256").alias("cover_sha"))
    stego = train.filter(pl.col("kind") != "Cover").join(cover, on="family_id", how="left")
    out["stego_identical_to_own_cover"] = {
        k: int(v)
        for k, v in stego.group_by("kind")
        .agg((pl.col("sha256") == pl.col("cover_sha")).sum())
        .sort("kind")
        .iter_rows()
    }
    same = stego.filter(pl.col("sha256") == pl.col("cover_sha"))
    out["stego_identical_to_own_cover_by_kind_table_quality"] = {
        f"{k}/{q}": n
        for k, q, n in same.group_by("kind", "q_from_table").len().sort("kind", "q_from_table").iter_rows()
    }
    out["families_with_any_stego_identical_to_cover"] = int(same["family_id"].n_unique())
    out["cross_split_byte_duplicates"] = int(test["sha256"].is_in(train["sha256"].implode()).sum())

    # Quality from the JPEG quantization tables (independent of the CSV)
    out["q_from_table_counts"] = {
        str(k): v for k, v in df.group_by("q_from_table").len().sort("q_from_table").iter_rows()
    }
    qfam = train.group_by("family_id").agg(pl.col("q_from_table").n_unique().alias("nq"))
    out["families_with_mixed_table_quality"] = int((qfam["nq"] > 1).sum())

    if quality is not None:
        out["quality_csv_rows"] = quality.height
        out["quality_csv_duplicate_keys"] = quality.height - quality.unique(["image", "kind"]).height
        j = df.join(quality, on=["image", "kind"], how="left")
        out["files_without_quality_row"] = int(j["quality"].is_null().sum())
        anti = quality.join(df.select("image", "kind"), on=["image", "kind"], how="anti")
        out["quality_rows_without_file"] = anti.height
        out["join_rows_equal_files"] = j.height == df.height
        known = j.filter(pl.col("q_from_table").is_not_null() & pl.col("quality").is_not_null())
        out["quality_csv_vs_table_compared"] = known.height
        out["quality_csv_vs_table_mismatch"] = int((known["quality"] != known["q_from_table"]).sum())
        out["quality_by_split_kind"] = {
            f"{s}/{k}/{q}": n
            for s, k, q, n in j.group_by("split", "kind", "quality")
            .len()
            .sort("split", "kind", "quality")
            .iter_rows()
        }
    return out


def check_sample_submission(df: pl.DataFrame, sub: pl.DataFrame) -> dict:
    """Submission contract: columns, row count, and Id set equal to the Test file names (aggregates only)."""
    test_names = set(df.filter(pl.col("split") == "test")["image"].to_list())
    out: dict = {"columns": sub.columns, "rows": sub.height}
    if "Id" in sub.columns:
        ids = sub["Id"].cast(pl.String)
        id_set = set(ids.to_list())
        out["duplicate_ids"] = sub.height - len(id_set)
        out["ids_missing_from_test_files"] = len(id_set - test_names)
        out["test_files_missing_from_ids"] = len(test_names - id_set)
        out["id_set_equals_test_files"] = id_set == test_names
    if "Label" in sub.columns:
        out["label_dtype"] = str(sub["Label"].dtype)
        out["label_unique_values"] = int(sub["Label"].n_unique())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--quality-csv", type=Path)
    ap.add_argument("--sample-submission", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--audit-json", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--reuse-manifest", action="store_true", help="re-audit an existing manifest parquet")
    a = ap.parse_args()

    manifest_path = a.out_dir / "files_manifest.parquet"
    if a.reuse_manifest:
        # Re-audit an existing manifest; the file listing must still match it exactly.
        df = pl.read_parquet(manifest_path).sort("relpath")
        if df["relpath"].to_list() != list_files(a.data_dir):
            raise SystemExit(
                "existing manifest does not match the files on disk; rebuild without --reuse-manifest"
            )
    else:
        rels = list_files(a.data_dir)
        root = str(a.data_dir)
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("forkserver")) as ex:
            rows = list(ex.map(inspect_file, [(root, r) for r in rels], chunksize=512))
        df = pl.DataFrame(rows, infer_schema_length=None).sort("relpath")
        a.out_dir.mkdir(parents=True, exist_ok=True)
        df.write_parquet(manifest_path)
    # Identity of the manifest content, independent of parquet encoding details
    content_hash = hashlib.sha256(
        "\n".join(f"{r}\t{s}" for r, s in df.select("relpath", "sha256").iter_rows()).encode()
    ).hexdigest()

    quality = pl.read_csv(a.quality_csv) if a.quality_csv and a.quality_csv.exists() else None
    result = audit(df, quality)
    result["manifest_content_sha256"] = content_hash
    result["quality_csv_sha256"] = (
        hashlib.sha256(a.quality_csv.read_bytes()).hexdigest() if quality is not None else None
    )
    result["quality_csv_present"] = quality is not None
    if a.sample_submission and a.sample_submission.exists():
        result["sample_submission"] = check_sample_submission(df, pl.read_csv(a.sample_submission))
        result["sample_submission_sha256"] = hashlib.sha256(a.sample_submission.read_bytes()).hexdigest()
    a.audit_json.parent.mkdir(parents=True, exist_ok=True)
    a.audit_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
