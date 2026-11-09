"""CPU-only contract tests for the phase-1 readiness tools, using synthetic fixtures."""

import io

import polars as pl
import pytest
from PIL import Image

from tools.readiness.build_manifest import audit, inspect_file
from tools.readiness.jpeg_quality import ZIGZAG, ijg_table, quality_from_table
from tools.readiness.legacy_split import audit_folds, legacy_family_folds


def test_ijg_table_known_values():
    assert ijg_table(50)[:4] == (16, 11, 10, 16)  # scale 100 = base table
    assert ijg_table(100) == (1,) * 64
    assert ijg_table(75)[0] == 8


@pytest.mark.parametrize("q", [75, 90, 95])
def test_quality_from_table_both_orders(q):
    nat = ijg_table(q)
    assert quality_from_table(nat) == q
    assert quality_from_table([nat[i] for i in ZIGZAG]) == q
    assert quality_from_table([7] * 64) is None


@pytest.mark.parametrize("q", [75, 90, 95])
def test_quality_from_real_pillow_jpeg(tmp_path, q):
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (120, 30, 200)).save(buf, "JPEG", quality=q)
    (tmp_path / "Cover").mkdir()
    (tmp_path / "Cover" / "00001.jpg").write_bytes(buf.getvalue())
    row = inspect_file((str(tmp_path), "Cover/00001.jpg"))
    assert row["q_from_table"] == q
    assert (row["family_id"], row["label"], row["split"]) == ("00001", 0, "train")


def _fixture(n_fam=20):
    rows = []
    for i in range(1, n_fam + 1):
        for lab, kind in enumerate(("Cover", "JMiPOD", "JUNIWARD", "UERD")):
            rows.append(
                {
                    "relpath": f"{kind}/{i:05d}.jpg",
                    "image": f"{i:05d}.jpg",
                    "kind": kind,
                    "family_id": f"{i:05d}",
                    "label": lab,
                    "quality": (75, 90, 95)[i % 3],
                }
            )
    return pl.DataFrame(rows)


def test_legacy_folds_keep_families_together():
    train = _fixture(25)
    folds = legacy_family_folds(train)
    assert folds.height == 25 and set(folds["fold"]) == {1, 2, 3, 4, 5}
    res = audit_folds(train, folds)
    assert res["families_in_multiple_folds"] == 0 and res["rows_without_fold"] == 0
    for f in res["per_fold"]:
        assert f["Cover"] == f["JMiPOD"] == f["JUNIWARD"] == f["UERD"]


def test_legacy_folds_reject_inconsistent_quality():
    train = _fixture(10).with_columns(
        pl.when((pl.col("family_id") == "00001") & (pl.col("kind") == "UERD"))
        .then(1)
        .otherwise(pl.col("quality"))
        .alias("quality")
    )
    with pytest.raises(ValueError):
        legacy_family_folds(train)


def test_audit_flags_incomplete_family_and_duplicates():
    base = _fixture(3).drop("quality")
    m = base.with_columns(
        split=pl.lit("train"),
        n_bytes=pl.lit(10),
        sha256=pl.col("relpath"),
        width=pl.lit(512),
        height=pl.lit(512),
        mode=pl.lit("RGB"),
        format=pl.lit("JPEG"),
        q_from_table=pl.lit(90),
        open_error=pl.lit(None, dtype=pl.String),
    )
    m = m.filter(pl.col("relpath") != "UERD/00002.jpg")  # missing variant
    m = m.with_columns(  # stego byte-identical to its cover
        pl.when(pl.col("relpath") == "JMiPOD/00003.jpg")
        .then(pl.lit("Cover/00003.jpg"))
        .otherwise("sha256")
        .alias("sha256")
    )
    out = audit(m, None)
    assert out["families_incomplete"] == 1
    assert out["stego_identical_to_own_cover"]["JMiPOD"] == 1
    assert out["byte_duplicate_groups"] == 1
