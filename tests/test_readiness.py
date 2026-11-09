"""CPU-only contract tests for the phase-1 readiness tools, using synthetic fixtures."""

import io

import polars as pl
import pytest
from PIL import Image

from tools.readiness.build_manifest import audit, check_sample_submission, inspect_file
from tools.readiness.decode_probe import decode_pil, sample, timed
from tools.readiness.host_receipt import _listing
from tools.readiness.jpeg_quality import ZIGZAG, ijg_table, quality_from_table
from tools.readiness.legacy_split import audit_folds, content_linked_families, legacy_family_folds


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
    with pytest.raises(ValueError, match="quality differs within a family"):
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
    m = m.with_columns(  # a Cover byte-identical to another family's Cover
        pl.when(pl.col("relpath") == "Cover/00002.jpg")
        .then(pl.lit("Cover/00001.jpg"))
        .otherwise("sha256")
        .alias("sha256")
    )
    out = audit(m, None)
    assert out["families_incomplete"] == 1
    assert out["stego_identical_to_own_cover"]["JMiPOD"] == 1
    assert out["stego_identical_to_own_cover_by_kind_table_quality"] == {"JMiPOD/90": 1}
    assert out["byte_duplicate_groups"] == 2
    assert out["byte_duplicate_kind_patterns"] == {"Cover+Cover": 1, "Cover+JMiPOD": 1}
    assert out["cross_family_byte_duplicate_groups"] == 1
    assert out["cross_family_byte_duplicate_kind_patterns"] == {"Cover+Cover": 1}


def test_sample_submission_contract():
    m = pl.DataFrame({"split": ["test", "test", "train"], "image": ["0001.jpg", "0002.jpg", "00001.jpg"]})
    ok = check_sample_submission(m, pl.DataFrame({"Id": ["0001.jpg", "0002.jpg"], "Label": [0, 0]}))
    assert ok["id_set_equals_test_files"] and ok["duplicate_ids"] == 0 and ok["rows"] == 2
    bad = check_sample_submission(m, pl.DataFrame({"Id": ["0001.jpg", "0001.jpg"], "Label": [0, 0]}))
    assert not bad["id_set_equals_test_files"]
    assert bad["duplicate_ids"] == 1 and bad["test_files_missing_from_ids"] == 1


def test_decode_probe_on_synthetic_jpegs(tmp_path):
    rels = []
    for kind in ("Cover", "UERD"):
        (tmp_path / kind).mkdir()
        for i in range(4):
            Image.new("RGB", (32, 24), (i * 40, 10, 90)).save(tmp_path / kind / f"{i:05d}.jpg", "JPEG")
            rels.append(f"{kind}/{i:05d}.jpg")
    man = tmp_path / "m.parquet"
    pl.DataFrame({"relpath": rels, "kind": [r.split("/")[0] for r in rels]}).write_parquet(man)
    picked = sample(man, 4, seed=0)
    assert len(picked) == 4 and sum(p.startswith("Cover/") for p in picked) == 2
    assert picked == sample(man, 4, seed=0)  # deterministic
    for w in (1, 2):
        r = timed(decode_pil, picked, str(tmp_path), w)
        assert r["images"] == 4 and r["pixels_checked"] == 4 * 32 * 24


def test_content_linked_families_across_folds():
    train = _fixture(25).with_columns(sha256=pl.col("relpath"))
    train = train.with_columns(  # family 00002's Cover has the same bytes as family 00001's Cover
        pl.when(pl.col("relpath") == "Cover/00002.jpg")
        .then(pl.lit("Cover/00001.jpg"))
        .otherwise("sha256")
        .alias("sha256")
    )
    folds = legacy_family_folds(train)
    res = content_linked_families(train, folds)
    f1, f2 = (folds.filter(pl.col("family_id") == f)["fold"].item() for f in ("00001", "00002"))
    assert res["byte_identical_groups_spanning_families"] == 1
    assert res["of_which_split_across_folds"] == int(f1 != f2)


def test_kaggle_listing_parse_drops_token():
    out = "Next Page Token = SECRETISH\nname,size,creationDate\nCover/00001.jpg,1,x\nCover/00002.jpg,2,x\n"
    res = _listing(out)
    assert res == {"first_page_rows": 2, "more_pages": True}
    assert "SECRETISH" not in str(res)
    assert _listing("name,size,creationDate\nimage_quality.csv,1,x\n") == {
        "first_page_rows": 1,
        "more_pages": False,
    }
