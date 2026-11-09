"""CPU-only contract tests for the phase-1 readiness tools, using synthetic fixtures."""

import io

import polars as pl
import pytest
from PIL import Image

from tools.readiness.build_manifest import (
    audit,
    changed_files,
    check_sample_submission,
    inspect_file,
    list_files,
)
from tools.readiness.decode_probe import decode_pil, sample, timed
from tools.readiness.host_receipt import _listing
from tools.readiness.jpeg_quality import ZIGZAG, ijg_table, quality_from_table
from tools.readiness.launch_gate import Thresholds, evaluate, parse_psi
from tools.readiness.legacy_split import (
    audit_folds,
    content_linked_families,
    join_quality,
    legacy_family_folds,
)


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


def test_reuse_manifest_detects_same_path_replacement(tmp_path):
    for kind in ("Cover", "UERD"):
        (tmp_path / kind).mkdir()
        for i in range(3):
            Image.new("RGB", (16, 16), (i * 50, 20, 90)).save(
                tmp_path / kind / f"{i:05d}.jpg", "JPEG", quality=90
            )
    df = pl.DataFrame([inspect_file((str(tmp_path), r)) for r in list_files(tmp_path)])
    assert changed_files(df, tmp_path, workers=1) == []
    # Same filename, different bytes (same size is not required to be caught, but test it anyway)
    target = tmp_path / "UERD" / "00001.jpg"
    old = target.read_bytes()
    target.write_bytes(old[:-3] + bytes(b ^ 0xFF for b in old[-3:-2]) + old[-2:])
    assert len(target.read_bytes()) == len(old)
    assert changed_files(df, tmp_path, workers=1) == ["UERD/00001.jpg"]
    Image.new("RGB", (16, 16), (1, 2, 3)).save(tmp_path / "Cover" / "00002.jpg", "JPEG", quality=75)
    assert changed_files(df, tmp_path, workers=1) == ["Cover/00002.jpg", "UERD/00001.jpg"]


def _manifest_and_quality(n_fam=25):
    t = _fixture(n_fam)
    return t.drop("quality").with_columns(split=pl.lit("train")), t.select("image", "kind", "quality")


def test_quality_join_is_one_to_one():
    m, q = _manifest_and_quality()
    assert join_quality(m, q).height == m.height == 100


def test_quality_join_rejects_duplicate_keys():
    m, q = _manifest_and_quality()
    with pytest.raises(ValueError, match="not 1:1"):
        join_quality(m, pl.concat([q, q.head(1)]))  # one duplicate key would expand 100 rows to 101


def test_quality_join_rejects_missing_quality():
    m, q = _manifest_and_quality()
    with pytest.raises(ValueError, match="quality missing"):
        join_quality(m, q.slice(1))


def _host(**kw):
    r = {
        "gpu": {"index": 0, "memory_used_mib": 890, "utilization_gpu_pct": 2},
        "logical_threads": 36,
        "loadavg_1": 8.0,
        "cpu_psi": {"some_avg60": 2.0},
        "io_psi": {"some_avg60": 1.0, "full_avg60": 0.5},
        "free_gib": {"root_fs": 120.0, "data_dir_fs": 120.0},
    }
    r.update(kw)
    return r


def test_launch_gate_go_when_idle():
    assert evaluate(_host(), Thresholds())["decision"] == "GO"


@pytest.mark.parametrize(
    ("override", "failing"),
    [
        ({"gpu": {"index": 0, "memory_used_mib": 4000, "utilization_gpu_pct": 2}}, "gpu_other_used_mib"),
        ({"gpu": {"index": 0, "memory_used_mib": 890, "utilization_gpu_pct": 60}}, "gpu_util_pct"),
        ({"gpu": None}, "gpu_visible"),
        ({"loadavg_1": 57.6}, "cpu_free_threads"),  # phase-1 receipt reading
        ({"cpu_psi": {"some_avg60": 30.8}}, "cpu_psi_some_avg60"),  # phase-1 receipt reading
        ({"cpu_psi": {}}, "cpu_psi_some_avg60"),  # PSI unavailable -> cannot show low occupancy
        ({"io_psi": {"some_avg60": 1.0, "full_avg60": 72.0}}, "io_psi_full_avg60"),
        ({"free_gib": {"root_fs": 44.0, "data_dir_fs": 120.0}}, "disk_free_gib_root_fs"),
    ],
)
def test_launch_gate_no_go_on_each_criterion(override, failing):
    res = evaluate(_host(**override), Thresholds())
    assert res["decision"] == "NO-GO" and res["failing"] == [failing]


def test_parse_psi():
    txt = "some avg10=1.47 avg60=8.18 avg300=19.17 total=1\nfull avg10=0.84 avg60=6.79 avg300=13.94 total=2\n"
    assert parse_psi(txt) == {"some_avg60": 8.18, "full_avg60": 6.79}
    assert parse_psi(None) == {}


def test_launch_gate_uses_measured_idle_cpu_over_loadavg():
    """loadavg counts IO-blocked tasks; a measured idle share decides when present."""
    busy_io = _host(loadavg_1=30.0, cpu_idle_fraction=0.8, cpu_window_s=5.0)
    res = evaluate(busy_io, Thresholds())
    assert res["decision"] == "GO" and res["checks"]["cpu_free_threads"]["value"] == pytest.approx(28.8)
    res = evaluate(_host(loadavg_1=2.0, cpu_idle_fraction=0.3, cpu_window_s=5.0), Thresholds())
    assert res["failing"] == ["cpu_free_threads"]


def test_launch_gate_io_psi_is_advisory_only_with_a_warm_cache_probe():
    hot_io = {"io_psi": {"some_avg60": 29.0, "full_avg60": 25.0}}
    cold = evaluate(_host(**hot_io), Thresholds())
    assert cold["failing"] == ["io_psi_full_avg60", "io_psi_some_avg60"]
    warm = evaluate(_host(**hot_io, read_probe={"files": 600, "files_per_s": 9962.0}), Thresholds())
    assert warm["decision"] == "GO" and warm["advisory_failing"] == ["io_psi_full_avg60", "io_psi_some_avg60"]
    slow = evaluate(_host(**hot_io, read_probe={"files": 600, "files_per_s": 150.0}), Thresholds())
    assert slow["failing"] == ["io_psi_full_avg60", "io_psi_some_avg60"]
    assert "io_read_probe_files_per_s" in slow["advisory_failing"]
