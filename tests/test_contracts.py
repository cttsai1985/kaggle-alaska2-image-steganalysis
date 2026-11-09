"""Config, data, transform, split and model contracts on synthetic fixtures (CPU)."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from alaska2.config import config_hash, load_config, parse_override
from alaska2.data import (
    EpochShuffleSampler,
    ImageDataset,
    apply_d4,
    check_image_size,
    decode_opencv,
    decode_pillow,
    load_manifest,
    normalize_to_tensor,
    train_records,
)
from alaska2.identity import folds_content_sha256, manifest_content_sha256
from alaska2.inputs import prepare_inputs
from alaska2.model import build_model, state_dict_sha256
from alaska2.splits import IdentityError, assert_family_separation, attach_folds, load_folds, subset_families

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
D4 = [{"op": "hflip", "p": 0.5}, {"op": "vflip", "p": 0.5}, {"op": "rot90", "p": 0.5}]


# --- config ---------------------------------------------------------------------------------------------


def test_configs_load_and_inherit():
    base = load_config(CONFIGS / "baseline_effb0_fold1.toml")
    smoke = load_config(CONFIGS / "smoke_effb0.toml")
    assert base["purpose"] == "baseline" and smoke["purpose"] == "smoke"
    assert smoke["train"]["max_steps_per_epoch"] == 200 and smoke["train"]["lr"] == base["train"]["lr"]
    assert smoke["data"] == base["data"] and config_hash(smoke) != config_hash(base)
    assert base["train"]["num_workers"] <= 16 and base["data"]["image_size"] == 512
    assert base["model"]["weights_revision"] and base["model"]["weights_sha256"]


def test_config_hash_ignores_source_path_but_not_content():
    a = load_config(CONFIGS / "baseline_effb0_fold1.toml")
    b = load_config(CONFIGS / "baseline_effb0_fold1.toml", ["train.lr=1e-4"])
    assert config_hash(a) == config_hash(load_config(CONFIGS / "baseline_effb0_fold1.toml"))
    assert config_hash(a) != config_hash(b) and b["train"]["lr"] == 1e-4


def test_parse_override_types():
    assert parse_override("train.epochs=3") == ("train.epochs", 3)
    assert parse_override('data.root="/x y"') == ("data.root", "/x y")
    assert parse_override("data.root=/plain") == ("data.root", "/plain")


@pytest.mark.parametrize(
    "override",
    [
        'transforms.train=[{op="resize", p=1.0}]',
        "train.num_workers=17",
        'train.device="auto"',
        'data.classes=["JMiPOD", "Cover", "JUNIWARD", "UERD"]',
        'purpose="leaderboard"',
    ],
)
def test_config_rejects_unsupported_values(override):
    with pytest.raises(ValueError):
        load_config(CONFIGS / "baseline_effb0_fold1.toml", [override])


# --- transforms and decoding ----------------------------------------------------------------------------


def test_d4_ops_only_permute_pixels_and_keep_jpeg_block_grid():
    rng = np.random.default_rng(0)
    blocks = rng.integers(0, 256, size=(8, 8, 3))
    img = np.kron(blocks, np.ones((8, 8, 1))).astype(np.uint8)  # 64x64, constant per 8x8 JPEG block
    for seed in range(40):
        out = apply_d4(img, D4, np.random.default_rng(seed))
        assert out.shape == img.shape and out.dtype == img.dtype
        assert np.array_equal(np.sort(out, axis=None), np.sort(img, axis=None))  # same values, moved
        tiles = out.reshape(8, 8, 8, 8, 3)
        assert (tiles == tiles[:, :1, :, :1]).all()  # every 8x8 block still constant: grid preserved


def test_d4_covers_all_eight_symmetries_and_is_seed_deterministic():
    img = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    seen = {apply_d4(img, D4, np.random.default_rng(s)).tobytes() for s in range(400)}
    assert len(seen) == 8
    a = apply_d4(img, D4, np.random.default_rng([1, 2, 3]))
    assert np.array_equal(a, apply_d4(img, D4, np.random.default_rng([1, 2, 3])))


def test_normalize_matches_legacy_albumentations_formula():
    img = np.random.default_rng(0).integers(0, 256, size=(4, 4, 3)).astype(np.uint8)
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    x = normalize_to_tensor(img, mean, std, 255.0)
    legacy = (img.astype(np.float32) - np.array(mean) * 255.0) / (np.array(std) * 255.0)  # A.Normalize
    assert x.shape == (3, 4, 4) and x.dtype == torch.float32
    assert np.allclose(x.numpy().transpose(1, 2, 0), legacy, atol=1e-5)


def test_decoders_return_rgb_in_the_same_order(tmp_path):
    from PIL import Image

    arr = np.zeros((16, 16, 3), np.uint8)
    arr[..., 0] = 220  # red
    Image.fromarray(arr).save(tmp_path / "r.jpg", quality=95)
    a, b = decode_opencv(str(tmp_path / "r.jpg")), decode_pillow(str(tmp_path / "r.jpg"))
    assert a.shape == b.shape == (16, 16, 3)
    assert a[..., 0].mean() > 200 and a[..., 2].mean() < 30
    assert np.abs(a.astype(int) - b.astype(int)).max() <= 2  # different IDCTs, same image


# --- records and splits ---------------------------------------------------------------------------------


def test_records_are_deterministic_and_in_class_order(fixture_data):
    m = load_manifest(fixture_data["root"] / "manifest" / "files_manifest.parquet")
    r1 = train_records(m)
    r2 = train_records(m.sample(fraction=1.0, shuffle=True, seed=1))
    assert r1.equals(r2)
    assert r1["kind"].head(4).to_list() == ["Cover", "JMiPOD", "JUNIWARD", "UERD"]
    assert r1["label"].head(4).to_list() == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        check_image_size(r1, 512)


def test_hashes_match_phase1_definitions(fixture_data):
    import hashlib

    m, f = fixture_data["manifest"], fixture_data["folds"]
    lines = "\n".join(f"{r}\t{s}" for r, s in m.select("relpath", "sha256").sort("relpath").iter_rows())
    assert manifest_content_sha256(m) == hashlib.sha256(lines.encode()).hexdigest()
    lines = "\n".join(f"{a}\t{b}" for a, b in f.sort("family_id").iter_rows())
    assert folds_content_sha256(f) == hashlib.sha256(lines.encode()).hexdigest()


def test_fold_hash_mismatch_is_rejected(fixture_data):
    path = fixture_data["root"] / "manifest" / "legacy_family_folds.parquet"
    folds, h = load_folds(path, None)
    assert load_folds(path, h)[1] == h
    with pytest.raises(IdentityError):
        load_folds(path, "0" * 64)


def test_prepare_inputs_separates_families(fixture_cfg):
    inputs = prepare_inputs(fixture_cfg())
    r = inputs.records
    assert set(r["role"].unique()) == {"fit", "valid"}
    roles = r.group_by("family_id").agg(pl.col("role").n_unique())
    assert (roles["role"] == 1).all()
    assert inputs.audit["valid"]["complete_families"] == inputs.audit["valid"]["families"]
    assert sum(inputs.audit["valid"]["by_kind"].values()) == inputs.audit["valid"]["rows"]
    assert inputs.identities["population"] == "full_data_single_holdout"


def test_identity_check_rejects_wrong_expected_manifest(fixture_cfg):
    with pytest.raises(IdentityError):
        prepare_inputs(fixture_cfg('data.expected_manifest_sha256="' + "f" * 64 + '"'))


def test_family_or_content_crossing_raises():
    base = pl.DataFrame(
        {"family_id": ["1", "1", "2"], "sha256": ["a", "b", "c"], "role": ["fit", "fit", "valid"]}
    )
    assert_family_separation(base)
    with pytest.raises(IdentityError, match="families cross"):
        assert_family_separation(base.with_columns(pl.Series("role", ["fit", "valid", "valid"])))
    with pytest.raises(IdentityError, match="byte-identical"):
        assert_family_separation(base.with_columns(pl.Series("sha256", ["a", "b", "a"])))


def test_attach_folds_requires_every_family():
    train = pl.DataFrame({"family_id": ["1", "2"], "x": [0, 1]})
    with pytest.raises(IdentityError):
        attach_folds(train, pl.DataFrame({"family_id": ["1"], "fold": pl.Series([1], dtype=pl.Int32)}))


def test_subset_is_deterministic_and_labeled(fixture_cfg):
    fams = pl.Series([f"{i:05d}" for i in range(50)])
    assert subset_families(fams, 10, 7) == subset_families(fams, 10, 7)
    assert subset_families(fams, 10, 7) != subset_families(fams, 10, 8)
    inputs = prepare_inputs(fixture_cfg("split.max_valid_families=2"))
    assert inputs.audit["valid"]["families"] == 2
    assert inputs.identities["population"] == "explicit_family_subset"


def test_dataset_and_sampler_are_resumable(fixture_cfg):
    cfg = fixture_cfg()
    inputs = prepare_inputs(cfg)
    ds = ImageDataset(inputs.records.filter(pl.col("role") == "fit"), inputs.images_dir, cfg, train=True)
    x, y, row = ds[(3, 1)]
    assert x.shape == (3, 64, 64) and y == int(ds.labels[3]) and row == 3
    assert torch.equal(x, ds[(3, 1)][0])  # augmentation = f(seed, epoch, row)
    s = EpochShuffleSampler(len(ds), seed=5)
    s.set_epoch(2)
    first = list(s)
    s2 = EpochShuffleSampler(len(ds), seed=5, epoch=2)
    assert first == list(s2) and sorted(i for i, _ in first) == list(range(len(ds)))


# --- model ----------------------------------------------------------------------------------------------


def test_model_has_four_class_head_and_stable_hash(fixture_cfg):
    model, ident = build_model(fixture_cfg(), pretrained=False)
    with torch.no_grad():
        assert model.eval()(torch.zeros(2, 3, 64, 64)).shape == (2, 4)
    h = state_dict_sha256(model)
    assert h == state_dict_sha256({k: v.clone() for k, v in model.state_dict().items()})
    with torch.no_grad():
        next(model.parameters()).add_(1e-3)
    assert state_dict_sha256(model) != h
    assert ident["timm_pretrained_mean"] == [0.485, 0.456, 0.406]


def test_pretrained_weights_must_match_pinned_hash(monkeypatch, tmp_path):
    import huggingface_hub

    from alaska2.model import WeightsIdentityError, resolve_pretrained

    fake = tmp_path / "model.safetensors"
    fake.write_bytes(b"not the pinned weights")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(fake))
    mcfg = load_config(CONFIGS / "baseline_effb0_fold1.toml")["model"]
    with pytest.raises(WeightsIdentityError, match="sha256"):
        resolve_pretrained(mcfg)
    with pytest.raises(WeightsIdentityError, match="revision"):
        resolve_pretrained({**mcfg, "weights_revision": ""})
