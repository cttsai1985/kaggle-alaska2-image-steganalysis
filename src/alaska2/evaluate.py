"""Held-out evaluation, unlabeled prediction and independent checkpoint verification.

Every prediction file carries image id, family id, kind, fold, quality, the model weight hash, the
checkpoint file hash, raw logits and class probabilities in the legacy class order, plus
``score = 1 - p_Cover``. A JSON sidecar records how the predictions were produced so ``verify`` can rebuild
them from the checkpoint alone and compare within a declared tolerance.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

from alaska2.constants import CLASSES, PURPOSE_LABELS
from alaska2.data import ImageDataset
from alaska2.identity import runtime_versions, sha256_file
from alaska2.inputs import prepare_inputs
from alaska2.metrics import binary_score, evaluation_report, softmax, weighted_auc
from alaska2.model import build_model, load_checkpoint, state_dict_sha256
from alaska2.runtime import autocast, make_loader, resolve_device, utc_now, write_json
from alaska2.splits import IdentityError

RUNTIME_OVERRIDES = ("data.root", "train.device", "train.num_workers", "eval.batch_size", "eval.amp_dtype")
CHECKED_IDENTITIES = ("input_manifest_sha256", "split_folds_sha256", "population_sha256")


def runtime_config(ckpt_cfg: dict, overrides: dict | None) -> dict:
    """The checkpoint's own config; only runtime keys (paths, device, workers, eval batch) may change."""
    cfg = copy.deepcopy(ckpt_cfg)
    for key, value in (overrides or {}).items():
        if value is None:
            continue
        if key not in RUNTIME_OVERRIDES:
            raise ValueError(f"{key} is not a runtime override; allowed: {RUNTIME_OVERRIDES}")
        sect, name = key.split(".")
        cfg[sect][name] = value
    return cfg


def load_model(ckpt_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    ckpt = load_checkpoint(ckpt_path)
    model, _ = build_model(ckpt["config"], pretrained=False)
    model.load_state_dict(ckpt["model"])
    got = state_dict_sha256(model)
    if got != ckpt["model_sha256"]:
        raise IdentityError(f"reloaded weights hash {got} != checkpoint model_sha256 {ckpt['model_sha256']}")
    model.to(device).eval()
    if ckpt["config"]["train"].get("channels_last"):
        model = model.to(memory_format=torch.channels_last)
    return model, ckpt


@torch.no_grad()
def infer(model, records: pl.DataFrame, images_dir: Path, cfg: dict, device: torch.device) -> np.ndarray:
    ds = ImageDataset(records, images_dir, cfg, train=False)
    loader = make_loader(ds, cfg, int(cfg["eval"]["batch_size"]), train=False)
    out = np.zeros((len(ds), len(CLASSES)), dtype=np.float32)
    amp = cfg["eval"].get("amp_dtype", cfg["train"]["amp_dtype"])
    for x, _, rows in loader:
        x = x.to(device, non_blocking=True)
        if cfg["train"].get("channels_last"):
            x = x.contiguous(memory_format=torch.channels_last)
        with autocast(device, amp):
            logits = model(x)
        out[rows.numpy()] = logits.float().cpu().numpy()
    return out


def _check_identities(ckpt: dict, inputs_ids: dict) -> None:
    diff = [k for k in CHECKED_IDENTITIES if ckpt["identities"].get(k) != inputs_ids.get(k)]
    if diff:
        raise IdentityError(f"inputs differ from the checkpoint's training identities on {diff}; rejected")


def prediction_frame(records: pl.DataFrame, logits: np.ndarray, ckpt: dict, ckpt_sha: str) -> pl.DataFrame:
    probs = softmax(logits)
    cols = {f"logit_{c}": logits[:, i].astype(np.float32) for i, c in enumerate(CLASSES)}
    cols |= {f"p_{c}": probs[:, i] for i, c in enumerate(CLASSES)}
    keep = [
        c
        for c in ("image", "family_id", "kind", "label", "fold", "quality", "relpath")
        if c in records.columns
    ]
    return records.select(keep).with_columns(
        **{k: pl.Series(v) for k, v in cols.items()},
        score=pl.Series(binary_score(probs)),
        model_sha256=pl.lit(ckpt["model_sha256"]),
        checkpoint_sha256=pl.lit(ckpt_sha),
    )


def assert_membership(pred: pl.DataFrame, expected: pl.DataFrame) -> dict:
    """Exact ID coverage: same (image, kind) set as the declared population, no duplicates."""
    key = ["image", "kind"]
    if pred.select(key).is_duplicated().any():
        raise IdentityError("duplicate (image, kind) rows in predictions")
    got = set(pred.select(key).iter_rows())
    want = set(expected.select(key).iter_rows())
    if got != want:
        raise IdentityError(f"prediction IDs differ: {len(want - got)} missing, {len(got - want)} extra")
    return {"rows": pred.height, "expected_rows": expected.height, "missing": 0, "extra": 0, "duplicates": 0}


CANONICAL_COLUMNS = ("family_id", "label", "fold", "quality")


def expected_population(inputs, split: str) -> pl.DataFrame:
    """The population a prediction file must cover exactly: holdout rows, or the unlabeled test set."""
    if split == "valid":
        return inputs.records.filter(pl.col("role") == "valid")
    if split == "test":
        return inputs.test
    raise ValueError("split must be valid or test")


def canonical_mismatches(saved: pl.DataFrame, canonical: pl.DataFrame) -> dict:
    """Per-column count of rows whose saved value differs from the canonical record (row-aligned frames)."""
    out = {}
    for c in CANONICAL_COLUMNS:
        if c not in canonical.columns:
            continue  # the unlabeled test set has no fold
        if c not in saved.columns:
            out[c] = "column missing"
            continue
        n = int(saved[c].cast(canonical[c].dtype, strict=False).ne_missing(canonical[c]).sum())
        if n:
            out[c] = n
    return out


def evaluate(
    ckpt_path: Path, out_dir: Path | None = None, overrides: dict | None = None, split: str = "valid"
) -> Path:
    """split = valid: holdout predictions + metrics. split = test: unlabeled predictions only."""
    ckpt_path = Path(ckpt_path)
    raw = load_checkpoint(ckpt_path)
    cfg = runtime_config(raw["config"], overrides)
    device = resolve_device(cfg["train"]["device"])
    inputs = prepare_inputs(cfg)
    _check_identities(raw, inputs.identities)
    model, ckpt = load_model(ckpt_path, device)
    ckpt_sha = sha256_file(ckpt_path)

    records = expected_population(inputs, split)
    if split == "test":
        records = records.with_columns(pl.lit(None, dtype=pl.Int32).alias("fold"))
    t0 = time.perf_counter()
    logits = infer(model, records, inputs.images_dir, cfg, device)
    wall = time.perf_counter() - t0
    pred = prediction_frame(records, logits, ckpt, ckpt_sha)
    membership = assert_membership(pred, records)

    run_dir = ckpt_path.parent.parent
    out_dir = Path(out_dir) if out_dir else run_dir / "predictions"
    stem = f"{split}_epoch{int(ckpt['epoch']):03d}"
    pred_path = out_dir / f"{stem}.parquet"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred.write_parquet(pred_path)
    meta = {
        "kind": "predictions",
        "split": split,
        "purpose": cfg["purpose"],
        "result_label": PURPOSE_LABELS[cfg["purpose"]],
        "created_at": utc_now(),
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "model_sha256": ckpt["model_sha256"],
        "epoch": int(ckpt["epoch"]),
        "identities": {**ckpt["identities"], "evaluated_inputs": inputs.identities},
        "runtime_overrides": {k: v for k, v in (overrides or {}).items() if v is not None},
        "device": str(device),
        "eval_amp_dtype": cfg["eval"].get("amp_dtype", cfg["train"]["amp_dtype"]),
        "decoder": cfg["data"]["decoder"],
        "transforms_valid": cfg["transforms"]["valid"],
        "normalize": {
            k: cfg["transforms"][k] for k in ("normalize_mean", "normalize_std", "max_pixel_value")
        },
        "classes": list(CLASSES),
        "score_definition": "score = 1 - p_Cover (softmax over logits in class order)",
        "membership": membership,
        "predictions_sha256": sha256_file(pred_path),
        "inference_wall_s": round(wall, 2),
        "versions": runtime_versions(),
        "verify_tolerance": cfg["verify"],
    }
    if split == "valid":
        meta["metrics"] = evaluation_report(pred)
    write_json(pred_path.with_suffix(".json"), meta)
    return pred_path


def verify(
    pred_path: Path,
    ckpt_path: Path | None = None,
    overrides: dict | None = None,
    out_path: Path | None = None,
) -> dict:
    """Rebuild saved predictions from the checkpoint alone and compare within the declared tolerance."""
    pred_path = Path(pred_path)
    meta = json.loads(pred_path.with_suffix(".json").read_text())
    prediction_sha = sha256_file(pred_path)
    saved = pl.read_parquet(pred_path)
    ckpt_path = Path(ckpt_path or meta["checkpoint"])
    checkpoint_sha = sha256_file(ckpt_path)
    raw = load_checkpoint(ckpt_path)
    given = {k: v for k, v in (overrides or {}).items() if v is not None}
    merged = {**meta.get("runtime_overrides", {}), **given}
    cfg = runtime_config(raw["config"], merged)
    tol = raw["config"]["verify"]
    device = resolve_device(cfg["train"]["device"])
    checks: dict = {
        "predictions_sha256_matches": prediction_sha == meta["predictions_sha256"],
        "checkpoint_sha256_matches": checkpoint_sha == meta["checkpoint_sha256"],
    }
    model, ckpt = load_model(ckpt_path, device)  # raises if the reloaded weights hash differs
    checks["model_sha256_matches"] = ckpt["model_sha256"] == meta["model_sha256"] and bool(
        (saved["model_sha256"] == meta["model_sha256"]).all()
    )
    inputs = prepare_inputs(cfg)
    _check_identities(ckpt, inputs.identities)
    expected = expected_population(inputs, meta["split"])
    # Exact holdout (or test) coverage before any inference: no fit rows, no missing rows, no duplicates.
    membership = assert_membership(saved, expected)
    records = saved.select("image", "kind").join(
        expected, on=["image", "kind"], how="left", validate="1:1", maintain_order="left"
    )
    mism = canonical_mismatches(saved, records)
    if mism:
        raise IdentityError(f"saved label/fold/family/quality differ from the canonical inputs: {mism}")
    checks["saved_rows_match_canonical_inputs"] = True
    logits = infer(model, records, inputs.images_dir, cfg, device)
    probs = softmax(logits)
    saved_p = saved.select([f"p_{c}" for c in CLASSES]).to_numpy()
    prob_delta = float(np.abs(probs - saved_p).max())
    logit_delta = float(np.abs(logits - saved.select([f"logit_{c}" for c in CLASSES]).to_numpy()).max())
    checks["score_matches_probabilities"] = bool(
        np.array_equal(saved["score"].to_numpy(), binary_score(saved_p))
    )
    checks["prediction_file_unchanged"] = sha256_file(pred_path) == prediction_sha
    checks["checkpoint_file_unchanged"] = sha256_file(ckpt_path) == checkpoint_sha
    metrics = evaluation_report(saved) if meta["split"] == "valid" else None
    if metrics is not None:
        checks["metrics_match_saved_predictions"] = metrics == meta.get("metrics")
    res = {
        "kind": "verify",
        "predictions_sha256": prediction_sha,
        "checkpoint_sha256": checkpoint_sha,
        "model_sha256": ckpt["model_sha256"],
        "predictions": str(pred_path),
        "checkpoint": str(ckpt_path),
        "purpose": meta["purpose"],
        "result_label": meta["result_label"],
        "device": str(device),
        "rows": saved.height,
        "membership": membership,
        "tolerance": tol,
        "observed": {"max_abs_prob_delta": prob_delta, "max_abs_logit_delta": logit_delta},
        "checks": checks,
        "verified_at": utc_now(),
    }
    ok = all(checks.values()) and prob_delta <= float(tol["prob_atol"])
    if meta["split"] == "valid":
        res["metrics"] = metrics
        y = records["label"].to_numpy() != 0  # canonical labels, never the saved column
        m_saved = weighted_auc(y, saved["score"].to_numpy())
        m_new = weighted_auc(y, binary_score(probs))
        res["observed"]["weighted_auc_saved"] = m_saved
        res["observed"]["weighted_auc_rebuilt"] = m_new
        res["observed"]["weighted_auc_delta"] = abs(m_new - m_saved)
        ok = ok and abs(m_new - m_saved) <= float(tol["metric_atol"])
    res["passed"] = bool(ok)
    write_json(Path(out_path) if out_path else pred_path.with_name(pred_path.stem + ".verify.json"), res)
    return res
