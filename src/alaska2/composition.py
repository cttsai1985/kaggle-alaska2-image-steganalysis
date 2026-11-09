"""Validated prediction provenance, fixed probability means and sample-ordered CSV export."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from alaska2.constants import CLASSES
from alaska2.evaluate import assert_membership, canonical_mismatches
from alaska2.identity import sha256_file
from alaska2.metrics import binary_score
from alaska2.model import load_checkpoint, state_dict_sha256
from alaska2.splits import IdentityError


@dataclass(frozen=True)
class VerifiedPrediction:
    frame: pl.DataFrame
    provenance: dict


def validated_predictions(path: Path, checkpoint: Path, expected: pl.DataFrame) -> VerifiedPrediction:
    """Require canonical membership, an actual producing checkpoint and its bound reconstruction receipt."""
    path, checkpoint = Path(path), Path(checkpoint)
    meta = json.loads(path.with_suffix(".json").read_text())
    verification = json.loads(path.with_name(path.stem + ".verify.json").read_text())
    prediction_sha, checkpoint_sha = sha256_file(path), sha256_file(checkpoint)
    raw = load_checkpoint(checkpoint)
    if meta.get("classes") != list(CLASSES) or raw.get("classes") != list(CLASSES):
        raise ValueError("class columns/order differ from the model contract")
    if meta.get("purpose") != raw["config"]["purpose"] or meta.get("split") not in ("valid", "test"):
        raise ValueError("prediction purpose/split differs from the producing model")
    identities = {
        key: meta["identities"].get(key)
        for key in ("input_manifest_sha256", "split_folds_sha256", "population_sha256")
    }
    if any(value != raw["identities"].get(key) for key, value in identities.items()):
        raise ValueError("prediction population/input/split identity differs from the checkpoint")
    actual = {
        "predictions_sha256": prediction_sha,
        "checkpoint_sha256": checkpoint_sha,
        "model_sha256": state_dict_sha256(raw["model"]),
    }
    if any(meta.get(key) != value or verification.get(key) != value for key, value in actual.items()):
        raise ValueError("prediction, checkpoint or verification bytes/identity differ")
    if raw["model_sha256"] != actual["model_sha256"] or not verification.get("passed"):
        raise ValueError("checkpoint weights or independent verification failed")
    required = {
        "predictions_sha256_matches",
        "checkpoint_sha256_matches",
        "model_sha256_matches",
        "saved_rows_match_canonical_inputs",
        "score_matches_probabilities",
        "prediction_file_unchanged",
        "checkpoint_file_unchanged",
    }
    if meta["split"] == "valid":
        required.add("metrics_match_saved_predictions")
    if not all(verification.get("checks", {}).get(key) is True for key in required):
        raise ValueError("independent verification is missing required checks")
    frame = pl.read_parquet(path)
    try:
        assert_membership(frame, expected)
    except IdentityError as exc:
        raise ValueError("prediction membership differs from the canonical population") from exc
    if not frame.select("image", "kind").equals(expected.select("image", "kind")):
        raise ValueError("prediction row ordering differs from the canonical population")
    if canonical_mismatches(frame, expected):
        raise ValueError("prediction family/label/fold/quality differs from canonical records")
    for key in ("model_sha256", "checkpoint_sha256"):
        if not bool((frame[key] == actual[key]).all()):
            raise ValueError(f"row-level {key} differs from the producing model")
    probs = frame.select([f"p_{kind}" for kind in CLASSES]).to_numpy()
    if (
        not np.isfinite(probs).all()
        or (probs < 0).any()
        or (probs > 1).any()
        or not np.allclose(probs.sum(axis=1), 1, atol=1e-8, rtol=0)
    ):
        raise ValueError("probabilities must be finite, normalized and within [0,1]")
    if not np.array_equal(frame["score"].to_numpy(), binary_score(probs)):
        raise ValueError("score differs from 1 - P(Cover)")
    fold = int(raw["config"]["split"]["holdout_fold"])
    if meta["split"] == "valid" and expected["fold"].unique().to_list() != [fold]:
        raise ValueError("these rows were not held out by the producing model")
    if (
        sha256_file(path) != actual["predictions_sha256"]
        or sha256_file(checkpoint) != actual["checkpoint_sha256"]
    ):
        raise ValueError("prediction/checkpoint changed while its provenance was checked")
    provenance = {
        **actual,
        "identities": identities,
        "classes": list(CLASSES),
        "purpose": meta["purpose"],
        "split": meta["split"],
        "producing_fold": fold,
        "rows": frame.height,
        "held_out": meta["split"] == "valid",
        "checkpoint_epoch": raw["epoch"],
        "training_complete": raw["epoch"] >= raw["config"]["train"]["epochs"],
    }
    return VerifiedPrediction(frame, provenance)


def probability_mean(sources: Sequence[VerifiedPrediction]) -> tuple[pl.DataFrame, dict]:
    """Use a predeclared equal mean only; never infer OOF provenance or choose weights on these labels."""
    if len(sources) < 2 or len({s.provenance["model_sha256"] for s in sources}) != len(sources):
        raise ValueError("a mean requires at least two distinct producing models")
    first = sources[0]
    for source in sources[1:]:
        if any(
            source.provenance[key] != first.provenance[key]
            for key in ("classes", "identities", "split", "purpose")
        ):
            raise ValueError("ensemble input identity/order/population differs")
        if not source.frame.select("image", "kind").equals(first.frame.select("image", "kind")):
            raise ValueError("ensemble row ordering differs")
    columns = [f"p_{kind}" for kind in CLASSES]
    probs = np.stack([s.frame.select(columns).to_numpy() for s in sources]).mean(axis=0)
    frame = first.frame.select(
        [
            column
            for column in ("image", "family_id", "kind", "label", "fold", "quality", "relpath")
            if column in first.frame.columns
        ]
    ).with_columns(
        **{column: pl.Series(probs[:, i]) for i, column in enumerate(columns)},
        score=pl.Series(binary_score(probs)),
    )
    receipt = {
        "algorithm": "unweighted_probability_mean",
        "weights": [1 / len(sources)] * len(sources),
        "sources": [s.provenance for s in sources],
        "rows": frame.height,
        "held_out_rows": frame.height if first.provenance["held_out"] else 0,
        "learned_stacking": "deferred; no meta-model is fitted",
    }
    return frame, receipt


def write_submission(predictions: pl.DataFrame, sample_path: Path, output: Path) -> dict:
    """Use the actual supplied sample schema and order, with exactly one finite probability per test ID."""
    if Path(output).suffix.lower() != ".csv":
        raise ValueError("submission output must have a CSV extension")
    if Path(output).resolve() == Path(sample_path).resolve():
        raise ValueError("submission output cannot overwrite its authoritative sample")
    sample = pl.read_csv(sample_path, schema_overrides={"Id": pl.String})
    if sample.columns != ["Id", "Label"] or sample.is_empty() or sample["Id"].is_duplicated().any():
        raise ValueError("sample must have unique Id,Label rows")
    if predictions["image"].is_duplicated().any() or set(predictions["image"]) != set(sample["Id"]):
        raise ValueError("submission IDs must match the actual sample exactly once")
    scores = predictions["score"].to_numpy()
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError("submission probabilities must be finite and within [0,1]")
    ordered = sample.select("Id").join(
        predictions.select(pl.col("image").alias("Id"), pl.col("score").alias("Label")),
        on="Id",
        how="left",
        validate="1:1",
        maintain_order="left",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_symlink():
        raise ValueError("submission output cannot be a symlink")
    with tempfile.TemporaryDirectory(prefix=".alaska2-export-", dir=output.parent) as temporary:
        pending = Path(temporary) / "submission.csv"
        ordered.write_csv(pending)
        saved = pl.read_csv(pending, schema_overrides={"Id": pl.String})
        if saved["Id"].to_list() != sample["Id"].to_list() or not np.array_equal(
            saved["Label"].to_numpy(), ordered["Label"].to_numpy()
        ):
            raise ValueError("saved submission did not preserve ID order/probabilities")
        if output.exists() and sha256_file(output) != sha256_file(pending):
            raise ValueError("submission output already contains different data; choose a fresh output")
        pending.replace(output)
    return {
        "rows": saved.height,
        "schema": saved.columns,
        "sample_sha256": sha256_file(sample_path),
        "submission_sha256": sha256_file(output),
        "id_order_verified": True,
        "kaggle_submission": False,
    }


def export_predictions(
    paths: Sequence[Path],
    checkpoints: Sequence[Path],
    sample: Path,
    output: Path,
    data_root: str | None = None,
) -> dict:
    """Export verified test outputs; a single retained model is not relabeled as an ensemble."""
    from alaska2.constants import PURPOSE_LABELS
    from alaska2.evaluate import runtime_config
    from alaska2.inputs import prepare_inputs
    from alaska2.runtime import write_json

    if not paths or len(paths) != len(checkpoints):
        raise ValueError("supply one producing checkpoint for each prediction file")
    output, sample = Path(output), Path(sample)
    if output.suffix.lower() != ".csv":
        raise ValueError("export output must have a CSV extension")
    protected = {sample.resolve(), *(Path(path).resolve() for path in checkpoints)}
    for path in paths:
        path = Path(path)
        protected.update(
            [
                path.resolve(),
                path.with_suffix(".json").resolve(),
                path.with_name(path.stem + ".verify.json").resolve(),
            ]
        )
    receipt_path = output.with_suffix(".json")
    if output.resolve() in protected or receipt_path.resolve() in protected:
        raise ValueError("export output/receipt cannot overwrite a producing source artifact")
    cfg = runtime_config(load_checkpoint(checkpoints[0])["config"], {"data.root": data_root})
    expected = prepare_inputs(cfg).test
    sources = [
        validated_predictions(path, checkpoint, expected)
        for path, checkpoint in zip(paths, checkpoints, strict=True)
    ]
    if any(source.provenance["split"] != "test" for source in sources):
        raise ValueError("submission export requires canonical test predictions")
    if len(sources) == 1:
        frame = sources[0].frame
        composition = {"algorithm": "single_verified_model", "sources": [sources[0].provenance]}
    else:
        frame, composition = probability_mean(sources)
    purpose = sources[0].provenance["purpose"]
    complete = all(source.provenance["training_complete"] for source in sources)
    label = (
        "INCOMPLETE_REAL_DATA_BASELINE" if purpose == "baseline" and not complete else PURPOSE_LABELS[purpose]
    )
    from alaska2.delivery import artifact_claim

    with artifact_claim(output.with_name("." + output.name + ".lock")):
        receipt = {
            "project_id": "alaska2",
            "result_label": label,
            "training_complete": complete,
            "source_models": composition["sources"],
            "composition": composition["algorithm"],
            "submission": write_submission(frame, sample, output),
            "learned_stacking": "deferred; no leakage-safe full OOF matrix has been established",
        }
        if receipt_path.exists():
            if receipt_path.is_symlink() or json.loads(receipt_path.read_text()) != receipt:
                raise ValueError("export receipt already belongs to different source/output data")
        write_json(receipt_path, receipt)
        return receipt
