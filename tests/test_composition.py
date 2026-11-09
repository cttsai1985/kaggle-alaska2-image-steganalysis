"""Prediction provenance, fixed arithmetic and submission identity, using synthetic model receipts."""

import json

import numpy as np
import polars as pl
import pytest
import torch

from alaska2.composition import probability_mean, validated_predictions, write_submission
from alaska2.constants import CLASSES
from alaska2.identity import sha256_file
from alaska2.metrics import evaluation_report
from alaska2.model import state_dict_sha256


def source(tmp_path, name, offset=0.0, holdout=1):
    folder = tmp_path / name
    folder.mkdir()
    expected = pl.DataFrame(
        {
            "image": ["family.jpg"] * 4,
            "family_id": ["family"] * 4,
            "kind": list(CLASSES),
            "label": list(range(4)),
            "fold": [1] * 4,
            "quality": [90] * 4,
        }
    )
    weights = {"weight": torch.tensor([offset])}
    model_sha = state_dict_sha256(weights)
    identities = {
        "input_manifest_sha256": "input",
        "split_folds_sha256": "folds",
        "population_sha256": "population",
    }
    checkpoint = folder / "model.pt"
    torch.save(
        {
            "config": {
                "purpose": "fixture",
                "split": {"holdout_fold": holdout},
                "data": {"classes": list(CLASSES)},
                "train": {"epochs": 2},
            },
            "classes": list(CLASSES),
            "model": weights,
            "model_sha256": model_sha,
            "identities": identities,
            "epoch": 1,
        },
        checkpoint,
    )
    ckpt_sha = sha256_file(checkpoint)
    probs = np.full((4, 4), 0.1)
    probs[:, 0] = 0.7 - offset
    probs[:, 1] += offset
    frame = expected.with_columns(
        **{f"p_{kind}": pl.Series(probs[:, i]) for i, kind in enumerate(CLASSES)},
        score=pl.Series(1 - probs[:, 0]),
        model_sha256=pl.lit(model_sha),
        checkpoint_sha256=pl.lit(ckpt_sha),
    )
    prediction = folder / "valid_epoch001.parquet"
    frame.write_parquet(prediction)
    meta = {
        "classes": list(CLASSES),
        "split": "valid",
        "purpose": "fixture",
        "model_sha256": model_sha,
        "checkpoint_sha256": ckpt_sha,
        "predictions_sha256": sha256_file(prediction),
        "identities": identities,
    }
    prediction.with_suffix(".json").write_text(json.dumps(meta))
    verify = {
        "passed": True,
        **{key: meta[key] for key in ["model_sha256", "checkpoint_sha256", "predictions_sha256"]},
        "checks": {
            key: True
            for key in [
                "model_sha256_matches",
                "checkpoint_sha256_matches",
                "predictions_sha256_matches",
                "saved_rows_match_canonical_inputs",
                "score_matches_probabilities",
                "prediction_file_unchanged",
                "checkpoint_file_unchanged",
                "metrics_match_saved_predictions",
            ]
        },
        "metrics": evaluation_report(frame),
    }
    prediction.with_name(prediction.stem + ".verify.json").write_text(json.dumps(verify))
    return prediction, checkpoint, expected


def rewrite_prediction(path, transform):
    frame = transform(pl.read_parquet(path))
    frame.write_parquet(path)
    for sidecar in [path.with_suffix(".json"), path.with_name(path.stem + ".verify.json")]:
        data = json.loads(sidecar.read_text())
        data["predictions_sha256"] = sha256_file(path)
        sidecar.write_text(json.dumps(data))


def test_probability_mean_matches_independent_arithmetic(tmp_path):
    a, ca, expected = source(tmp_path, "a", 0.0)
    b, cb, _ = source(tmp_path, "b", 0.2)
    sources = [validated_predictions(a, ca, expected), validated_predictions(b, cb, expected)]
    result, receipt = probability_mean(sources)
    assert result["p_Cover"].to_list() == pytest.approx([0.6] * 4)
    assert result["score"].to_list() == pytest.approx([0.4] * 4)
    assert receipt["algorithm"] == "unweighted_probability_mean"
    assert receipt["held_out_rows"] == 4


@pytest.mark.parametrize(
    "alter",
    [
        lambda frame: pl.concat([frame, frame.head(1)]),
        lambda frame: frame.reverse(),
        lambda frame: frame.with_columns(pl.lit(float("nan")).alias("p_Cover")),
        lambda frame: frame.with_columns(pl.lit(2.0).alias("p_Cover")),
        lambda frame: frame.with_columns(pl.lit(0.0).alias("score")),
    ],
)
def test_rejects_ambiguous_or_invalid_prediction_rows(tmp_path, alter):
    pred, checkpoint, expected = source(tmp_path, "a")
    rewrite_prediction(pred, alter)
    with pytest.raises(ValueError):
        validated_predictions(pred, checkpoint, expected)


def test_rejects_rows_not_held_out_by_the_producing_model(tmp_path):
    pred, checkpoint, expected = source(tmp_path, "a", holdout=2)
    with pytest.raises(ValueError, match="held.out"):
        validated_predictions(pred, checkpoint, expected)


def test_does_not_claim_distinct_models_from_duplicate_inputs(tmp_path):
    pred, checkpoint, expected = source(tmp_path, "a")
    verified = validated_predictions(pred, checkpoint, expected)
    with pytest.raises(ValueError, match="distinct"):
        probability_mean([verified, verified])


def test_submission_uses_supplied_sample_order(tmp_path):
    sample = tmp_path / "sample.csv"
    pl.DataFrame({"Id": ["b.jpg", "a.jpg"], "Label": [0, 0]}).write_csv(sample)
    pred = pl.DataFrame({"image": ["a.jpg", "b.jpg"], "score": [0.2, 0.8]})
    output = tmp_path / "submission.csv"
    receipt = write_submission(pred, sample, output)
    saved = pl.read_csv(output)
    assert saved["Id"].to_list() == ["b.jpg", "a.jpg"]
    assert saved["Label"].to_list() == [0.8, 0.2]
    assert receipt["rows"] == 2 and receipt["sample_sha256"] == sha256_file(sample)


def test_submission_rejects_missing_or_duplicate_ids(tmp_path):
    sample = tmp_path / "sample.csv"
    pl.DataFrame({"Id": ["a.jpg", "b.jpg"], "Label": [0, 0]}).write_csv(sample)
    for images in [["a.jpg"], ["a.jpg", "a.jpg"]]:
        with pytest.raises(ValueError):
            write_submission(
                pl.DataFrame({"image": images, "score": [0.2] * len(images)}), sample, tmp_path / "bad.csv"
            )


def test_mean_does_not_carry_a_single_models_logits(tmp_path):
    a, ca, expected = source(tmp_path, "a", 0.0)
    b, cb, _ = source(tmp_path, "b", 0.2)
    for path in [a, b]:
        rewrite_prediction(path, lambda frame: frame.with_columns(pl.lit(99.0).alias("logit_Cover")))
    result, _ = probability_mean(
        [validated_predictions(a, ca, expected), validated_predictions(b, cb, expected)]
    )
    assert not any(column.startswith("logit_") for column in result.columns)


def test_rejects_unverified_class_order_and_stale_artifact(tmp_path):
    pred, checkpoint, expected = source(tmp_path, "a")
    sidecar = pred.with_suffix(".json")
    meta = json.loads(sidecar.read_text())
    meta["classes"] = list(reversed(CLASSES))
    sidecar.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="class"):
        validated_predictions(pred, checkpoint, expected)
    meta["classes"] = list(CLASSES)
    meta["predictions_sha256"] = "stale"
    sidecar.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="identity"):
        validated_predictions(pred, checkpoint, expected)


def test_submission_cannot_overwrite_its_sample(tmp_path):
    sample = tmp_path / "sample.csv"
    pl.DataFrame({"Id": ["a.jpg"], "Label": [0]}).write_csv(sample)
    original = sample.read_bytes()
    with pytest.raises(ValueError, match="sample"):
        write_submission(pl.DataFrame({"image": ["a.jpg"], "score": [0.2]}), sample, sample)
    assert sample.read_bytes() == original
