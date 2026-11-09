"""A retained-checkpoint report must distinguish verified evaluation from completed training."""

import hashlib
import json

import pytest

from tools.runs.retained_receipt import retained_receipt


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def retained(tmp_path):
    run, pred = tmp_path / "run", tmp_path / "predictions"
    ckpt = run / "checkpoints/epoch_004.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"retained immutable checkpoint")
    digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    ids = {"input_manifest_sha256": "input", "split_folds_sha256": "fold", "population_sha256": "population"}
    put(
        run / "run_manifest.json",
        {
            "run_id": "test-incomplete",
            "purpose": "baseline",
            "status": "interrupted",
            "plan": {"epochs": 5},
            "final_checkpoint": "checkpoints/epoch_004.pt",
            "identities": ids,
            "epochs": [{"epoch": 4, "checkpoint_sha256": digest, "model_sha256": "weights"}],
            "population_audit": {"valid": {"rows": 8, "families": 2}},
            "source": {"git_sha": "source", "dirty": True},
            "config": {"purpose": "baseline", "verify": {"prob_atol": 0.001, "metric_atol": 0.0001}},
        },
    )
    pred.mkdir()
    (pred / "valid_epoch004.parquet").write_bytes(b"private prediction fixture")
    put(
        pred / "valid_epoch004.json",
        {
            "epoch": 4,
            "purpose": "baseline",
            "split": "valid",
            "checkpoint_sha256": digest,
            "model_sha256": "weights",
            "identities": ids,
            "predictions_sha256": hashlib.sha256((pred / "valid_epoch004.parquet").read_bytes()).hexdigest(),
            "membership": {"rows": 8, "expected_rows": 8, "missing": 0, "extra": 0, "duplicates": 0},
            "metrics": {"pooled": {"weighted_auc": 0.6}},
            "inference_wall_s": 1.0,
            "device": "cuda:0",
            "versions": {"torch": "fixture"},
            "runtime_overrides": {},
        },
    )
    put(
        pred / "valid_epoch004.verify.json",
        {
            "metrics": {"pooled": {"weighted_auc": 0.6}},
            "passed": True,
            "rows": 8,
            "predictions_sha256": hashlib.sha256((pred / "valid_epoch004.parquet").read_bytes()).hexdigest(),
            "checkpoint_sha256": digest,
            "model_sha256": "weights",
            "checks": {
                "checkpoint_sha256_matches": True,
                "predictions_sha256_matches": True,
                "model_sha256_matches": True,
                "saved_rows_match_canonical_inputs": True,
                "score_matches_probabilities": True,
                "prediction_file_unchanged": True,
                "checkpoint_file_unchanged": True,
                "metrics_match_saved_predictions": True,
            },
            "observed": {"max_abs_prob_delta": 0.0, "weighted_auc_delta": 0.0},
        },
    )
    return run, pred


def test_reports_verified_evaluation_without_promoting_incomplete_training(retained):
    report = retained_receipt(*retained)
    assert report["training_complete"] is False
    assert report["evaluation_verified"] is True
    assert report["result_label"] == "INCOMPLETE_REAL_DATA_BASELINE"
    assert report["evaluated_epoch"] == 4
    assert "/tmp/" not in json.dumps(report)


@pytest.mark.parametrize(
    "target,field,value",
    [
        ("valid_epoch004.json", "model_sha256", "wrong-model"),
        ("valid_epoch004.verify.json", "passed", False),
        ("valid_epoch004.json", "epoch", 5),
    ],
)
def test_rejects_unverified_or_wrong_checkpoint_evidence(retained, target, field, value):
    run, pred = retained
    path = pred / target
    data = json.loads(path.read_text())
    data[field] = value
    put(path, data)
    with pytest.raises(ValueError):
        retained_receipt(run, pred)


def test_rejects_changed_prediction_bytes(retained):
    run, pred = retained
    (pred / "valid_epoch004.parquet").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="prediction"):
        retained_receipt(run, pred)


def test_rejects_delta_outside_checkpoint_tolerance(retained):
    run, pred = retained
    path = pred / "valid_epoch004.verify.json"
    data = json.loads(path.read_text())
    data["observed"]["max_abs_prob_delta"] = 0.1
    put(path, data)
    with pytest.raises(ValueError, match="tolerance"):
        retained_receipt(run, pred)


def test_rejects_verification_from_different_prediction_bytes(retained):
    run, pred = retained
    path = pred / "valid_epoch004.verify.json"
    data = json.loads(path.read_text())
    data["predictions_sha256"] = "previous-file"
    put(path, data)
    with pytest.raises(ValueError, match="verification.*identity"):
        retained_receipt(run, pred)


def test_rejects_analysis_of_different_prediction_artifact(retained, tmp_path):
    run, pred = retained
    analysis = tmp_path / "analysis.json"
    put(analysis, {"predictions_sha256": "wrong", "model_sha256": "weights"})
    with pytest.raises(ValueError, match="analysis.*identity"):
        retained_receipt(run, pred, analysis)


def test_rejects_changed_sidecar_metrics(retained):
    run, pred = retained
    path = pred / "valid_epoch004.json"
    data = json.loads(path.read_text())
    data["metrics"]["pooled"]["weighted_auc"] = 99.0
    put(path, data)
    with pytest.raises(ValueError, match="metric"):
        retained_receipt(run, pred)


def test_never_promotes_fixture_to_real_model_result(retained):
    run, pred = retained
    path = run / "run_manifest.json"
    data = json.loads(path.read_text())
    data["purpose"] = "fixture"
    put(path, data)
    with pytest.raises(ValueError, match="baseline"):
        retained_receipt(run, pred)


def test_rejects_analysis_metrics_inconsistent_with_verified_metrics(retained, tmp_path):
    run, pred = retained
    meta = json.loads((pred / "valid_epoch004.json").read_text())
    analysis = tmp_path / "analysis.json"
    put(
        analysis,
        {
            "predictions_sha256": meta["predictions_sha256"],
            "model_sha256": meta["model_sha256"],
            "pooled": {"weighted_auc": 99.0},
        },
    )
    with pytest.raises(ValueError, match="analysis.*metric"):
        retained_receipt(run, pred, analysis)
