"""Audit held-out provenance, export a real model's test CSV, and exercise recoverable offline delivery.

Private images, identifiers and model files stay outside Git. Public output contains aggregate receipts only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import polars as pl

from alaska2.composition import export_predictions, validated_predictions
from alaska2.delivery import DeliveryError, RecoveryRequired, create_bundle, deliver, verify_bundle
from alaska2.evaluate import runtime_config
from alaska2.identity import git_source, sha256_file
from alaska2.inputs import prepare_inputs
from alaska2.model import load_checkpoint
from alaska2.runtime import write_json


def transfer_drill(bundle: Path, root: Path, fault: str) -> dict:
    """A retry verifies prior destination bytes; it never relaunches model training or inference."""
    manifest = verify_bundle(bundle)
    destination, state = root / "destination", root / "state"
    receipt = root / "receipt.json"
    if receipt.exists():
        prior = json.loads(receipt.read_text())
        result = deliver(bundle, destination, state)
        if result["manifest_sha256"] != prior["recovered"]["manifest_sha256"]:
            raise ValueError("drill identity changed on repeat")
        return prior
    try:
        deliver(bundle, destination, state, fault=fault)
    except DeliveryError as error:
        stage = destination / manifest["experiment_id"] / "attempt-1"
        first = next(iter(manifest["files"]))
        if fault == "corrupt_copy":
            if sha256_file(stage / first) == manifest["files"][first]["sha256"]:
                raise AssertionError("corruption treatment was not active") from error
        elif fault == "after_copy" and (not (stage / first).is_file() or (stage / "manifest.json").exists()):
            raise AssertionError("partial-copy treatment was not active") from error
        elif fault == "after_publish" and not (stage.parent / "current.json").exists():
            raise AssertionError("post-publish treatment was not active") from error
    else:
        raise AssertionError("transfer fault did not interrupt the attempt")
    if verify_bundle(bundle) != manifest:
        raise AssertionError("scientific artifacts changed after transfer failure")
    try:
        deliver(bundle, destination, state)
    except RecoveryRequired:
        pass
    else:
        raise AssertionError("incomplete transfer silently retried without explicit recovery")
    recovered = deliver(bundle, destination, state, recover=True)
    if recovered["experiment_id"] != manifest["experiment_id"] or recovered["attempt"] != 2:
        raise AssertionError("recovery changed logical identity or lost attempt history")
    result = {
        "fault": fault,
        "fault_active": True,
        "source_bundle_unchanged": verify_bundle(bundle) == manifest,
        "recovery_required": True,
        "recovered": recovered,
    }
    write_json(receipt, result)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--predictions-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--public-receipt", type=Path, required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    run = json.loads((args.run_dir / "run_manifest.json").read_text())
    checkpoint = args.run_dir / run["final_checkpoint"]
    raw = load_checkpoint(checkpoint)
    cfg = runtime_config(raw["config"], {"data.root": str(args.data_root)})
    inputs = prepare_inputs(cfg)
    epoch = int(raw["epoch"])
    valid = args.predictions_dir / f"valid_epoch{epoch:03d}.parquet"
    test = args.predictions_dir / f"test_epoch{epoch:03d}.parquet"
    heldout = inputs.records.filter(pl.col("role") == "valid")
    validation = validated_predictions(valid, checkpoint, heldout)
    test_source = validated_predictions(test, checkpoint, inputs.test)
    sample = args.data_root / "raw/extracted/sample_submission.csv"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv = args.out_dir / "submission.csv"
    export = export_predictions([test], [checkpoint], sample, csv, str(args.data_root))
    identity = {
        "run_id": run["run_id"],
        "model_sha256": raw["model_sha256"],
        "checkpoint_sha256": sha256_file(checkpoint),
        "identities": raw["identities"],
        "result_label": export["result_label"],
        "submission_sha256": export["submission"]["submission_sha256"],
        "training_source": run["source"],
    }
    files = {
        "model/checkpoint.pt": checkpoint,
        "metadata/run_manifest.json": args.run_dir / "run_manifest.json",
        "predictions/holdout.parquet": valid,
        "predictions/test.parquet": test,
        "predictions/holdout.json": valid.with_suffix(".json"),
        "predictions/test.json": test.with_suffix(".json"),
        "verification/holdout.json": valid.with_name(valid.stem + ".verify.json"),
        "verification/test.json": test.with_name(test.stem + ".verify.json"),
        "submission/submission.csv": csv,
        "submission/receipt.json": csv.with_suffix(".json"),
    }
    bundle = create_bundle(identity, files, args.out_dir / "offline-bundle")
    manifest = verify_bundle(bundle)
    drills = [
        transfer_drill(bundle, args.out_dir / "drills" / fault, fault)
        for fault in ["after_copy", "corrupt_copy", "after_publish"]
    ]
    quality = deliver(
        bundle,
        args.out_dir / "quality-gate-destination",
        args.out_dir / "quality-state",
        quality_accepted=False,
    )
    delivered = deliver(bundle, args.out_dir / "delivered", args.out_dir / "delivery-state")
    blob = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD:legacy/script/stacker.py"], text=True
    ).strip()
    baseline_blob = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "e00f0b8:script/stacker.py"], text=True
    ).strip()
    if blob != baseline_blob:
        raise ValueError("legacy source was changed")
    report = {
        "kind": "prediction_provenance_and_delivery",
        "measurement_status": "MEASURED",
        "result_label": export["result_label"],
        "client_source": git_source(root),
        "validation_source": validation.provenance,
        "test_source": test_source.provenance,
        "coverage": {
            "kind": "single_holdout",
            "producing_fold": cfg["split"]["holdout_fold"],
            "heldout_rows": heldout.height,
            "labeled_rows": inputs.records.height,
            "full_oof": False,
            "all_validation_rows_heldout_by_producer": True,
        },
        "submission": export["submission"],
        "composition": "single_verified_model",
        "ensemble_arithmetic": "mean arithmetic checked on synthetic fixtures; no real ensemble gain claimed",
        "legacy_audit": {
            "baseline_blob_sha": baseline_blob,
            "unchanged": True,
            "cached_predictions": "not supplied; OOF provenance unverified",
            "findings": [
                "generate_stacking_data_split concatenates cached frames without producing-fold/model proofs",
                "run_stacker can fit and score on the same validation rows",
                "cached filename architecture/score fragments do not establish OOF status",
            ],
        },
        "learned_stacking": "deferred: no leakage-safe full OOF matrix or separate meta-model assessment",
        "platform_contract_proposal": json.loads((root / "configs/platform-contract.json").read_text()),
        "platform_namespace_created": False,
        "offline_bundle": {
            "experiment_id": manifest["experiment_id"],
            "files": len(manifest["files"]),
            "verified": True,
        },
        "drill_label": "CONTROLLED_REAL_MODEL_FILESYSTEM_TRANSFER_DRILL",
        "transfer_drills": drills,
        "quality_gate_drill": quality,
        "delivery": delivered,
        "remote_service_verified": False,
        "limits": [
            "filesystem destination verified; no MLflow/S3 transfer claimed",
            "single labeled holdout, not full OOF or cross-validation",
            "CSV generated locally; no Kaggle submission",
        ],
    }
    text = json.dumps(report, sort_keys=True)
    if "/home/" in text or "/inputs/" in text or "/output/" in text:
        raise ValueError("private path survived aggregate redaction")
    write_json(args.public_receipt, report)
    print(
        json.dumps(
            {
                "experiment_id": manifest["experiment_id"],
                "submission_rows": export["submission"]["rows"],
                "delivery_state": delivered["state"],
                "transport": "filesystem",
            }
        )
    )


if __name__ == "__main__":
    main()
