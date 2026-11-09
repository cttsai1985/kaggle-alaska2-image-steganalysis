"""Validate and summarize evaluation of a predeclared incomplete-run checkpoint.

python -m tools.runs.retained_receipt --run-dir PRIVATE_RUN --predictions-dir PRIVATE_OUTPUT --out RECEIPT
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alaska2.identity import sha256_file


def retained_receipt(run_dir: Path, predictions_dir: Path, analysis_path: Path | None = None) -> dict:
    """Bind independently rebuilt predictions to the last completed checkpoint; keep incompleteness."""
    run = json.loads((run_dir / "run_manifest.json").read_text())
    if run["status"] != "interrupted":
        raise ValueError("this fallback requires an interrupted training manifest")
    last = run["epochs"][-1]
    epoch = int(last["epoch"])
    if epoch >= int(run["plan"]["epochs"]):
        raise ValueError("this fallback requires an incomplete epoch budget")
    checkpoint = run_dir / run["final_checkpoint"]
    if sha256_file(checkpoint) != last["checkpoint_sha256"]:
        raise ValueError("checkpoint bytes differ from the training receipt")
    stem = f"valid_epoch{epoch:03d}"
    pred = predictions_dir / f"{stem}.parquet"
    meta = json.loads(pred.with_suffix(".json").read_text())
    ver = json.loads((predictions_dir / f"{stem}.verify.json").read_text())
    if meta["epoch"] != epoch or any(meta[k] != last[k] for k in ("checkpoint_sha256", "model_sha256")):
        raise ValueError("prediction checkpoint/model identity differs from the retained epoch")
    if sha256_file(pred) != meta["predictions_sha256"]:
        raise ValueError("prediction bytes differ from their sidecar")
    for key in ("input_manifest_sha256", "split_folds_sha256", "population_sha256"):
        if meta["identities"][key] != run["identities"][key]:
            raise ValueError(f"prediction identity differs on {key}")
    if any(ver.get(key) != meta[key] for key in ("predictions_sha256", "checkpoint_sha256", "model_sha256")):
        raise ValueError("verification artifact identity differs from the saved predictions")
    rows = run["population_audit"]["valid"]["rows"]
    membership = meta["membership"]
    if membership != {"rows": rows, "expected_rows": rows, "missing": 0, "extra": 0, "duplicates": 0}:
        raise ValueError("prediction membership differs from the declared holdout")
    required_checks = (
        "checkpoint_sha256_matches",
        "predictions_sha256_matches",
        "model_sha256_matches",
        "saved_rows_match_canonical_inputs",
    )
    if (
        not ver["passed"]
        or ver["rows"] != rows
        or not all(ver["checks"].get(key) is True for key in required_checks)
    ):
        raise ValueError("independent verification did not pass all required checks")
    tolerance = run["config"]["verify"]
    observed = ver["observed"]
    if not (
        0 <= observed["max_abs_prob_delta"] <= tolerance["prob_atol"]
        and 0 <= observed["weighted_auc_delta"] <= tolerance["metric_atol"]
    ):
        raise ValueError("observed reconstruction difference exceeds checkpoint tolerance")
    report = {
        "kind": "retained_checkpoint_evaluation",
        "measurement_status": "MEASURED",
        "result_label": "INCOMPLETE_REAL_DATA_BASELINE",
        "run_id": run["run_id"],
        "training_status": run["status"],
        "training_complete": False,
        "planned_epochs": run["plan"]["epochs"],
        "evaluated_epoch": epoch,
        "checkpoint_sha256": last["checkpoint_sha256"],
        "model_sha256": last["model_sha256"],
        "training_source": run["source"],
        "identities": run["identities"],
        "population": run["population_audit"]["valid"],
        "membership": membership,
        "metrics": meta["metrics"],
        "predictions_sha256": meta["predictions_sha256"],
        "device": meta["device"],
        "versions": meta["versions"],
        "inference_wall_s": meta["inference_wall_s"],
        "runtime_overrides": {k: v for k, v in meta["runtime_overrides"].items() if k != "data.root"},
        "evaluation_verified": True,
        "verification": {"observed": observed, "tolerance": tolerance, "checks": ver["checks"]},
        "limits": [
            "single family holdout; not cross-validation or a leaderboard score",
            "last completed epoch fallback; intended training budget remains incomplete",
            "no new training, epoch selection on holdout labels, or CUDA replay equivalence claim",
        ],
    }

    if analysis_path is not None:
        analysis = json.loads(analysis_path.read_text())
        if any(analysis.get(key) != meta[key] for key in ("predictions_sha256", "model_sha256")):
            raise ValueError("analysis artifact identity differs from verified predictions")
        analysis["result_label"] = report["result_label"]
        report["analysis"] = analysis
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = retained_receipt(args.run_dir, args.predictions_dir, args.analysis)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ("result_label", "evaluated_epoch", "evaluation_verified")}))


if __name__ == "__main__":
    main()
