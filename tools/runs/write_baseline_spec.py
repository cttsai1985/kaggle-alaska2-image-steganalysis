"""Write the predeclared baseline spec (evidence/phase2/baseline_spec.json) BEFORE the baseline launches.

The epoch count comes from the CUDA smoke receipt; everything else is fixed here, so the spec is committed
before any baseline label is seen.

    uv run python -m tools.runs.write_baseline_spec --smoke evidence/phase2/cuda_smoke.json \
        --out evidence/phase2/baseline_spec.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alaska2.config import config_hash, load_config

CONFIG = "configs/baseline_effb0_fold1.toml"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    smoke = json.loads(a.smoke.read_text())
    b = smoke["baseline_budget"]
    if b["E"] < 2:
        raise SystemExit(f"E = {b['E']} < 2: full-data baseline is NO-GO; do not write a spec for it")
    sets = [f"train.epochs={b['E']}"]
    cfg = load_config(CONFIG, sets)
    spec = {
        "kind": "baseline_spec",
        "declared_before_training": True,
        "result_label": "REAL_DATA_BASELINE",
        "config": CONFIG,
        "overrides": sets,
        "config_sha256_with_overrides": config_hash(cfg),
        "population": {
            "name": "full_data_single_holdout",
            "statement": "ONE single holdout (legacy fold 1, family-level). This is NOT cross-validation, "
            "and one training seed: no seed-stability or variance claim follows from it.",
            "fit": "legacy folds 2-5: 240,000 rows / 60,000 families (Cover + 3 stego kinds per family)",
            "valid": "legacy fold 1: 60,000 rows / 15,000 families; 15,000 rows per kind; 20,000 per quality",
            "family_separation": "0 families and 0 identical files cross fit/valid (asserted before fit)",
            "test_set": "not used for training, selection or evaluation; no Kaggle submission",
        },
        "recipe": {
            "backbone": f"{cfg['model']['backbone']} ({cfg['model']['weights_repo']}"
            f"@{cfg['model']['weights_revision'][:12]})",
            "input": "512x512 native RGB, no resize; train D4 flips/rot90 only; valid untransformed",
            "epochs": b["E"],
            "batch_size": cfg["train"]["batch_size"],
            "optimizer": f"AdamW lr {cfg['train']['lr']} wd {cfg['train']['weight_decay']}",
            "schedule": f"{cfg['train']['warmup_steps']} warmup steps then cosine to 0 over all E epochs",
            "loss": f"4-class CE, label smoothing {cfg['train']['label_smoothing']}",
            "precision": f"{cfg['train']['amp_dtype']} autocast, channels_last",
            "seed": cfg["train"]["seed"],
            "num_workers": cfg["train"]["num_workers"],
            "score": "1 - p_Cover",
        },
        "checkpoint_selection_policy": {
            "policy": "last completed epoch (epoch E) is the model",
            "holdout_labels_used_for_selection": False,
            "declared_at": "before training (this file is committed before launch)",
            "if_interrupted": "the run is reported as incomplete; the last completed epoch is evaluated and "
            "labeled as such, never presented as the declared E-epoch baseline",
        },
        "budget": {
            "wall_clock_cap_s": 10_800,
            "train_eval_timeout_s": b["train_eval_timeout_s"],
            "enforcement": f"timeout --kill-after=60 {b['train_eval_timeout_s']} around train+evaluate "
            "(cap minus the measured eval + verify time); SIGTERM closes the manifest as interrupted; "
            "verify then runs in a fresh process, all under nice -n 19 ionice -c3",
            "derivation": b,
            "runs": "one run, fresh run_id, nohup + log in the run dir; no duplicate launches",
            "second_model": "only if an additional 2 h is available after the baseline wall time is measured",
        },
        "evaluation_question": "Does the fixed recipe give a holdout competition score traceable to its "
        "population, and is that pooled number broad across JPEG-quality cohorts and stego kinds or "
        "concentrated in some of them / in one ROC region?",
        "competing_explanations_and_controls": [
            {
                "explanation": "family leakage inflates the score",
                "control": "frozen legacy family folds; 0 crossing families asserted",
            },
            {
                "explanation": "the score is driven by easy cohorts (e.g. one JPEG quality)",
                "control": "per-quality and kind x quality cohorts with counts",
            },
            {
                "explanation": "weighted AUC and standard AUC disagree (low-TPR band weighted 2x)",
                "control": "band contributions [0,0.4] vs [0.4,1] and both metrics on the same predictions",
            },
            {
                "explanation": "epoch picked on holdout labels",
                "control": "fixed last-epoch policy declared here",
            },
            {
                "explanation": "prediction / label / ID mismatch",
                "control": "verify: exact holdout membership, canonical labels, fresh-process reload",
            },
            {
                "explanation": "difference between cohorts is holdout sampling noise",
                "control": "family-level bootstrap intervals (holdout sampling only; not seed variance)",
            },
        ],
        "reported_metrics": [
            "competition weighted AUC (exact) and legacy implementation",
            "standard AUC",
            "ROC band contributions and shortfalls, operating points",
            "per-quality, per stego kind, kind x quality cohorts with row/family/pos/neg counts",
            "4-class confusion matrix",
            "reload delta: max |p_rebuilt - p_saved| (tol 1e-3), |delta weighted AUC| (tol 1e-4)",
        ],
        "not_claimed": [
            "no leaderboard score, rank or percentile",
            "no cross-validation, no seed stability",
            "no acquisition-domain generalization beyond this dataset",
        ],
        "smoke_receipt_run_id": smoke["run_id"],
        "launch_conditions": {
            "operator_override": smoke.get("operator_override"),
            "host_contention_at_smoke_launch": smoke.get("host_contention_at_launch"),
            "baseline_launch_contention": "recorded at launch in the run dir (host_contention.json) and in "
            "docs/phase2-baseline.md",
            "throughput_limit": smoke.get("loader_bound"),
        },
    }
    a.out.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"E": b["E"], "config_sha256": spec["config_sha256_with_overrides"]}, indent=2))


if __name__ == "__main__":
    main()
