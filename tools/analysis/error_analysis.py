"""Diagnose a fixed high-loss cohort without changing any held-out row, model or inference policy."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import polars as pl

from alaska2.identity import git_source, sha256_file
from alaska2.metrics import binary_target, weighted_auc
from tools.analysis.modeling_study import verified_holdout


def analyze_errors(frame: pl.DataFrame, fraction: float = 0.05) -> dict:
    """Binary log-loss mining diagnoses confidence errors; it does not decompose weighted AUC."""
    if not 0 < fraction < 1:
        raise ValueError("hard-cohort fraction must be between zero and one")
    target = binary_target(frame["label"].to_numpy())
    scores = frame["score"].to_numpy()
    true_probability = np.where(target == 1, scores, 1 - scores)
    loss = -np.log(np.clip(true_probability, 1e-15, 1))
    hard, counts = np.zeros(frame.height, dtype=bool), {}
    for label, name in ((0, "cover_rows"), (1, "stego_rows")):
        rows = np.flatnonzero(target == label)
        count = math.ceil(fraction * len(rows))
        hard[rows[np.argsort(-loss[rows], kind="stable")[:count]]] = True
        counts[name] = count
    cohorts = {}
    for column in ("kind", "quality"):
        values = frame[column].to_numpy()
        cohorts[column] = []
        for value in sorted(set(values)):
            mask = values == value
            selected, total = int((hard & mask).sum()), int(mask.sum())
            cohorts[column].append(
                {
                    column: value.item() if hasattr(value, "item") else value,
                    "rows": total,
                    "hard_rows": selected,
                    "hard_rate": selected / total,
                    "enrichment_over_full_population_rate": (selected / total) / float(hard.mean()),
                    "mean_true_class_probability": float(true_probability[mask].mean()),
                }
            )
    return {
        "rows": frame.height,
        "families": frame["family_id"].n_unique(),
        "weighted_auc_reconciled": float(weighted_auc(target, scores)),
        "hard_rows": int(hard.sum()),
        "selection": {
            "fraction_within_each_binary_class": fraction,
            **counts,
            "rule": "largest binary log loss; canonical retained-row order breaks ties",
        },
        "diagnostic_is_weighted_auc_decomposition": False,
        "fixed_half_probability_threshold": {
            "false_flagged_cover": int(((target == 0) & (scores >= 0.5)).sum()),
            "missed_stego": int(((target == 1) & (scores < 0.5)).sum()),
            "threshold_tuned": False,
        },
        "by_kind": cohorts["kind"],
        "by_quality": cohorts["quality"],
        "hard_mean_true_class_probability": float(true_probability[hard].mean()),
        "other_mean_true_class_probability": float(true_probability[~hard].mean()),
        "all_evaluation_rows_retained": True,
        "refit_or_promotion_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frame, producer = verified_holdout(args.predictions, args.checkpoint, args.data_root)
    result = {
        "kind": "heldout_hard_case_diagnostic",
        "measurement_status": "MEASURED",
        **analyze_errors(frame),
        "producer": producer,
        "analysis_source": git_source(),
        "analyzer_sha256": sha256_file(__file__),
    }
    if sha256_file(args.predictions) != producer["predictions_sha256"]:
        raise ValueError("prediction bytes changed during analysis")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("rows", "hard_rows", "by_kind", "by_quality")}))


if __name__ == "__main__":
    main()
