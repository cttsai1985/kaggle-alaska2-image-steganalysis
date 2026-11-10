"""Compare two predeclared class-probability readouts on the same saved family holdout (CPU only)."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import polars as pl

from alaska2.composition import validated_predictions
from alaska2.constants import CLASSES
from alaska2.identity import git_source, sha256_file
from alaska2.inputs import prepare_inputs
from alaska2.metrics import binary_target, weighted_auc
from alaska2.model import load_checkpoint


def verified_holdout(predictions: Path, checkpoint: Path, data_root: Path) -> tuple[pl.DataFrame, dict]:
    """Reuse the canonical population/checkpoint/reconstruction contract, never a sidecar alone."""
    raw = load_checkpoint(checkpoint)
    cfg = copy.deepcopy(raw["config"])
    cfg["data"]["root"] = str(data_root)
    expected = prepare_inputs(cfg).records.filter(pl.col("role") == "valid")
    source = validated_predictions(predictions, checkpoint, expected)
    if not source.provenance["held_out"]:
        raise ValueError("analysis requires the producing model's complete held-out population")
    return source.frame, source.provenance


def _score(labels: np.ndarray, scores: np.ndarray) -> dict:
    return {"weighted_auc": float(weighted_auc(binary_target(labels), scores)), "rows": len(labels)}


def _paired_bootstrap(
    frame: pl.DataFrame, scores: tuple[np.ndarray, np.ndarray], n_boot: int, seed: int
) -> dict:
    _, inverse = np.unique(frame["family_id"].to_numpy(), return_inverse=True)
    groups = [np.flatnonzero(inverse == index) for index in range(int(inverse.max()) + 1)]
    labels = binary_target(frame["label"].to_numpy())
    rng, deltas = np.random.default_rng(seed), []
    for _ in range(n_boot):
        rows = np.concatenate([groups[index] for index in rng.integers(0, len(groups), len(groups))])
        deltas.append(
            weighted_auc(labels[rows], scores[1][rows]) - weighted_auc(labels[rows], scores[0][rows])
        )
    return {
        "draws": n_boot,
        "resampling_seed": seed,
        "unit": "source family; both readouts share each draw",
        "delta": "max_stego_probability minus union_probability",
        "p2_5": float(np.percentile(deltas, 2.5)),
        "p97_5": float(np.percentile(deltas, 97.5)),
        "scope": "fixed-model/fixed-holdout resampling; not training-seed or split-plan sensitivity",
    }


def compare_readouts(frame: pl.DataFrame, n_boot: int = 64, seed: int = 20261109) -> dict:
    """Change the probability readout only; never fit, select weights, remove rows or change defaults."""
    labels = frame["label"].to_numpy()
    baseline = frame["score"].to_numpy()
    alternative = frame.select([f"p_{kind}" for kind in CLASSES[1:]]).to_numpy().max(axis=1)
    ranking_changed = not np.array_equal(
        np.argsort(baseline, kind="stable"), np.argsort(alternative, kind="stable")
    )
    names = ("union_probability", "max_stego_probability")
    conditions = {
        name: _score(labels, score) for name, score in zip(names, (baseline, alternative), strict=True)
    }
    cohorts = []
    for column in ("quality", "kind"):
        values = frame[column].to_numpy()
        for value in sorted(set(values)):
            mask = values == value if column == "quality" else np.isin(values, ["Cover", value])
            if column == "kind" and value == "Cover":
                continue
            pair = {
                name: _score(labels[mask], score[mask])
                for name, score in zip(names, (baseline, alternative), strict=True)
            }
            cohorts.append(
                {
                    "cohort": column,
                    "value": value.item() if hasattr(value, "item") else value,
                    "rows": int(mask.sum()),
                    "conditions": pair,
                    "delta": pair[names[1]]["weighted_auc"] - pair[names[0]]["weighted_auc"],
                }
            )
    return {
        "rows": frame.height,
        "families": frame["family_id"].n_unique(),
        "conditions": conditions,
        "delta": conditions[names[1]]["weighted_auc"] - conditions[names[0]]["weighted_auc"],
        "treatment_active": {
            "different_scores": int(np.count_nonzero(baseline != alternative)),
            "ranking_changed": bool(ranking_changed),
        },
        "cohorts": cohorts,
        "paired_family_bootstrap": _paired_bootstrap(frame, (baseline, alternative), n_boot, seed),
        "default_changed": False,
        "fit_performed": False,
        "controls": "same checkpoint, rows, labels, source families, JPEG decoder and held-out probabilities",
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
        "kind": "fixed_probability_readout_comparison",
        "measurement_status": "MEASURED",
        **compare_readouts(frame),
        "producer": producer,
        "analysis_source": git_source(),
        "analyzer_sha256": sha256_file(__file__),
    }
    if sha256_file(args.predictions) != producer["predictions_sha256"]:
        raise ValueError("prediction bytes changed during analysis")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("rows", "conditions", "delta", "treatment_active")}))


if __name__ == "__main__":
    main()
