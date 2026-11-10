"""A fixed readout comparison and hard-case diagnostic preserve the full held-out population."""

import numpy as np
import polars as pl

from tools.analysis.error_analysis import analyze_errors
from tools.analysis.modeling_study import compare_readouts


def population():
    probabilities = np.array(
        [
            [0.45, 0.50, 0.03, 0.02],
            [0.35, 0.22, 0.22, 0.21],
            [0.30, 0.30, 0.30, 0.10],
            [0.25, 0.10, 0.10, 0.55],
            [0.70, 0.20, 0.05, 0.05],
            [0.10, 0.70, 0.10, 0.10],
            [0.20, 0.10, 0.60, 0.10],
            [0.30, 0.10, 0.10, 0.50],
        ]
    )
    return pl.DataFrame(
        {
            "family_id": ["a"] * 4 + ["b"] * 4,
            "kind": ["Cover", "JMiPOD", "JUNIWARD", "UERD"] * 2,
            "label": [0, 1, 2, 3] * 2,
            "quality": [75] * 4 + [95] * 4,
            **{
                f"p_{name}": probabilities[:, i]
                for i, name in enumerate(("Cover", "JMiPOD", "JUNIWARD", "UERD"))
            },
            "score": 1 - probabilities[:, 0],
        }
    )


def test_controlled_readout_changes_only_the_scoring_rule():
    frame = population()
    before = frame.clone()
    result = compare_readouts(frame, n_boot=8, seed=17)
    assert frame.equals(before)
    assert result["rows"] == 8 and result["families"] == 2
    assert result["treatment_active"]["different_scores"] > 0
    assert result["treatment_active"]["ranking_changed"] is True
    assert (
        result["conditions"]["union_probability"]["weighted_auc"]
        > result["conditions"]["max_stego_probability"]["weighted_auc"]
    )
    assert result["default_changed"] is False
    assert (
        result["paired_family_bootstrap"]
        == compare_readouts(frame, n_boot=8, seed=17)["paired_family_bootstrap"]
    )


def test_hard_case_diagnostic_reconciles_all_rows_and_keeps_private_cases_out():
    frame = population()
    result = analyze_errors(frame, fraction=0.25)
    assert result["rows"] == 8 and result["hard_rows"] == 3
    assert result["selection"]["cover_rows"] == 1 and result["selection"]["stego_rows"] == 2
    assert sum(row["hard_rows"] for row in result["by_kind"]) == 3
    assert sum(row["rows"] for row in result["by_quality"]) == 8
    assert result["diagnostic_is_weighted_auc_decomposition"] is False
    assert "a" not in result and "image" not in result


def test_verified_holdout_uses_canonical_validation_records(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import tools.analysis.modeling_study as study

    frame = population().with_columns(pl.lit("valid").alias("role"))
    monkeypatch.setattr(study, "load_checkpoint", lambda path: {"config": {"data": {"root": "old"}}})
    monkeypatch.setattr(study, "prepare_inputs", lambda cfg: SimpleNamespace(records=frame))

    def validated(path, checkpoint, expected):
        assert expected.equals(frame)
        return SimpleNamespace(frame=frame, provenance={"held_out": True})

    monkeypatch.setattr(study, "validated_predictions", validated)
    actual, _ = study.verified_holdout(tmp_path / "p", tmp_path / "c", tmp_path / "data")
    assert actual.equals(frame)
