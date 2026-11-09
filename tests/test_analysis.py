"""Holdout analysis on synthetic saved predictions (CPU, no model): FIXTURE_NOT_A_MODEL_RESULT."""

import json

import numpy as np
import polars as pl
import pytest

from alaska2.constants import CLASSES
from alaska2.metrics import softmax, weighted_auc
from tools.analysis.holdout_analysis import analyse, operating_points


def _fake_predictions(tmp_path, n_fam=300, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n_fam):
        q = (75, 90, 95)[f % 3]
        for lab, kind in enumerate(CLASSES):
            logits = rng.normal(size=4)
            logits[lab] += 1.5 if q == 75 else 0.5  # QF75 made easier on purpose
            rows.append(
                {
                    "image": f"{f:05d}.jpg",
                    "family_id": f,
                    "kind": kind,
                    "label": lab,
                    "quality": q,
                    **{f"logit_{c}": float(v) for c, v in zip(CLASSES, logits, strict=True)},
                }
            )
    df = pl.DataFrame(rows)
    p = softmax(df.select([f"logit_{c}" for c in CLASSES]).to_numpy())
    df = df.with_columns(
        **{f"p_{c}": pl.Series(p[:, i]) for i, c in enumerate(CLASSES)}, score=pl.Series(1 - p[:, 0])
    )
    path = tmp_path / "valid_epoch001.parquet"
    df.write_parquet(path)
    meta = {
        "split": "valid",
        "result_label": "FIXTURE_NOT_A_MODEL_RESULT",
        "predictions_sha256": "x",
        "model_sha256": "y",
        "epoch": 1,
    }
    path.with_suffix(".json").write_text(json.dumps(meta))
    return path, df


def test_analysis_counts_metrics_and_redaction(tmp_path):
    pytest.importorskip("matplotlib", reason="plots need `uv run --group analysis`")
    path, df = _fake_predictions(tmp_path)
    out = analyse(path, tmp_path / "ev", n_boot=20, seed=1, make_plots=True)
    y = df["label"].to_numpy() != 0
    assert out["pooled"]["weighted_auc"] == pytest.approx(weighted_auc(y, df["score"].to_numpy()))
    assert out["pooled"]["weighted_auc"] == pytest.approx(out["pooled"]["weighted_auc_legacy_impl"], abs=1e-9)
    assert out["pooled"]["auc"] == pytest.approx(out["pooled"]["auc_sklearn"], abs=1e-12)
    assert sum(out["pooled"]["band_contribution"]) == pytest.approx(out["pooled"]["weighted_auc"])
    assert out["coverage"]["by_kind"] == {k: 300 for k in CLASSES}
    assert out["by_quality"]["75"]["weighted_auc"] > out["by_quality"]["95"]["weighted_auc"]
    assert out["by_quality"]["75"]["rows"] == 400 and out["by_quality"]["75"]["negatives"] == 100
    b = out["bootstrap"]["pooled"]["weighted_auc"]
    assert b["p2.5"] <= out["pooled"]["weighted_auc"] <= b["p97.5"]
    assert (tmp_path / "ev" / "holdout_roc.png").stat().st_size > 1000
    text = json.dumps(out)
    assert "00001.jpg" not in text  # no image ids leak into the aggregate


def test_operating_points_on_a_known_roc():
    y = np.array([1, 1, 0, 1, 0, 0])
    s = np.array([6, 5, 4, 3, 2, 1], dtype=float)
    pts = operating_points(y, s)
    # ROC vertices: (0,0) (0,1/3) (0,2/3) (1/3,2/3) (1/3,1) (2/3,1) (1,1)
    assert pts["fpr_at_tpr"]["0.4"] == pytest.approx(0.0)
    assert pts["fpr_at_tpr"]["0.8"] == pytest.approx(1 / 3)
    assert pts["tpr_at_fpr"]["0.1"] == pytest.approx(2 / 3)


def test_chance_reference_equals_the_metric_on_tied_scores():
    from tools.analysis.holdout_analysis import CHANCE_WEIGHTED_AUC

    y = np.array([0, 1] * 50)
    assert weighted_auc(y, np.zeros(100)) == pytest.approx(CHANCE_WEIGHTED_AUC, abs=1e-12)
