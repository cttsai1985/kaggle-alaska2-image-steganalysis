"""Weighted-AUC contract: independent fixtures, the legacy implementation and the 4-class -> binary score."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from alaska2.constants import CLASSES, COVER_INDEX, TPR_THRESHOLDS, TPR_WEIGHTS
from alaska2.metrics import (
    binary_score,
    binary_target,
    plain_auc,
    roc_vertices,
    softmax,
    weighted_auc,
    weighted_auc_parts,
)

LEGACY = Path(__file__).resolve().parents[1] / "legacy" / "alaska_utils" / "eval_metrics.py"


def legacy_metric():
    spec = importlib.util.spec_from_file_location("legacy_eval_metrics", LEGACY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.alaska_weighted_auc


Y10 = np.array([0] * 5 + [1] * 5)


def test_published_parameters():
    assert TPR_THRESHOLDS == (0.0, 0.4, 1.0) and TPR_WEIGHTS == (2.0, 1.0)
    assert CLASSES == ("Cover", "JMiPOD", "JUNIWARD", "UERD") and COVER_INDEX == 0


def test_perfect_ranking_scores_one():
    parts = weighted_auc_parts(Y10, np.arange(10.0))
    assert parts["weighted_auc"] == pytest.approx(1.0, abs=1e-15)
    assert parts["band_area"] == pytest.approx([0.4, 0.6])
    assert sum(parts["band_contribution"]) == pytest.approx(1.0)


def test_reversed_ranking_scores_zero():
    assert weighted_auc(Y10, -np.arange(10.0)) == pytest.approx(0.0, abs=1e-15)


def test_all_tied_is_the_diagonal_not_one_half():
    # ROC = diagonal TPR = FPR. Band [0, .4]: .4*.4/2 + .4*.6 = .32 (x2); band [.4, 1]: .6*.6/2 = .18.
    assert weighted_auc(Y10, np.zeros(10)) == pytest.approx((2 * 0.32 + 0.18) / 1.4, abs=1e-15)
    assert plain_auc(Y10, np.zeros(10)) == pytest.approx(0.5, abs=1e-15)


def test_boundary_vertex_exactly_at_tpr_0_4():
    # 2 of 5 positives first (TPR .4 at FPR 0), then all negatives, then the rest: band 1 full, band 2 empty.
    s = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    s[[8, 9]] = 3.0  # two positives on top
    s[:5] = 2.0  # negatives next
    parts = weighted_auc_parts(Y10, s)
    assert parts["band_area"] == pytest.approx([0.4, 0.0], abs=1e-15)
    assert parts["weighted_auc"] == pytest.approx(0.8 / 1.4, abs=1e-15)
    assert legacy_metric()(Y10, s) == pytest.approx(0.8 / 1.4, abs=1e-12)


def test_segment_crossing_tpr_0_4_is_interpolated():
    # ROC (0,0)->(0,.2)->(.4,.6) [tie: 2 pos + 2 neg]->(.4,1)->(1,1). Hand integral: band areas .38 / .38.
    y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    s = np.array([10, 5, 5, 1, 1, 5, 5, 0, 0, 0], dtype=float)
    fpr, tpr = roc_vertices(y, s)
    assert fpr.tolist() == pytest.approx([0, 0, 0.4, 0.4, 1.0])
    assert tpr.tolist() == pytest.approx([0, 0.2, 0.6, 1.0, 1.0])
    parts = weighted_auc_parts(y, s)
    assert parts["band_area"] == pytest.approx([0.38, 0.38], abs=1e-15)
    assert parts["weighted_auc"] == pytest.approx(1.14 / 1.4, abs=1e-15)
    assert plain_auc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-15)


@pytest.mark.parametrize("n,seed", [(1_000, 0), (20_000, 1), (60_000, 2)])
def test_matches_legacy_on_continuous_random_scores(n, seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    s = rng.normal(size=n) + 0.8 * y
    assert weighted_auc(y, s) == pytest.approx(legacy_metric()(y, s), abs=1e-12)
    assert plain_auc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)


def test_legacy_differs_only_slightly_with_heavy_ties():
    # Legacy pads instead of interpolating the segment that crosses TPR .4; declared tolerance 1e-3.
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 20_000)
    s = np.round(rng.normal(size=20_000) + 0.8 * y, 1)
    assert abs(weighted_auc(y, s) - legacy_metric()(y, s)) < 1e-3
    assert plain_auc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)


def test_invariant_to_monotone_transform_and_row_order():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 5_000)
    s = rng.normal(size=5_000) + y
    base = weighted_auc(y, s)
    perm = rng.permutation(5_000)
    assert weighted_auc(y[perm], s[perm]) == pytest.approx(base, abs=1e-15)
    assert weighted_auc(y, 1 / (1 + np.exp(-3 * s))) == pytest.approx(base, abs=1e-12)


def test_rejects_degenerate_inputs():
    with pytest.raises(ValueError):
        weighted_auc(np.ones(5), np.arange(5.0))
    with pytest.raises(ValueError):
        weighted_auc(Y10, np.r_[np.arange(9.0), np.nan])


def test_binary_score_is_one_minus_p_cover_from_softmax():
    rng = np.random.default_rng(5)
    logits = rng.normal(size=(64, 4)) * 3
    p = softmax(logits)
    assert np.allclose(p.sum(axis=1), 1.0)
    s = binary_score(p)
    assert np.allclose(s, 1.0 - p[:, 0])
    assert np.allclose(s, p[:, 1:].sum(axis=1))  # = P(any stego class)
    # Cover logit up -> score down; any stego logit up -> score up.
    bumped = logits.copy()
    bumped[:, 0] += 1.0
    assert (binary_score(softmax(bumped)) < s).all()
    bumped = logits.copy()
    bumped[:, 2] += 1.0
    assert (binary_score(softmax(bumped)) > s).all()
    assert binary_target(np.array([0, 1, 2, 3])).tolist() == [0, 1, 1, 1]
    with pytest.raises(ValueError):
        binary_score(p[:, :3])


def test_softmax_is_stable_for_large_logits():
    p = softmax(np.array([[1000.0, 0.0, 0.0, 0.0], [-1000.0, 0.0, 0.0, 0.0]]))
    assert np.isfinite(p).all() and p[0, 0] == pytest.approx(1.0) and p[1, 0] == pytest.approx(0.0)


def test_torch_label_smoothing_equals_legacy_formula():
    # legacy/script/trainer.py LabelSmoothing: -(0.95 * logp[target] + 0.05 * mean(logp)) on one-hot targets.
    import torch

    g = torch.Generator().manual_seed(0)
    x = torch.randn(32, 4, generator=g) * 3
    t = torch.randint(0, 4, (32,), generator=g)
    logp = torch.log_softmax(x, dim=-1)
    legacy = -(0.95 * logp.gather(1, t[:, None]).squeeze(1) + 0.05 * logp.mean(-1)).mean()
    ours = torch.nn.CrossEntropyLoss(label_smoothing=0.05)(x, t)
    assert torch.allclose(ours, legacy, atol=1e-6)
