"""Competition weighted AUC, implemented from the published definition without sklearn's ROC code.

Definition (competition evaluation page): the ROC curve is split into horizontal TPR bands
``[0, 0.4]`` and ``[0.4, 1]``; the area of each band under the curve is weighted 2 and 1, and the sum is
normalized by ``sum(weight * band height)`` = 1.4 so that a perfect ranking scores 1.

Implementation: exact ROC vertices from score ties (a tie group is a straight diagonal segment, the usual
ROC convention), then the closed-form integral of ``clip(TPR, lo, hi) - lo`` over FPR on each linear
segment. ``clip(y, lo, hi) - lo = relu(y - lo) - relu(y - hi)``, so each band needs only
``R(c) = integral of relu(TPR - c) dFPR``.

Known difference from the legacy/Kaggle-notebook code (``legacy/alaska_utils/eval_metrics.py``): that code
keeps only ROC points inside each band and pads to the band edge instead of interpolating the segment that
crosses a band edge. It agrees with this exact value when ROC vertices are dense near TPR 0.4 and differs
when a long segment crosses the edge (for all-tied scores: legacy 0.5714, exact 0.5857).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from alaska2.constants import CLASSES, COVER_INDEX, TPR_THRESHOLDS, TPR_WEIGHTS


def roc_vertices(y_true: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(fpr, tpr) at every distinct threshold, descending, starting at (0, 0) and ending at (1, 1)."""
    y = np.asarray(y_true).astype(bool).ravel()
    s = np.asarray(score, dtype=np.float64).ravel()
    if y.shape != s.shape:
        raise ValueError("y_true and score differ in length")
    if not np.isfinite(s).all():
        raise ValueError("score contains NaN or inf")
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("weighted AUC needs both positive and negative rows")
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    last_of_group = np.r_[s[1:] != s[:-1], True]  # close a vertex only after the whole tie group
    tp = np.cumsum(y)[last_of_group]
    fp = np.cumsum(~y)[last_of_group]
    return np.r_[0.0, fp / n_neg], np.r_[0.0, tp / n_pos]


def _area_above(fpr: np.ndarray, tpr: np.ndarray, c: float) -> float:
    """Exact integral over FPR of relu(TPR - c) for the piecewise-linear ROC."""
    x0, x1, y0, y1 = fpr[:-1], fpr[1:], tpr[:-1] - c, tpr[1:] - c
    dx = x1 - x0
    both = (y0 >= 0) & (y1 >= 0)
    cross = (y0 >= 0) != (y1 >= 0)
    area = np.where(both, dx * (y0 + y1) / 2.0, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        top = np.maximum(y0, y1)
        tri = dx * top * top / (2.0 * np.abs(y1 - y0))
    area = area + np.where(cross & (dx > 0), tri, 0.0)
    return float(area.sum())


def weighted_auc_parts(
    y_true: np.ndarray,
    score: np.ndarray,
    tpr_thresholds: Sequence[float] = TPR_THRESHOLDS,
    weights: Sequence[float] = TPR_WEIGHTS,
) -> dict:
    fpr, tpr = roc_vertices(y_true, score)
    t = np.asarray(tpr_thresholds, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if len(t) != len(w) + 1 or np.any(np.diff(t) <= 0):
        raise ValueError("tpr_thresholds must be increasing with one more entry than weights")
    above = [_area_above(fpr, tpr, c) for c in t]
    bands = np.asarray([above[i] - above[i + 1] for i in range(len(w))])
    norm = float(np.dot(np.diff(t), w))
    return {
        "weighted_auc": float(np.dot(bands, w) / norm),
        "band_area": bands.tolist(),  # unweighted area of each TPR band (max = band height)
        "band_height": np.diff(t).tolist(),
        "band_contribution": (bands * w / norm).tolist(),  # sums to weighted_auc
        "n_roc_vertices": int(len(fpr)),
    }


def weighted_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    return weighted_auc_parts(y_true, score)["weighted_auc"]


def plain_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """Standard ROC AUC by the same exact integration (one band [0, 1], weight 1)."""
    return weighted_auc_parts(y_true, score, (0.0, 1.0), (1.0,))["weighted_auc"]


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def binary_score(probs: np.ndarray) -> np.ndarray:
    """Competition probability that an image carries a payload: 1 - P(Cover) (legacy generate_submission)."""
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != len(CLASSES):
        raise ValueError(f"expected (n, {len(CLASSES)}) class probabilities in order {CLASSES}")
    return 1.0 - p[:, COVER_INDEX]


def binary_target(labels: np.ndarray) -> np.ndarray:
    return (np.asarray(labels) != COVER_INDEX).astype(np.int8)


def _summary(df: pl.DataFrame) -> dict:
    y = binary_target(df["label"].to_numpy())
    s = df["score"].to_numpy()
    out = {
        "rows": df.height,
        "families": df["family_id"].n_unique(),
        "positives": int(y.sum()),
        "negatives": int((1 - y).sum()),
    }
    if y.min() != y.max():
        parts = weighted_auc_parts(y, s)
        out.update(
            weighted_auc=parts["weighted_auc"],
            band_contribution=parts["band_contribution"],
            auc=plain_auc(y, s),
        )
    return out


def evaluation_report(pred: pl.DataFrame) -> dict:
    """Pooled metrics plus quality and stego-kind cohorts, each with its row/family counts."""
    probs = pred.select([f"p_{c}" for c in CLASSES]).to_numpy()
    acc = float((probs.argmax(axis=1) == pred["label"].to_numpy()).mean())
    out = {"pooled": _summary(pred), "four_class_accuracy": acc, "by_quality": {}, "cover_vs_kind": {}}
    for (q,), part in sorted(pred.group_by(["quality"]), key=lambda kv: kv[0][0]):
        out["by_quality"][str(q)] = _summary(part)
    for k in CLASSES[1:]:
        out["cover_vs_kind"][k] = _summary(pred.filter(pl.col("kind").is_in(["Cover", k])))
    return out
