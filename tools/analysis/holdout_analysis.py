"""Holdout analysis from saved predictions only (CPU; never loads a model or an image).

Reads one ``predictions/valid_epochNNN.parquet`` (+ JSON sidecar) and writes an aggregate JSON and PNG plots:

* pooled competition weighted AUC (exact) vs the legacy implementation vs standard AUC;
* ROC region contributions: TPR band [0, 0.4] (weight 2) and [0.4, 1] (weight 1), each band's shortfall
  from its maximum, and operating points (TPR at fixed FPR, FPR at fixed TPR);
* cohorts by JPEG quality, by stego kind (Cover vs kind, same covers) and kind x quality, each with
  row / family / positive / negative counts;
* family-level bootstrap intervals (resampling holdout families, Cover and its stego variants together).
  These describe sampling uncertainty of this one holdout for this one model. They are not seed or
  training variance, and they are not cross-validation.

Output contains aggregates only: no image ids, family ids or per-row values.

    uv run --group analysis python -m tools.analysis.holdout_analysis \
        --predictions RUN/predictions/valid_epoch00E.parquet --out-dir evidence/phase2
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import polars as pl

from alaska2.constants import CLASSES
from alaska2.identity import sha256_file
from alaska2.metrics import binary_target, plain_auc, roc_vertices, weighted_auc_parts

LEGACY_METRIC = Path(__file__).resolve().parents[2] / "legacy" / "alaska_utils" / "eval_metrics.py"
FPR_POINTS = (0.001, 0.01, 0.05, 0.1)
# Diagonal (chance / all-tied) ROC under the competition definition: band areas 0.32 and 0.18.
CHANCE_BAND_CONTRIBUTION = (2 * 0.32 / 1.4, 1 * 0.18 / 1.4)
CHANCE_WEIGHTED_AUC = sum(CHANCE_BAND_CONTRIBUTION)  # 0.5857, not 0.5
TPR_POINTS = (0.4, 0.8, 0.9)


def _legacy_weighted_auc():
    spec = importlib.util.spec_from_file_location("legacy_eval_metrics", LEGACY_METRIC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.alaska_weighted_auc


def operating_points(y: np.ndarray, s: np.ndarray) -> dict:
    """Interpolated on the exact ROC polyline (vertices are sorted by threshold, so fpr/tpr are monotone)."""
    fpr, tpr = roc_vertices(y, s)

    def fpr_at(t: float) -> float:  # smallest FPR on the polyline whose TPR reaches t
        i = int(np.searchsorted(tpr, t, side="left"))
        if i == 0:
            return float(fpr[0])
        y0, y1 = tpr[i - 1], tpr[i]
        return float(fpr[i - 1] + (fpr[i] - fpr[i - 1]) * (t - y0) / (y1 - y0))

    def tpr_at(f: float) -> float:  # largest TPR on the polyline at FPR f (top of a vertical jump)
        i = int(np.searchsorted(fpr, f, side="right"))
        if i >= len(fpr):
            return float(tpr[-1])
        x0, x1 = fpr[i - 1], fpr[i]
        return float(tpr[i - 1] + (tpr[i] - tpr[i - 1]) * (f - x0) / (x1 - x0))

    return {
        "tpr_at_fpr": {str(f): tpr_at(f) for f in FPR_POINTS},
        "fpr_at_tpr": {str(t): fpr_at(t) for t in TPR_POINTS},
    }


def summary(df: pl.DataFrame, with_points: bool = False) -> dict:
    y = binary_target(df["label"].to_numpy())
    s = df["score"].to_numpy()
    out = {
        "rows": df.height,
        "families": df["family_id"].n_unique(),
        "positives": int(y.sum()),
        "negatives": int((1 - y).sum()),
    }
    parts = weighted_auc_parts(y, s)
    out.update(
        weighted_auc=parts["weighted_auc"],
        auc=plain_auc(y, s),
        band_area=parts["band_area"],
        band_contribution=parts["band_contribution"],
        band_lift_over_chance=[
            c - k for c, k in zip(parts["band_contribution"], CHANCE_BAND_CONTRIBUTION, strict=True)
        ],
        weighted_auc_lift_over_chance=parts["weighted_auc"] - CHANCE_WEIGHTED_AUC,
        auc_lift_over_chance=plain_auc(y, s) - 0.5,
        band_shortfall=[
            h * w / 1.4 - c
            for h, w, c in zip(parts["band_height"], (2.0, 1.0), parts["band_contribution"], strict=True)
        ],
    )
    if with_points:
        out["operating_points"] = operating_points(y, s)
    return out


def family_bootstrap(df: pl.DataFrame, n_boot: int, seed: int, cohort_col: str | None = None) -> dict:
    """Percentile intervals for weighted AUC / AUC, resampling whole families with replacement."""
    fam = df["family_id"].to_numpy()
    uniq, inv = np.unique(fam, return_inverse=True)
    y = binary_target(df["label"].to_numpy())
    s = df["score"].to_numpy()
    groups = df[cohort_col].to_numpy() if cohort_col else None
    rows_by_fam = np.argsort(inv, kind="stable")
    bounds = np.r_[0, np.cumsum(np.bincount(inv, minlength=len(uniq)))]
    rng = np.random.default_rng(seed)
    keys = ["pooled"] if groups is None else [str(g) for g in np.unique(groups)]
    draws: dict[str, dict[str, list[float]]] = {k: {"weighted_auc": [], "auc": []} for k in keys}
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([rows_by_fam[bounds[f] : bounds[f + 1]] for f in pick])
        for k in keys:
            sel = idx if groups is None else idx[groups[idx].astype(str) == k]
            draws[k]["weighted_auc"].append(weighted_auc_parts(y[sel], s[sel])["weighted_auc"])
            draws[k]["auc"].append(plain_auc(y[sel], s[sel]))
    out = {}
    for k, d in draws.items():
        out[k] = {
            m: {
                "p2.5": float(np.percentile(v, 2.5)),
                "p97.5": float(np.percentile(v, 97.5)),
                "sd": float(np.std(v, ddof=1)),
            }
            for m, v in d.items()
        }
    return out


def confusion(df: pl.DataFrame) -> dict:
    probs = df.select([f"p_{c}" for c in CLASSES]).to_numpy()
    pred = probs.argmax(axis=1)
    lab = df["label"].to_numpy()
    m = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    np.add.at(m, (lab, pred), 1)
    return {
        "rows_true_cols_pred": m.tolist(),
        "classes": list(CLASSES),
        "per_class_recall": {c: float(m[i, i] / m[i].sum()) for i, c in enumerate(CLASSES)},
        "four_class_accuracy": float(np.trace(m) / m.sum()),
    }


def score_quantiles(df: pl.DataFrame) -> dict:
    out = {}
    for (kind, q), part in sorted(df.group_by(["kind", "quality"]), key=lambda kv: kv[0]):
        v = part["score"].to_numpy()
        out[f"{kind}|q{q}"] = {p: float(np.percentile(v, float(p))) for p in ("5", "25", "50", "75", "95")}
    return out


def plots(df: pl.DataFrame, out_dir: Path, label: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = []
    colors = {"75": "#1f77b4", "90": "#ff7f0e", "95": "#2ca02c", "pooled": "#222222"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    cohorts = [("pooled", df)] + [
        (str(q), part) for (q,), part in sorted(df.group_by(["quality"]), key=lambda kv: kv[0][0])
    ]
    for ax, logx in zip(axes, (False, True), strict=True):
        if not logx:
            ax.axhspan(0, 0.4, color="#d62728", alpha=0.08, label="TPR [0, 0.4]: weight 2")
            ax.axhspan(0.4, 1, color="#7f7f7f", alpha=0.06, label="TPR [0.4, 1]: weight 1")
            ax.plot(
                [0, 1],
                [0, 1],
                color="#999999",
                lw=0.8,
                ls=":",
                label=f"chance (wAUC {CHANCE_WEIGHTED_AUC:.4f})",
            )
        for name, part in cohorts:
            fpr, tpr = roc_vertices(binary_target(part["label"].to_numpy()), part["score"].to_numpy())
            wa = weighted_auc_parts(binary_target(part["label"].to_numpy()), part["score"].to_numpy())
            tag = "pooled" if name == "pooled" else f"QF {name}"
            ax.plot(
                fpr,
                tpr,
                color=colors[name],
                lw=2.0 if name == "pooled" else 1.3,
                label=f"{tag} (wAUC {wa['weighted_auc']:.4f}, n={part.height})",
            )
        ax.axhline(0.4, color="#d62728", lw=0.8, ls="--")
        ax.set_xlabel("FPR (Cover flagged as stego)" + (", log scale" if logx else ""))
        ax.set_ylabel("TPR (stego detected)")
        if logx:
            ax.set_xscale("log")
            ax.set_xlim(1e-4, 1)
        else:
            ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle(f"Holdout ROC, legacy fold 1 single holdout ({label})", fontsize=10)
    fig.tight_layout()
    p = out_dir / "holdout_roc.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    files.append(p.name)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    rows = []
    for (q,), part in sorted(df.group_by(["quality"]), key=lambda kv: kv[0][0]):
        for k in CLASSES[1:]:
            sub = part.filter(pl.col("kind").is_in(["Cover", k]))
            rows.append((f"QF{q}\n{k}", summary(sub)))
    x = np.arange(len(rows))
    c0 = [r[1]["band_contribution"][0] for r in rows]
    c1 = [r[1]["band_contribution"][1] for r in rows]
    ax.bar(x, c0, color="#d62728", alpha=0.75, label="TPR [0, 0.4] contribution (max 0.571)")
    ax.bar(x, c1, bottom=c0, color="#7f7f7f", alpha=0.75, label="TPR [0.4, 1] contribution (max 0.429)")
    ax.axhline(
        CHANCE_WEIGHTED_AUC,
        color="black",
        lw=1,
        ls=":",
        label=f"chance (diagonal ROC) {CHANCE_WEIGHTED_AUC:.4f}",
    )
    ax.axhline(
        CHANCE_BAND_CONTRIBUTION[0], color="#d62728", lw=0.8, ls=":", label="chance, TPR [0, 0.4] band"
    )
    ax.set_xticks(x, [r[0] for r in rows], fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("weighted AUC (stacked band contributions)")
    ax.set_title("Cover vs each stego kind, by JPEG quality (same covers within a quality)", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = out_dir / "holdout_cohorts.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    files.append(p.name)
    return files


def analyse(pred_path: Path, out_dir: Path, n_boot: int, seed: int, make_plots: bool = True) -> dict:
    meta = json.loads(pred_path.with_suffix(".json").read_text())
    if meta.get("split") != "valid":
        raise ValueError("analysis needs holdout (split = valid) predictions")
    prediction_sha = sha256_file(pred_path)
    if prediction_sha != meta["predictions_sha256"]:
        raise ValueError("prediction hash differs from its sidecar")
    df = pl.read_parquet(pred_path)
    y = binary_target(df["label"].to_numpy())
    s = df["score"].to_numpy()

    from sklearn.metrics import roc_auc_score

    pooled = summary(df, with_points=True)
    pooled["weighted_auc_legacy_impl"] = float(_legacy_weighted_auc()(y, s))
    pooled["auc_sklearn"] = float(roc_auc_score(y, s))
    pooled["distinct_scores"] = int(np.unique(s).size)

    by_quality = {
        str(q): summary(part, with_points=True)
        for (q,), part in sorted(df.group_by(["quality"]), key=lambda kv: kv[0][0])
    }
    by_kind = {k: summary(df.filter(pl.col("kind").is_in(["Cover", k])), True) for k in CLASSES[1:]}
    kind_quality = {}
    for (q,), part in sorted(df.group_by(["quality"]), key=lambda kv: kv[0][0]):
        for k in CLASSES[1:]:
            kind_quality[f"{k}|q{q}"] = summary(part.filter(pl.col("kind").is_in(["Cover", k])))

    def rank(d: dict, m: str) -> list[str]:
        return sorted(d, key=lambda k: -d[k][m])

    out = {
        "kind": "holdout_analysis",
        "source": "saved predictions only (no model reload, no images)",
        "result_label": meta["result_label"],
        "population": "legacy fold 1 single holdout (family-level), not cross-validation",
        "predictions_sha256": prediction_sha,
        "model_sha256": meta["model_sha256"],
        "epoch": meta["epoch"],
        "rows": df.height,
        "families": df["family_id"].n_unique(),
        "coverage": {
            "by_kind": dict(sorted(df.group_by("kind").len().iter_rows())),
            "by_quality": {str(k): v for k, v in sorted(df.group_by("quality").len().iter_rows())},
            "kind_by_quality": {
                f"{k}|q{q}": n for k, q, n in sorted(df.group_by(["kind", "quality"]).len().iter_rows())
            },
            "families_complete_4_variants": int((df.group_by("family_id").len()["len"] == 4).sum()),
        },
        "metric_definition": {
            "tpr_bands": [[0.0, 0.4], [0.4, 1.0]],
            "weights": [2, 1],
            "normalization": 1.4,
            "max_band_contribution": [0.8 / 1.4, 0.6 / 1.4],
            "chance_band_contribution": list(CHANCE_BAND_CONTRIBUTION),
            "chance_weighted_auc": CHANCE_WEIGHTED_AUC,
            "chance_auc": 0.5,
            "score": "1 - p_Cover",
        },
        "pooled": pooled,
        "by_quality": by_quality,
        "cover_vs_kind": by_kind,
        "cover_vs_kind_by_quality": kind_quality,
        "rank_agreement": {
            "quality_by_weighted_auc": rank(by_quality, "weighted_auc"),
            "quality_by_auc": rank(by_quality, "auc"),
            "kind_by_weighted_auc": rank(by_kind, "weighted_auc"),
            "kind_by_auc": rank(by_kind, "auc"),
        },
        "confusion": confusion(df),
        "score_quantiles_by_kind_quality": score_quantiles(df),
        "bootstrap": {
            "method": f"family-level resampling with replacement, {n_boot} draws, numpy seed {seed}",
            "meaning": "sampling uncertainty of this holdout for this one trained model; "
            "not seed/training variance, not CV",
            "pooled": family_bootstrap(df, n_boot, seed)["pooled"],
            "by_quality": family_bootstrap(df, n_boot, seed + 1, "quality"),
        },
        "redaction": "aggregates only; no image ids, family ids or per-row values",
    }
    if sha256_file(pred_path) != prediction_sha:
        raise ValueError("prediction hash changed during analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    if make_plots:
        out["plots"] = plots(df, out_dir, meta["result_label"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--out-name", default="holdout_analysis.json")
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261109)
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()
    res = analyse(a.predictions, a.out_dir, a.bootstrap, a.seed, make_plots=not a.no_plots)
    (a.out_dir / a.out_name).write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: res[k] for k in ("result_label", "rows", "pooled")}, indent=2))


if __name__ == "__main__":
    main()
