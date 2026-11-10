"""Guard the frozen portfolio contract and source-backed front-page result display."""

import argparse
import hashlib
import json
import re
import subprocess
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STANDARD_SHA256 = "39802103af834f8e3856bd49892b888c03870194d4ca8718ba986b39c1d226fc"
START = "<!-- portfolio-results:start -->"
END = "<!-- portfolio-results:end -->"


def public_results(context, root=None):
    root = root or ROOT
    if context.get("status") == "draft":
        raise ValueError("Draft portfolio: recover paired submission and reference evidence before release")
    source = context["source"]
    count = source["reference_teams"]
    if not isinstance(count, int) or count < 1:
        raise ValueError("Invalid reference population")
    metric = context.get("metric", "r2")
    metric_label = {
        "mean_spearman": "mean Spearman ↑",
        "roc_auc": "ROC AUC ↑",
        "log_loss": "log loss ↓",
        "r2": "R² ↑",
        "weighted_auc": "weighted AUC ↑",
    }[metric]
    snapshot = source.get("export_snapshot_utc")
    if not snapshot:
        found = re.search(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d", source["csv_filename"])
        if not found:
            raise ValueError("Missing dated leaderboard snapshot")
        snapshot = found[0] + "Z"
    if context.get("schema_version") == 2:
        provenance = source["submission_source"]
        path = root / provenance["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != provenance["sha256"]:
            raise ValueError("Paired submission source hash differs")
        paired_rows = {row["ref"]: row for row in json.loads(path.read_text())}
        histogram = source["score_histogram"]
        if sum(row["teams"] for row in histogram) != count:
            raise ValueError("Reference histogram does not reconcile")
    rows = []
    for item in context.get("scores", context.get("comparisons", [])):
        if item.get("status") == "not_submitted":
            label = item["label"].replace("|", "/").replace("\n", " ")
            rows.append(f"| {label} | Not submitted | Not submitted | Not applicable |")
            continue
        if "submission" in item:
            submission = item["submission"]
            label = item.get(
                "label",
                f"Historical submission {submission['ref']} ({submission['date'][:10]})",
            )
            if context.get("schema_version") == 2:
                if paired_rows.get(submission["ref"]) != submission:
                    raise ValueError("Scores differ from the retained paired submission")
                public = Decimal(submission["publicScore"])
                better = sum(
                    row["teams"]
                    for row in histogram
                    if (
                        Decimal(row["score"]) > public
                        if context["direction"] == "higher"
                        else Decimal(row["score"]) < public
                    )
                )
                equal_count = sum(row["teams"] for row in histogram if Decimal(row["score"]) == public)
                if (
                    better + 1 != item["score_equivalent_start_position"]
                    or equal_count != item["tied_reference_teams"]
                ):
                    raise ValueError("Score comparison differs from retained histogram")
            score = submission["publicScore"]
            private = submission["privateScore"]
            position = item["score_equivalent_start_position"]
            equal = item["tied_reference_teams"]
            top = item["top_percent"]
        else:
            label, score = item["label"], item["public_score"]
            private = item["private_score"]
            provenance = item["private_score_source"]
            source_path = root / provenance["path"]
            if source_path.suffix == ".json":
                paired = json.loads(source_path.read_text())
                for key in provenance["pointer"].split("/")[1:]:
                    paired = paired[int(key)] if isinstance(paired, list) else paired[key]
                if paired["publicScore"] != score or paired["privateScore"] != private:
                    raise ValueError("Public/private scores differ from the paired submission")
            elif f"**{score}**" not in source_path.read_text() or (
                f"**{private}**" not in source_path.read_text()
            ):
                raise ValueError("Historical score pair missing from its retained source")
            position = item["score_equivalent_best_position"]
            equal = item["teams_with_equal_displayed_score"]
            top = item["top_percent_by_best_position"]
            better, worse = (
                item["teams_with_strictly_better_score"],
                item["teams_with_strictly_worse_score"],
            )
            if position != 1 + better or better + equal + worse != count:
                raise ValueError("Reference counts do not reconcile")
        if item["reference_teams"] != count or equal < 0 or not Decimal(score).is_finite():
            raise ValueError("Invalid score/reference scope")
        calculated = Decimal(100) * position / count
        if abs(calculated - Decimal(str(top))) > Decimal("1e-10"):
            raise ValueError("Top percentage differs from position/denominator")
        last = position + max(equal, 1) - 1
        if (
            "score_equivalent_last_tied_position" in item
            and last != item["score_equivalent_last_tied_position"]
        ):
            raise ValueError("Tie interval differs from reference count")
        if not Decimal(private).is_finite():
            raise ValueError("Invalid private leaderboard score")
        label = label.replace("|", "/").replace("\n", " ")
        rows.append(f"| {label} | {score} | {private} | **{calculated:.2f}%** |")
    if not rows:
        raise ValueError("Missing scored comparisons")
    text = [
        START,
        "## Results",
        "",
        f"**Leaderboard scores — {metric_label}.**",
        "",
        "| Model / submission | Public LB | Private LB | Public Top % |",
        "|---|---:|---:|---:|",
        *rows,
        "",
        (
            f"Public Top % uses the **{snapshot[:10]} public leaderboard snapshot** "
            "(score-equivalent comparison, not an awarded final rank). "
            "[Detailed results](docs/results.md) · "
            "[Reference ranks, ties and calculation](docs/leaderboard-context.json)."
        ),
        "",
        "Private Top % is unavailable without a matched private leaderboard; "
        "OOF/holdout results have no leaderboard percentile.",
        END,
    ]
    return "\n".join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Refresh the visible table from the retained reference receipt",
    )
    parser.add_argument("--kind", choices=["solution", "infrastructure"], default="solution")
    parser.add_argument(
        "--check-github",
        action="store_true",
        help="Compare the reviewed public profile with live GitHub metadata (requires gh)",
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    standard = root / "docs/portfolio-standard.md"
    if hashlib.sha256(standard.read_bytes()).hexdigest() != STANDARD_SHA256:
        raise ValueError("Portfolio standard changed without its versioned acceptance update")
    audit = (root / "docs/requirements-audit.md").read_text()
    missing = [f"R{i:02}" for i in range(1, 27) if not re.search(rf"\bR{i:02}\b", audit)]
    if missing:
        raise ValueError(f"Incomplete requirement coverage: {missing}")
    if args.kind == "infrastructure":
        for name in ["requirements.md", "requirements-audit.md"]:
            if not (root / "docs" / name).is_file():
                raise ValueError(f"Missing infrastructure requirement evidence: {name}")
        print("Frozen portfolio standard and infrastructure applicability record verified")
        return
    if args.check_github:
        profile = json.loads((root / "docs/public-profile.json").read_text())
        live = json.loads(
            subprocess.check_output(
                ["gh", "repo", "view", profile["repository"], "--json", "name,url,description"],
                text=True,
            )
        )
        if live != profile["metadata"]:
            raise ValueError("GitHub metadata differs from the reviewed public profile")
    expected = public_results(json.loads((root / "docs/leaderboard-context.json").read_text()), root)
    path = root / "README.md"
    readme = path.read_text()
    if args.write:
        if START in readme:
            first, tail = readme.split(START, 1)
            _, tail = tail.split(END, 1)
            readme = first + expected + tail
        else:
            opening, separator, remainder = readme.partition("\n## ")
            if not separator:
                raise ValueError("Missing README section structure")
            readme = opening.rstrip() + "\n\n" + expected + "\n\n## " + remainder
        path.write_text(readme)
        print("Updated the primary results display; run without --write for the full acceptance check")
        return
    if readme.count(START) != 1 or readme.count(END) != 1:
        raise ValueError("Missing/duplicate primary results display")
    actual = START + readme.split(START, 1)[1].split(END, 1)[0] + END
    if actual != expected:
        raise ValueError("README scores/ranks/percentages differ from the reference; inspect before --write")
    if readme.split(START, 1)[0].count("\n") > 35 or readme.split(START, 1)[0].count("\n## "):
        raise ValueError("Results must be visible before model/reproduction/infrastructure detail")
    if "docs/error-analysis.md" not in readme:
        raise ValueError("README must link the project-specific error analysis")
    for name in [
        "results.md",
        "modeling-study.md",
        "problem-and-value.md",
        "requirements-audit.md",
        "error-analysis.md",
    ]:
        document = root / "docs" / name
        if not document.is_file():
            raise ValueError(f"Missing required portfolio evidence: {name}")
        if "Draft —" in document.read_text():
            raise ValueError(f"Unfinished portfolio evidence: {name}")
    analysis = json.loads((root / "docs/error-analysis.json").read_text())
    recorded = analysis.get("analyzer_sha256") or analysis["source_sha256"]["analysis_script"]
    configuration = root / "docs/portfolio.json"
    settings = json.loads(configuration.read_text()) if configuration.exists() else {}
    analyzer = settings.get("analysis_script", "scripts/analyze_hard_cases.py")
    measured = hashlib.sha256((root / analyzer).read_bytes()).hexdigest()
    if recorded != measured:
        raise ValueError("Error-analysis receipt does not identify the current analyzer")
    print("Frozen standard, primary score/rank display and required analysis links verified")


if __name__ == "__main__":
    main()
