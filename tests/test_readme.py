"""Keep the public solution summary tied to its retained evidence."""

import hashlib
import json
import re
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_preserves_author_and_results_first():
    text = (ROOT / "README.md").read_text()
    assert "My Image Steganalysis Solution" in text
    assert "Chia-Ta Tsai" in text
    assert text.split("\n## ")[1].startswith("Results\n")
    assert "0.83540" in text
    assert "incomplete-run fallback" in text
    assert "not an awarded final rank" in text


def test_readme_historical_scores_match_paired_records_and_reference():
    text = (ROOT / "README.md").read_text()
    context = json.loads((ROOT / "docs/leaderboard-context.json").read_text())
    source = context["source"]
    paired_path = ROOT / source["submission_source"]["path"]
    paired_digest = hashlib.sha256(paired_path.read_bytes()).hexdigest()
    assert paired_digest == source["submission_source"]["sha256"]
    paired = {row["ref"]: row for row in json.loads(paired_path.read_text())}
    assert len(paired) == 3
    histogram = source["score_histogram"]
    count = source["reference_teams"]
    assert sum(row["teams"] for row in histogram) == count
    assert source["export_snapshot_utc"][:10] in text
    details = (ROOT / "docs/results.md").read_text()
    assert "Scores and submission provenance have not been recovered" not in details
    for item in context["scores"]:
        if item.get("status") == "not_submitted":
            assert "| Not submitted | Not submitted | Not applicable |" in text
            continue
        row = paired[item["submission"]["ref"]]
        assert row == item["submission"]
        score = Decimal(row["publicScore"])
        better = sum(group["teams"] for group in histogram if Decimal(group["score"]) > score)
        tied = sum(group["teams"] for group in histogram if Decimal(group["score"]) == score)
        assert item["score_equivalent_start_position"] == better + 1
        assert item["score_equivalent_last_tied_position"] == better + tied
        assert item["tied_reference_teams"] == tied
        percent = Decimal(100) * (better + 1) / count
        assert Decimal(item["top_percent"]) == percent
        expected = f"| {item['label']} | {row['publicScore']} | {row['privateScore']} | **{percent:.2f}%** |"
        assert expected in text
        assert f"public {row['publicScore']} / private {row['privateScore']}" in details


def test_readme_relative_links_resolve():
    text = (ROOT / "README.md").read_text()
    for target in re.findall(r"\]\(([^)]+)\)", text):
        if "://" in target or target.startswith("#"):
            continue
        assert (ROOT / target.split("#", 1)[0]).is_file(), target
