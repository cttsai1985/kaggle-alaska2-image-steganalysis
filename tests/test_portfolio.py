"""The public writeup must stay tied to measured, hash-bound evidence."""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_portfolio_display_and_receipts_are_consistent():
    subprocess.run([sys.executable, str(ROOT / "scripts/check_portfolio.py")], check=True)


def test_portfolio_rejects_changed_public_score(tmp_path):
    shutil.copytree(ROOT / "docs", tmp_path / "docs")
    text = (ROOT / "README.md").read_text().replace("| 0.932 |", "| 0.999 |", 1)
    (tmp_path / "README.md").write_text(text)
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_portfolio.py"), "--root", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "README scores/ranks/percentages differ" in result.stderr
