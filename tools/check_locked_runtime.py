"""Reject mismatched Python/direct runtime versions before an offline teaching rerun."""

from __future__ import annotations

import argparse
import importlib.metadata
import re
import sys
import tomllib
from pathlib import Path


def check_locked_runtime(root: Path) -> None:
    if sys.version_info[:2] != (3, 12):
        raise ValueError("use the locked Python 3.12 runtime")
    project = tomllib.loads((root / "pyproject.toml").read_text())
    locked = {p["name"]: p["version"] for p in tomllib.loads((root / "uv.lock").read_text())["package"]}
    needed = project["project"]["dependencies"] + sum(
        [project["dependency-groups"][group] for group in ("lab", "analysis")], []
    )
    for requirement in needed:
        name = re.match(r"[A-Za-z0-9_-]+", requirement).group().lower().replace("_", "-")
        if importlib.metadata.version(name) != locked[name]:
            raise ValueError(f"install locked offline dependency: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    check_locked_runtime(parser.parse_args().root)


if __name__ == "__main__":
    main()
