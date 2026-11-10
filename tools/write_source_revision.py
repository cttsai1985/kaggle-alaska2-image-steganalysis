"""Bind a standalone image's actual package bytes to the explicitly supplied build revision."""

import argparse
import json
import re
from pathlib import Path

from alaska2.identity import maintained_source_sha256, package_source_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--git-sha", required=True)
    parser.add_argument("--dirty", choices=["0", "1"], default="0")
    parser.add_argument("--out", type=Path, default=Path("source_revision.json"))
    args = parser.parse_args()
    if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", args.git_sha):
        raise ValueError("supply the actual source Git revision")
    args.out.write_text(
        json.dumps(
            {
                "git_sha": args.git_sha,
                "dirty": args.dirty == "1",
                "package_sha256": package_source_sha256(),
                "maintained_sha256": maintained_source_sha256(Path.cwd()),
            },
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
