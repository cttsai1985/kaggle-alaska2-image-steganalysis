"""Build a public source/lock capsule from clean committed source; never include private runtime files."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True).stdout


def package_source(root: Path, output: Path) -> dict:
    """Use Git's committed tree rather than scanning a checkout containing private runtime outputs."""
    root, output = root.resolve(), output.resolve()
    if _git(root, "status", "--porcelain", "--untracked-files=no").strip():
        raise ValueError("source packaging requires clean committed source")
    if output.is_relative_to(root):
        raise ValueError("release outputs must be outside source")
    output.mkdir(parents=True, exist_ok=True)
    source = _git(root, "rev-parse", "HEAD").decode().strip()
    archive = _git(root, "archive", "--format=zip", "HEAD")
    files = {"alaska2-source.zip": archive, "uv.lock": (root / "uv.lock").read_bytes()}
    receipt = {
        "kind": "offline_source_capsule",
        "measurement_status": "MEASURED",
        "source_git_sha": source,
        "source_dirty": False,
        "hosted_kaggle_execution": "unverified",
        "dependencies_included": False,
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
    }
    files["offline_bundle.json"] = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    for name, data in files.items():
        path = output / name
        if path.exists() and path.read_bytes() != data:
            raise ValueError(f"conflicting release asset: {name}")
        path.write_bytes(data)
    (output / "SHA256SUMS").write_text(
        "".join(f"{hashlib.sha256(data).hexdigest()}  {name}\n" for name, data in sorted(files.items()))
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package_source(Path(__file__).resolve().parents[1], args.out), indent=2))


if __name__ == "__main__":
    main()
