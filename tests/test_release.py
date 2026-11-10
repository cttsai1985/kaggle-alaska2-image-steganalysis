"""Portable source capsules exclude private/untracked inputs and bind every public file."""

import json
import subprocess
import zipfile

import pytest

from tools.package_offline import package_source


def test_source_capsule_uses_committed_files_and_verifiable_hashes(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "pyproject.toml").write_text('[project]\nname="alaska2"\nversion="0.1.0"\n')
    (root / "uv.lock").write_text("version = 1\n")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "public source",
        ],
        check=True,
    )
    (root / "token").write_text("private")
    out = tmp_path / "release"
    receipt = package_source(root, out)
    with zipfile.ZipFile(out / "alaska2-source.zip") as archive:
        assert set(archive.namelist()) == {"pyproject.toml", "uv.lock"}
    assert receipt["source_dirty"] is False
    assert json.loads((out / "offline_bundle.json").read_text()) == receipt
    assert "alaska2-source.zip" in (out / "SHA256SUMS").read_text()
    (root / "uv.lock").write_text("changed")
    with pytest.raises(ValueError, match="clean"):
        package_source(root, tmp_path / "dirty")


def test_release_version_matches_package_project_lock_and_installed_metadata():
    import importlib.metadata
    import tomllib
    from pathlib import Path

    from alaska2 import __version__

    root = Path(__file__).resolve().parents[1]
    project_version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    lock = tomllib.loads((root / "uv.lock").read_text())
    locked_version = next(package["version"] for package in lock["package"] if package["name"] == "alaska2")
    assert __version__ == project_version == locked_version == importlib.metadata.version("alaska2")
