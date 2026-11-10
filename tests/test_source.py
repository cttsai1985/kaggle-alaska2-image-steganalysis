"""Standalone images retain an explicit source revision instead of silently losing provenance."""

import json

import pytest

from alaska2.identity import git_source, maintained_source_sha256, package_source_sha256


def test_build_manifest_supplies_source_when_git_metadata_is_unavailable(tmp_path):
    metadata = {
        "git_sha": "a" * 40,
        "dirty": False,
        "package_sha256": package_source_sha256(),
        "maintained_sha256": maintained_source_sha256(tmp_path),
    }
    (tmp_path / "source_revision.json").write_text(json.dumps(metadata))
    source = git_source(tmp_path)
    assert source == metadata


def test_changed_runtime_code_cannot_inherit_clean_build_status(tmp_path):
    metadata = {
        "git_sha": "a" * 40,
        "dirty": False,
        "package_sha256": "0" * 64,
        "maintained_sha256": maintained_source_sha256(tmp_path),
    }
    (tmp_path / "source_revision.json").write_text(json.dumps(metadata))
    source = git_source(tmp_path)
    assert source["dirty"] is True
    assert source["package_sha256"] == package_source_sha256()


@pytest.mark.parametrize("filename", ["fixture.py", "runs/retained_gpu.sh"])
def test_standalone_image_detects_changed_executable_tool(tmp_path, filename):
    from alaska2.identity import maintained_source_sha256

    tool = tmp_path / "tools" / filename
    tool.parent.mkdir(parents=True)
    tool.write_text("value = 1\n")
    metadata = {
        "git_sha": "a" * 40,
        "dirty": False,
        "package_sha256": package_source_sha256(),
        "maintained_sha256": maintained_source_sha256(tmp_path),
    }
    (tmp_path / "source_revision.json").write_text(json.dumps(metadata))
    assert git_source(tmp_path)["dirty"] is False
    tool.write_text("value = 2\n")
    assert git_source(tmp_path)["dirty"] is True
