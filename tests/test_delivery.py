"""Scientific identity survives failed transfers; completion requires destination byte verification."""

import json

import pytest

from alaska2.delivery import DeliveryError, RecoveryRequired, create_bundle, deliver, verify_bundle


def bundle(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"unchanged model checkpoint")
    return create_bundle(
        {"run_id": "fixture", "model_sha256": "weights", "result_label": "FIXTURE_NOT_A_MODEL_RESULT"},
        {"models/model.bin": source},
        tmp_path / "bundle",
    )


def test_bundle_identity_and_repeated_transfer_are_stable(tmp_path):
    path = bundle(tmp_path)
    manifest = verify_bundle(path)
    assert manifest["project_id"] == "alaska2"
    first = deliver(path, tmp_path / "destination", tmp_path / "state")
    again = deliver(path, tmp_path / "destination", tmp_path / "state")
    assert first["experiment_id"] == again["experiment_id"] == manifest["experiment_id"]
    assert first["state"] == "delivered" and again["reused"] is True
    assert again["transport"] == "filesystem"


@pytest.mark.parametrize("fault", ["after_copy", "corrupt_copy", "after_publish"])
def test_failed_transfer_requires_explicit_recovery_without_retraining(tmp_path, fault):
    path = bundle(tmp_path)
    before = verify_bundle(path)
    with pytest.raises(DeliveryError):
        deliver(path, tmp_path / "destination", tmp_path / "state", fault=fault)
    assert verify_bundle(path) == before
    with pytest.raises(RecoveryRequired):
        deliver(path, tmp_path / "destination", tmp_path / "state")
    recovered = deliver(path, tmp_path / "destination", tmp_path / "state", recover=True)
    assert recovered["state"] == "delivered" and recovered["attempt"] == 2
    assert recovered["experiment_id"] == before["experiment_id"]
    assert verify_bundle(path) == before


def test_completed_marker_cannot_hide_changed_destination_bytes(tmp_path):
    path = bundle(tmp_path)
    result = deliver(path, tmp_path / "destination", tmp_path / "state")
    destination = tmp_path / "destination" / result["experiment_id"]
    pointer = json.loads((destination / "current.json").read_text())
    (destination / pointer["directory"] / "models/model.bin").write_bytes(b"corrupt")
    with pytest.raises(DeliveryError):
        deliver(path, tmp_path / "destination", tmp_path / "state")
    assert deliver(path, tmp_path / "destination", tmp_path / "state", recover=True)["state"] == "delivered"


def test_conflicting_bundle_and_path_escape_are_rejected(tmp_path):
    path = bundle(tmp_path)
    source = tmp_path / "changed.bin"
    source.write_bytes(b"changed")
    with pytest.raises(DeliveryError):
        create_bundle(
            {"run_id": "fixture", "model_sha256": "weights", "result_label": "FIXTURE_NOT_A_MODEL_RESULT"},
            {"models/model.bin": source},
            path,
        )
    with pytest.raises(ValueError):
        create_bundle({"run_id": "unsafe"}, {"../outside.bin": source}, tmp_path / "unsafe")


def test_rejected_quality_is_retained_research_not_a_transfer_retry(tmp_path):
    path = bundle(tmp_path)
    original = verify_bundle(path)
    result = deliver(path, tmp_path / "destination", tmp_path / "state", quality_accepted=False)
    assert result["state"] == "quality_rejected"
    assert result["experiment_id"] == original["experiment_id"]
    assert verify_bundle(path) == original
    assert not (tmp_path / "destination").exists()


def test_live_importer_cannot_be_taken_over_and_killed_importer_can_recover(tmp_path):
    import os
    import subprocess
    import sys
    import time
    from pathlib import Path

    path = bundle(tmp_path)
    ready = tmp_path / "copy-started"
    script = """
import sys,time
from pathlib import Path
import alaska2.delivery as d
bundle,destination,state,ready=map(Path,sys.argv[1:])
copy=d.shutil.copyfile
def pause(source,target):
    result=copy(source,target)
    ready.write_text('copy reached')
    while True: time.sleep(1)
    return result
d.shutil.copyfile=pause
d.deliver(bundle,destination,state)
"""
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            script,
            str(path),
            str(tmp_path / "destination"),
            str(tmp_path / "state"),
            str(ready),
        ],
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 15
        while not ready.exists() and time.monotonic() < deadline:
            assert proc.poll() is None, "importer exited before reaching the controlled copy"
            time.sleep(0.02)
        assert ready.exists(), "controlled copy was not reached"
        with pytest.raises(DeliveryError, match="active"):
            deliver(path, tmp_path / "destination", tmp_path / "alternate-state", recover=True)
    finally:
        proc.kill()
        proc.communicate(timeout=10)
    with pytest.raises(RecoveryRequired):
        deliver(path, tmp_path / "destination", tmp_path / "state")
    result = deliver(path, tmp_path / "destination", tmp_path / "state", recover=True)
    assert result["state"] == "delivered" and result["attempt"] == 2


def test_bundle_control_files_cannot_follow_symlinks(tmp_path):
    path = bundle(tmp_path)
    manifest = verify_bundle(path)
    source = tmp_path / "source.bin"
    outside = tmp_path / "outside-owner.json"
    outside.write_bytes((path / ".bundle-owner.json").read_bytes())
    (path / ".bundle-owner.json").unlink()
    (path / ".bundle-owner.json").symlink_to(outside)
    original = outside.read_bytes()
    with pytest.raises(ValueError, match="symlink"):
        create_bundle(manifest["identity"], {"models/model.bin": source}, path)
    assert outside.read_bytes() == original


@pytest.mark.parametrize("name", [".", "folder//model.bin", "./model.bin"])
def test_noncanonical_member_names_are_rejected(tmp_path, name):
    source = tmp_path / "source.bin"
    source.write_bytes(b"source")
    with pytest.raises(ValueError):
        create_bundle(
            {"run_id": "fixture", "model_sha256": "weights", "result_label": "FIXTURE_NOT_A_MODEL_RESULT"},
            {name: source},
            tmp_path / "unsafe",
        )


def test_packaging_refuses_to_overwrite_unowned_data(tmp_path):
    destination = tmp_path / "unowned"
    existing = destination / "models/model.bin"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"operator data")
    source = tmp_path / "source.bin"
    source.write_bytes(b"replacement")
    with pytest.raises(DeliveryError, match="unowned"):
        create_bundle(
            {"run_id": "fixture", "model_sha256": "weights", "result_label": "FIXTURE_NOT_A_MODEL_RESULT"},
            {"models/model.bin": source},
            destination,
        )
    assert existing.read_bytes() == b"operator data"


def test_changing_local_state_cannot_replace_destination_identity(tmp_path):
    original = bundle(tmp_path)
    identity = verify_bundle(original)["identity"]
    other_source = tmp_path / "other.bin"
    other_source.write_bytes(b"different model artifact")
    conflicting = create_bundle(identity, {"models/model.bin": other_source}, tmp_path / "other-bundle")
    before = deliver(original, tmp_path / "destination", tmp_path / "state-a")
    try:
        deliver(conflicting, tmp_path / "destination", tmp_path / "state-b")
    except DeliveryError:
        pass
    with pytest.raises(DeliveryError):
        deliver(conflicting, tmp_path / "destination", tmp_path / "state-b", recover=True)
    assert (
        deliver(original, tmp_path / "destination", tmp_path / "state-a")["manifest_sha256"]
        == before["manifest_sha256"]
    )


def test_canonical_platform_contract_is_explicit_and_pinned():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    contract = json.loads((root / "configs/platform-contract.json").read_text())
    assert contract == {"experiment_name": "kaggle/alaska2", "artifact_namespace": "kaggle/alaska2"}
