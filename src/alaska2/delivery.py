"""Offline immutable run bundles and explicit, byte-verified filesystem transfer recovery.

The filesystem transport is also the fault-drill backend. It does not claim an MLflow/S3 service transfer.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import shutil
import sqlite3
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

from alaska2.identity import sha256_file


class DeliveryError(RuntimeError):
    pass


class RecoveryRequired(DeliveryError):
    pass


def _text(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"


def _digest(value: dict) -> str:
    return hashlib.sha256(_text(value).encode()).hexdigest()


def _member(root: Path, name: str) -> Path:
    rel = PurePosixPath(name)
    if (
        not rel.parts
        or rel.as_posix() != name
        or rel.is_absolute()
        or ".." in rel.parts
        or "\\" in name
        or ":" in name
    ):
        raise ValueError("bundle member must be a safe relative path")
    path = root / name
    for parent in [path, *path.parents]:
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError("bundle member cannot traverse a symlink")
    return path


@contextmanager
def artifact_claim(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("claim file cannot be a symlink")
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise DeliveryError(
                "an active importer owns this experiment; recovery cannot take it over"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _verify_members(directory: Path, manifest: dict) -> None:
    for name, expected in manifest["files"].items():
        path = _member(directory, name)
        if (
            not path.is_file()
            or path.stat().st_size != expected["bytes"]
            or sha256_file(path) != expected["sha256"]
        ):
            raise DeliveryError(f"bundle member integrity failure: {name}")


def verify_bundle(directory: Path) -> dict:
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["project_id"] != "alaska2" or manifest["experiment_id"] != "alaska2-" + _digest(
        manifest["identity"]
    ):
        raise DeliveryError("bundle scientific identity differs")
    if not manifest["files"]:
        raise DeliveryError("empty artifact bundle")
    _verify_members(directory, manifest)
    return manifest


def create_bundle(identity: dict, files: dict[str, Path], destination: Path) -> Path:
    """Resume packaging only when the planned manifest agrees, preserving scientific identity."""
    destination = Path(destination)
    if not identity.get("run_id") or not identity.get("model_sha256") or not identity.get("result_label"):
        raise ValueError("bundle identity requires run_id, model_sha256 and result_label")
    members = {}
    for name, source in sorted(files.items()):
        _member(destination, name)
        if name == "manifest.json" or name.startswith((".bundle", ".manifest")):
            raise ValueError("artifact name collides with bundle control files")
        source = Path(source)
        if source.is_symlink() or not source.is_file():
            raise ValueError("artifact source must be a regular file")
        members[name] = {"sha256": sha256_file(source), "bytes": source.stat().st_size}
    if not members:
        raise ValueError("cannot package an empty artifact set")
    manifest = {
        "schema_version": 1,
        "project_id": "alaska2",
        "experiment_id": "alaska2-" + _digest(identity),
        "identity": identity,
        "files": members,
    }
    with artifact_claim(destination / ".bundle.lock"):
        owner = _member(destination, ".bundle-owner.json")
        if not owner.exists() and any(
            item.name not in {".bundle.lock", ".bundle-owner.partial"} for item in destination.iterdir()
        ):
            raise DeliveryError("refusing to overwrite an unowned nonempty bundle directory")
        if owner.exists() and json.loads(owner.read_text()) != {"manifest_sha256": _digest(manifest)}:
            raise DeliveryError("conflicting bundle manifest; keep the original experiment")
        owner_pending = _member(destination, ".bundle-owner.partial")
        owner_pending.write_text(_text({"manifest_sha256": _digest(manifest)}))
        owner_pending.replace(owner)
        for name, source in files.items():
            target = _member(destination, name)
            if target.is_file() and sha256_file(target) == members[name]["sha256"]:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".partial")
            if partial.is_symlink():
                raise ValueError("partial artifact cannot be a symlink")
            shutil.copyfile(source, partial)
            if sha256_file(partial) != members[name]["sha256"]:
                raise DeliveryError("source artifact changed while packaging")
            partial.replace(target)
        _verify_members(destination, manifest)
        temporary = _member(destination, ".manifest.partial")
        temporary.write_text(_text(manifest))
        temporary.replace(destination / "manifest.json")
    verify_bundle(destination)
    return destination


def _record(state: Path, result: dict) -> dict:
    state.mkdir(parents=True, exist_ok=True)
    journal = _member(state, result["experiment_id"] + ".events.jsonl")
    with journal.open("a", encoding="utf-8") as stream:
        stream.write(_text(result))
    return result


def _published(target: Path, manifest: dict, manifest_sha: str) -> dict:
    try:
        pointer = json.loads(_member(target, "current.json").read_text())
        directory = _member(target, pointer["directory"])
        if (
            pointer["manifest_sha256"] != manifest_sha
            or sha256_file(directory / "manifest.json") != manifest_sha
        ):
            raise DeliveryError("destination manifest differs")
        _verify_members(directory, manifest)
        return pointer
    except (OSError, KeyError, ValueError) as exc:
        raise DeliveryError("destination is incomplete or corrupt; explicit recovery required") from exc


def deliver(
    bundle: Path,
    destination: Path,
    state: Path,
    *,
    recover: bool = False,
    fault: str | None = None,
    quality_accepted: bool = True,
) -> dict:
    """Transfer under an OS-owned claim; crashed/failed attempts require explicit, same-identity recovery."""
    if fault not in {None, "after_copy", "corrupt_copy", "after_publish"}:
        raise ValueError("unknown transfer fault")
    bundle, destination, state = Path(bundle), Path(destination), Path(state)
    manifest = verify_bundle(bundle)
    experiment = manifest["experiment_id"]
    result = {"experiment_id": experiment, "transport": "filesystem", "manifest_sha256": _digest(manifest)}
    _record(state, {**result, "state": "requested", "recover": recover, "fault": fault})
    if not quality_accepted:
        return _record(
            state,
            {
                **result,
                "state": "quality_rejected",
                "evaluation_retained": True,
                "next_action": "model/policy decision; not a dependency retry",
            },
        )
    target = _member(destination, experiment)
    with artifact_claim(target / ".delivery.lock"):
        database = _member(target, ".delivery.sqlite3")
        if not database.exists() and any(item.name != ".delivery.lock" for item in target.iterdir()):
            raise DeliveryError("unowned destination has no authoritative identity ledger")
        db = sqlite3.connect(database)
        try:
            db.execute(
                "CREATE TABLE IF NOT EXISTS experiment (id TEXT PRIMARY KEY, "
                "manifest TEXT, target TEXT, status TEXT, attempt INTEGER)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS attempts (id INTEGER PRIMARY KEY, status TEXT, error TEXT)"
            )
            target_hash = hashlib.sha256(str(destination.resolve()).encode()).hexdigest()
            row = db.execute(
                "SELECT manifest,target,status,attempt FROM experiment WHERE id=?", (experiment,)
            ).fetchone()
            if row and (row[0] != result["manifest_sha256"] or row[1] != target_hash):
                raise DeliveryError("conflicting manifest or destination for the same logical experiment")
            if row and row[2] == "delivered":
                try:
                    _published(target, manifest, result["manifest_sha256"])
                except DeliveryError:
                    if not recover:
                        raise
                else:
                    return _record(
                        state,
                        {
                            **result,
                            "state": "delivered",
                            "attempt": row[3],
                            "reused": True,
                            "verified_files": len(manifest["files"]),
                        },
                    )
            if row and not recover:
                raise RecoveryRequired("previous importer stopped incomplete; explicit recovery required")
            attempt = db.execute("INSERT INTO attempts(status) VALUES ('running')").lastrowid
            db.execute(
                "INSERT OR REPLACE INTO experiment VALUES (?,?,?,?,?)",
                (experiment, result["manifest_sha256"], target_hash, "running", attempt),
            )
            db.commit()
            directory = _member(target, f"attempt-{attempt}")
            try:
                directory.mkdir(parents=True, exist_ok=False)
                for index, name in enumerate(manifest["files"]):
                    output = _member(directory, name)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(_member(bundle, name), output)
                    if index == 0 and fault == "after_copy":
                        raise DeliveryError("FAULT_DRILL: transfer interrupted after first copy")
                    if index == 0 and fault == "corrupt_copy":
                        output.write_bytes(b"FAULT_DRILL_CORRUPTED_TRANSFER")
                (directory / "manifest.json").write_text(_text(manifest))
                _verify_members(directory, manifest)
                pointer = {"directory": directory.name, "manifest_sha256": result["manifest_sha256"]}
                pending = _member(target, f"current-{attempt}.partial")
                pending.write_text(_text(pointer))
                pending.replace(target / "current.json")
                if fault == "after_publish":
                    raise DeliveryError("FAULT_DRILL: stopped after publishing before completion receipt")
                _published(target, manifest, result["manifest_sha256"])
                db.execute("UPDATE attempts SET status='delivered' WHERE id=?", (attempt,))
                db.execute("UPDATE experiment SET status='delivered' WHERE id=?", (experiment,))
                db.commit()
                return _record(
                    state,
                    {
                        **result,
                        "state": "delivered",
                        "attempt": attempt,
                        "reused": False,
                        "verified_files": len(manifest["files"]),
                    },
                )
            except Exception as exc:
                db.execute(
                    "UPDATE attempts SET status='failed',error=? WHERE id=?", (type(exc).__name__, attempt)
                )
                db.execute("UPDATE experiment SET status='failed' WHERE id=?", (experiment,))
                db.commit()
                _record(state, {**result, "state": "failed", "attempt": attempt, "error": type(exc).__name__})
                raise DeliveryError(
                    f"transfer incomplete: {type(exc).__name__}; local bundle retained"
                ) from exc
        finally:
            db.close()
