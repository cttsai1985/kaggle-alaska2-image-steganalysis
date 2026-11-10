"""Prepare and operate the isolated ALASKA2 Notebook Lab without printing its token."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import secrets
import subprocess
import tempfile
import tomllib
from pathlib import Path


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise ValueError("Lab preparation needs the source Git checkout")
    return result.stdout.strip()


def prepare(root: Path, data_dir: Path | None = None, output_dir: Path | None = None) -> dict:
    """Reuse a nonempty private token; materialize explicit paths, source revision and loopback binding."""
    root = root.resolve()
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    if project["name"] != "alaska2":
        raise ValueError("expected the alaska2 source checkout")
    output = (output_dir or root.parent / (root.name + "-outputs")).resolve()
    data = (data_dir or root.parent / (root.name + "-empty-data")).resolve()
    if (
        output.is_relative_to(root)
        or root.is_relative_to(output)
        or output.is_relative_to(data)
        or data.is_relative_to(output)
    ):
        raise ValueError("outputs must not overlap source or read-only data")
    if data_dir is not None and (
        not data.is_dir() or data.is_relative_to(root) or data.is_relative_to(output)
    ):
        raise ValueError("supplied data must exist outside source and writable outputs")
    if data_dir is None:
        data.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    local = root / ".local"
    local.mkdir(exist_ok=True)
    token = local / "jupyter_token"
    with (local / "prepare.lock").open("a+") as claim:
        fcntl.flock(claim, fcntl.LOCK_EX)
        if token.is_symlink():
            raise ValueError("Lab token cannot be a symlink")
        if not token.exists() or len(token.read_text().strip()) < 32:
            with tempfile.NamedTemporaryFile(mode="w", dir=local, delete=False) as pending:
                pending.write(secrets.token_urlsafe(32) + "\n")
                temporary = Path(pending.name)
            temporary.replace(token)
        token.chmod(0o600)
    common = Path(_git(root, "rev-parse", "--git-common-dir"))
    if not common.is_absolute():
        common = (root / common).resolve()
    source = _git(root, "rev-parse", "HEAD")
    dirty = bool(_git(root, "status", "--porcelain"))
    mounts = [
        {"type": "bind", "source": str(root), "target": "/workspace"},
        {"type": "bind", "source": str(data), "target": "/data", "read_only": True},
        {"type": "bind", "source": str(output), "target": "/outputs"},
        {"type": "bind", "source": str(token), "target": "/run/secrets/alaska2_lab_token", "read_only": True},
        {"type": "bind", "source": str(common), "target": str(common), "read_only": True},
    ]
    override = {
        "services": {
            "lab": {
                "image": "alaska2/lab:" + project["version"],
                "user": f"{os.getuid()}:{os.getgid()}",
                "ports": ["127.0.0.1:8894:8888"],
                "build": {
                    "args": {"ALASKA2_SOURCE_GIT_SHA": source, "ALASKA2_SOURCE_DIRTY": str(int(dirty))}
                },
                "volumes": mounts,
            }
        }
    }
    compose = local / "compose.lab.yml"
    compose.write_text(json.dumps(override, indent=2) + "\n")
    return {
        "compose_file": str(compose),
        "token_file": str(token),
        "output_dir": str(output),
        "url": "http://127.0.0.1:8894/lab",
        "image": "alaska2/lab:" + project["version"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "build", "up", "down", "config"])
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu", action="store_true", help="explicitly opt in to the GPU override")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    info = prepare(root, args.data_dir, args.output_dir)
    if args.action == "prepare":
        print(json.dumps(info, indent=2))
        return
    command = [
        "docker",
        "compose",
        "--project-name",
        "alaska2",
        "-f",
        str(root / "compose.yml"),
        "-f",
        info["compose_file"],
    ]
    if args.gpu:
        command += ["-f", str(root / "compose.gpu.yml")]
    command += {
        "build": ["build"],
        "up": ["up", "--build", "--detach"],
        "down": ["down"],
        "config": ["config", "--format", "json"],
    }[args.action]
    subprocess.run(command, cwd=root, check=True)
    if args.action == "up":
        print(json.dumps({"url": info["url"], "token_file": info["token_file"]}))


if __name__ == "__main__":
    main()
