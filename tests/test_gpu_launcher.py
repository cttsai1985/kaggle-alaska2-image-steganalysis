"""The GPU launcher must reserve shared memory and keep source/data mounts read-only."""

import json
import os
import subprocess
from pathlib import Path


def test_gpu_launcher_resource_and_mount_contract(tmp_path):
    runtime, data, output = (tmp_path / name for name in ("runtime", "data", "output"))
    python = runtime / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o755)
    data.mkdir()
    capture = tmp_path / "args.json"
    docker = tmp_path / "docker"
    docker.write_text(
        "#!/usr/bin/env python3\nimport json,os,sys\n"
        "open(os.environ['CAPTURE'], 'w').write(json.dumps(sys.argv[1:]))\n"
    )
    docker.chmod(0o755)
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        ["bash", str(root / "tools/runs/retained_gpu.sh"), str(runtime), str(data), str(output), "retained"],
        check=True,
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "CAPTURE": str(capture)},
    )
    args = json.loads(capture.read_text())
    assert args[args.index("--shm-size") + 1] == "2g"
    assert args[args.index("--cpus") + 1] == "6"
    assert args[args.index("--network") + 1] == "none"
    mounts = [args[i + 1] for i, item in enumerate(args) if item == "--mount"]
    assert any(f"source={data},target=/inputs,readonly" in mount for mount in mounts)
    assert any(f"source={output},target=/output" in mount and "readonly" not in mount for mount in mounts)
