"""Verify the running Lab's HTTP auth, actual mounts, write behavior and checkout identity."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


def _docker(*args: str) -> str:
    return subprocess.check_output(["docker", *args], stderr=subprocess.STDOUT, text=True).strip()


def verify_lab(root: Path, container: str = "alaska2-lab-1") -> dict:
    token = (root / ".local/jupyter_token").read_text().strip()
    url = "http://127.0.0.1:8894/api/status"
    deadline = time.monotonic() + 30
    while True:
        try:
            urllib.request.urlopen(url, timeout=2)
            raise ValueError("Lab accepted an unauthenticated API request")
        except urllib.error.HTTPError as error:
            rejected = error.code
            if rejected not in (401, 403):
                raise
            break
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.2)
    request = urllib.request.Request(url, headers={"Authorization": "token " + token})
    with urllib.request.urlopen(request, timeout=10) as response:
        authenticated = response.status
        json.load(response)
    info = json.loads(_docker("inspect", container))[0]
    ports = info["NetworkSettings"]["Ports"]["8888/tcp"]
    mounts = {mount["Destination"]: mount for mount in info["Mounts"]}
    if (
        not ports
        or any(port["HostIp"] != "127.0.0.1" for port in ports)
        or mounts["/data"]["RW"]
        or not mounts["/outputs"]["RW"]
        or mounts["/run/secrets/alaska2_lab_token"]["RW"]
    ):
        raise ValueError("running Lab violates the isolation contract")
    probe = subprocess.run(
        [
            "docker",
            "exec",
            container,
            "python",
            "-c",
            'from pathlib import Path; Path("/data/.write-probe").write_text("x")',
        ],
        capture_output=True,
        text=True,
    )
    if not probe.returncode or "Read-only file system" not in probe.stderr:
        raise ValueError("read-only input write was not rejected")
    _docker(
        "exec",
        container,
        "python",
        "-c",
        "from pathlib import Path; "
        'p=Path("/outputs/.write-probe"); p.write_text("x"); assert p.read_text()=="x"; p.unlink()',
    )
    source = json.loads(
        _docker(
            "exec",
            container,
            "python",
            "-c",
            "import json; from alaska2.identity import git_source; print(json.dumps(git_source()))",
        )
    )
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if source["git_sha"] != head:
        raise ValueError("Lab source differs from the prepared checkout")
    return {
        "kind": "lab_validation",
        "measurement_status": "MEASURED",
        "unauthenticated_http": rejected,
        "authenticated_http": authenticated,
        "loopback_only": True,
        "data_read_only_actual_write_rejected": True,
        "outputs_write_readback": True,
        "token_read_only": True,
        "token_printed": False,
        "running_uid": _docker("exec", container, "id", "-u"),
        "image_id": info["Image"],
        "source": source,
        "devcontainer_ide_attach": "unverified",
        "hosted_kaggle_execution": "unverified",
        "gpu_requested": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = verify_lab(Path(__file__).resolve().parents[1])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {key: result[key] for key in ("unauthenticated_http", "authenticated_http", "loopback_only")}
        )
    )


if __name__ == "__main__":
    main()
