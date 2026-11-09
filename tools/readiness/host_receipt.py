"""Redacted P620 host readiness receipt.

Records hardware, GPU occupancy, toolchain versions, Docker GPU visibility and
Kaggle access as booleans. It never reads or prints environment variables,
tokens, usernames or home paths; command output is parsed into named fields and
raw output is discarded.

    uv run python -m tools.readiness.host_receipt --data-dir ~/data/kaggle/alaska2-image-steganalysis \
        --out evidence/phase1/host_receipt.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

COMPETITION = "alaska2-image-steganalysis"
QUALITY_DATASET = "cttsai/alaska2-image-steganalysis-image-quality"
DOCKER_GPU_IMAGE = "nvidia/cuda:12.4.1-base-ubuntu22.04"  # already present locally; no pull


def run(cmd: list[str], timeout: int = 120) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return p.returncode, p.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return -1, type(exc).__name__


def first_line(cmd: list[str]) -> str | None:
    rc, out = run(cmd)
    return out.splitlines()[0].strip() if rc == 0 and out else None


def os_release() -> dict:
    info = {}
    for line in Path("/etc/os-release").read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            info[k] = v.strip('"')
    return {"pretty_name": info.get("PRETTY_NAME"), "version_id": info.get("VERSION_ID")}


def cpu() -> dict:
    model = None
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            model = line.split(":", 1)[1].strip()
            break
    return {"model": model, "logical_threads_visible": os.cpu_count()}


def mem_gib() -> dict:
    vals = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, v = line.split(":", 1)
        if k in ("MemTotal", "MemAvailable"):
            vals[k] = round(int(v.split()[0]) / 2**20, 1)
    return {"total_gib": vals.get("MemTotal"), "available_gib": vals.get("MemAvailable")}


def _pressure(kind: str) -> str | None:
    try:
        return Path(f"/proc/pressure/{kind}").read_text().splitlines()[0]
    except OSError:
        return None


def disk(path: Path) -> dict:
    u = shutil.disk_usage(path)
    return {"total_gib": round(u.total / 2**30, 1), "free_gib": round(u.free / 2**30, 1)}


def gpus() -> dict:
    fields = "index,name,memory.total,memory.used,utilization.gpu,driver_version"
    rc, out = run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
    devices = []
    if rc == 0:
        for line in out.splitlines():
            idx, name, tot, used, util, drv = (x.strip() for x in line.split(","))
            devices.append(
                {
                    "index": int(idx),
                    "name": name,
                    "memory_total_mib": int(tot),
                    "memory_used_mib": int(used),
                    "memory_free_mib": int(tot) - int(used),
                    "utilization_gpu_pct": int(util),
                    "driver_version": drv,
                }
            )
    rc2, header = run(["nvidia-smi"])
    cuda = None
    if rc2 == 0 and "CUDA Version:" in header:
        cuda = header.split("CUDA Version:", 1)[1].split()[0]
    rc3, apps = run(["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"])
    # Per-process memory only; PIDs and process names are not recorded.
    app_mem = [int(x) for x in apps.splitlines() if x.strip().isdigit()] if rc3 == 0 else []
    return {
        "nvidia_smi_ok": rc == 0,
        "cuda_driver_api_version": cuda,
        "devices": devices,
        "compute_processes_visible": len(app_mem),
        "compute_process_memory_mib": sorted(app_mem, reverse=True),
        "note": "WSL2 often hides per-process GPU attribution; memory.used is the authoritative occupancy",
    }


def docker_gpu(timeout: int = 300) -> dict:
    """``docker run --gpus all`` with a named, self-owned container; partial output is kept on timeout."""
    ver = first_line(["docker", "--version"])
    name = f"alaska2-gpucheck-{int(time.time())}"
    cmd = ["docker", "run", "--rm", "--name", name, "--gpus", "all", DOCKER_GPU_IMAGE, "nvidia-smi", "-L"]
    t0 = time.monotonic()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        rc, out, status = p.returncode, p.stdout, "exited"
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        rc, status = None, "timeout"
        # Only the container this function created (unique name) is removed.
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=120, check=False)
    elapsed = round(time.monotonic() - t0, 1)
    names = [ln.split("(UUID")[0].strip() for ln in out.splitlines() if ln.startswith("GPU ")]
    return {
        "docker_version": ver,
        "gpus_all_check_image": DOCKER_GPU_IMAGE,
        "exit_status": status,
        "exit_code": rc,
        "elapsed_s": elapsed,
        "gpu_visible_in_container": bool(names),
        "gpus_all_ok": rc == 0 and bool(names),
        "gpus": names,
    }


def _listing(out: str) -> dict:
    """Parse a ``--csv`` file listing page: data rows only; the page token itself is not recorded."""
    lines = [ln for ln in out.splitlines() if ln.strip()]
    paginated = any(ln.startswith("Next Page Token") for ln in lines)
    rows = [ln for ln in lines if not ln.startswith("Next Page Token") and not ln.startswith("name,")]
    return {"first_page_rows": len(rows), "more_pages": paginated}


def kaggle_access() -> dict:
    ver = first_line(["kaggle", "--version"])
    rc_c, out_c = run(["kaggle", "competitions", "files", COMPETITION, "--csv"])
    rc_d, out_d = run(["kaggle", "datasets", "files", QUALITY_DATASET, "--csv"])
    # Only success booleans and first-page shape are kept; no account identity or page tokens.
    return {
        "kaggle_cli_version": ver,
        "competition": COMPETITION,
        "competition_files_list_ok": rc_c == 0,
        "competition_files_listing": _listing(out_c) if rc_c == 0 else None,
        "quality_dataset": QUALITY_DATASET,
        "quality_dataset_files_list_ok": rc_d == 0,
        "quality_dataset_files_listing": _listing(out_d) if rc_d == 0 else None,
        "note": "file listing is paginated (first page only) and proves access, not completeness; "
        "download completeness is verified in download_receipt/dataset_audit",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--skip-docker", action="store_true")
    ap.add_argument("--skip-kaggle", action="store_true")
    a = ap.parse_args()

    receipt = {
        "receipt": "phase1_host_readiness",
        "captured_at_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host_label": "P620-WSL2",
        "os": {
            **os_release(),
            "kernel": platform.release(),
            "wsl2": "microsoft" in platform.release().lower(),
        },
        "cpu": cpu(),
        "memory": mem_gib(),
        "load": {
            "loadavg_1_5_15": [round(x, 1) for x in os.getloadavg()],
            "io_pressure": _pressure("io"),
            "cpu_pressure": _pressure("cpu"),
            "note": "host is shared; these are point-in-time occupancy readings",
        },
        "disk": {"root_fs": disk(Path("/")), "data_dir_fs": disk(a.data_dir)},
        "gpu": gpus(),
        "toolchain": {
            "python": platform.python_version(),
            "uv": first_line(["uv", "--version"]),
        },
        "docker": {"skipped": True} if a.skip_docker else docker_gpu(),
        "kaggle": {"skipped": True} if a.skip_kaggle else kaggle_access(),
        "redaction": "no env vars, tokens, usernames, hostnames, PIDs or absolute home paths recorded",
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
