"""Fresh host-occupancy gate, run immediately before every expensive launch (CUDA smoke, baseline).

Checks GPU, CPU, IO and disk against fixed thresholds and exits 0 only on GO. On NO-GO it exits 3
and names each failing criterion. ``--wait-max-s`` polls (waits) instead of competing; it never
stops, renices or otherwise touches other processes or containers. Nothing identifying is
recorded: no PIDs, process names, env vars, usernames or hostnames.

    uv run python -m tools.readiness.launch_gate --data-dir ~/data/kaggle/alaska2-image-steganalysis \
        --label smoke --out evidence/phase1/launch_gate.json [--wait-max-s 3600 --poll-s 120]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Thresholds:
    # GPU: our job is not running yet, so everything in use belongs to someone else.
    gpu_other_used_mib_max: int = 2048
    gpu_util_pct_max: int = 20
    # CPU: the job needs ~14 threads (12 loader workers + main + pin thread); keep >= 16 idle.
    cpu_free_threads_min: float = 16.0
    cpu_psi_some_avg60_max: float = 10.0
    # IO: the loader streams ~32 GB per epoch until the page cache is warm.
    io_psi_some_avg60_max: float = 20.0
    io_psi_full_avg60_max: float = 10.0
    # Disk: operator floor 40 GB on / plus the run's own expected outputs.
    disk_floor_gib: float = 40.0
    run_output_gib: float = 5.0


def parse_psi(text: str | None) -> dict[str, float]:
    """``/proc/pressure/*`` text -> {"some_avg60": x, "full_avg60": y} (missing lines omitted)."""
    out: dict[str, float] = {}
    for line in (text or "").splitlines():
        parts = line.split()
        if not parts:
            continue
        for kv in parts[1:]:
            k, _, v = kv.partition("=")
            if k == "avg60":
                out[f"{parts[0]}_avg60"] = float(v)
    return out


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text()
    except OSError:
        return None


def gpu_reading() -> list[dict] | None:
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if p.returncode != 0:
        return None
    devs = []
    for line in p.stdout.strip().splitlines():
        idx, used, util = (x.strip() for x in line.split(","))
        devs.append({"index": int(idx), "memory_used_mib": int(used), "utilization_gpu_pct": int(util)})
    return devs


def reading(data_dir: Path, gpu_index: int) -> dict:
    devs = gpu_reading()
    gpu = next((d for d in devs or [] if d["index"] == gpu_index), None)
    return {
        "gpu": gpu,
        "logical_threads": os.cpu_count(),
        "loadavg_1": round(os.getloadavg()[0], 2),
        "cpu_psi": parse_psi(_read("/proc/pressure/cpu")),
        "io_psi": parse_psi(_read("/proc/pressure/io")),
        "free_gib": {
            "root_fs": round(shutil.disk_usage("/").free / 2**30, 1),
            "data_dir_fs": round(shutil.disk_usage(data_dir).free / 2**30, 1),
        },
    }


def evaluate(r: dict, t: Thresholds) -> dict:
    """Pure decision: every criterion with its value, limit and pass flag; GO only if all pass."""
    checks: dict[str, dict] = {}

    def check(name: str, value, limit, ok: bool) -> None:
        checks[name] = {"value": value, "limit": limit, "pass": bool(ok)}

    g = r.get("gpu")
    if g is None:
        check("gpu_visible", None, "nvidia-smi ok", False)
    else:
        used, util = g["memory_used_mib"], g["utilization_gpu_pct"]
        check("gpu_other_used_mib", used, f"<= {t.gpu_other_used_mib_max}", used <= t.gpu_other_used_mib_max)
        check("gpu_util_pct", util, f"<= {t.gpu_util_pct_max}", util <= t.gpu_util_pct_max)
    free_threads = round(r["logical_threads"] - r["loadavg_1"], 2)
    check(
        "cpu_free_threads",
        free_threads,
        f">= {t.cpu_free_threads_min}",
        free_threads >= t.cpu_free_threads_min,
    )
    for name, src, key, lim in (
        ("cpu_psi_some_avg60", "cpu_psi", "some_avg60", t.cpu_psi_some_avg60_max),
        ("io_psi_some_avg60", "io_psi", "some_avg60", t.io_psi_some_avg60_max),
        ("io_psi_full_avg60", "io_psi", "full_avg60", t.io_psi_full_avg60_max),
    ):
        v = r[src].get(key)
        # PSI unavailable is treated as a failure: occupancy cannot be shown to be low.
        check(name, v, f"<= {lim}", v is not None and v <= lim)
    need = t.disk_floor_gib + t.run_output_gib
    for fs, free in r["free_gib"].items():
        check(f"disk_free_gib_{fs}", free, f">= {need}", free >= need)
    failing = sorted(k for k, c in checks.items() if not c["pass"])
    return {"decision": "GO" if not failing else "NO-GO", "failing": failing, "checks": checks}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--label", required=True, help="which launch this gate is for, e.g. smoke / baseline")
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--run-output-gib", type=float, default=Thresholds.run_output_gib)
    ap.add_argument("--wait-max-s", type=int, default=0, help="poll until GO for up to this long (0 = once)")
    ap.add_argument("--poll-s", type=int, default=120)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    t = Thresholds(run_output_gib=a.run_output_gib)
    t0 = time.monotonic()
    attempts = []
    while True:
        r = reading(a.data_dir, a.gpu_index)
        res = evaluate(r, t)
        attempts.append(
            {
                "at_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "decision": res["decision"],
                "failing": res["failing"],
            }
        )
        if res["decision"] == "GO" or time.monotonic() - t0 + a.poll_s > a.wait_max_s:
            break
        time.sleep(a.poll_s)
    out = {
        "gate": "launch_occupancy_gate",
        "label": a.label,
        "thresholds": asdict(t),
        "final": res,
        "attempts": attempts,
        "waited_s": round(time.monotonic() - t0, 1),
        "redaction": "aggregate host readings only; no PIDs, process names, env vars or identities",
    }
    text = json.dumps(out, indent=2, sort_keys=True)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text + "\n")
    print(text)
    sys.exit(0 if res["decision"] == "GO" else 3)


if __name__ == "__main__":
    main()
