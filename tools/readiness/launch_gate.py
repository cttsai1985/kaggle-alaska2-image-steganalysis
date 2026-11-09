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
    # A warm page cache makes host IO pressure irrelevant to our reads: if random image files read at
    # >= this rate (the loader needs a few hundred per second), IO PSI is recorded as advisory only.
    io_probe_min_files_per_s: float = 2000.0
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


def _cpu_times() -> tuple[int, int] | None:
    """(idle + iowait, total) jiffies from the aggregate ``cpu`` line of /proc/stat."""
    text = _read("/proc/stat")
    if not text:
        return None
    vals = [int(v) for v in text.splitlines()[0].split()[1:]]
    return vals[3] + vals[4], sum(vals[:8])  # user nice system idle iowait irq softirq steal


def cpu_idle_fraction(window_s: float) -> float | None:
    """Measured idle share of all CPUs over a window. loadavg also counts tasks blocked on IO (D state),
    so on an IO-heavy host it overstates CPU use; the idle share is what our workers can actually get."""
    a = _cpu_times()
    if a is None or window_s <= 0:
        return None
    time.sleep(window_s)
    b = _cpu_times()
    if b is None or b[1] == a[1]:
        return None
    return round((b[0] - a[0]) / (b[1] - a[1]), 4)


def read_probe(images_dir: Path, n: int) -> dict | None:
    """Read n random image files once (whole file); files/s shows whether reads hit the page cache."""
    import random

    if n <= 0 or not images_dir.is_dir():
        return None
    rng = random.Random()  # fresh draw each time: re-probing the same files would always look cached
    files = []
    for kind in sorted(p for p in images_dir.iterdir() if p.is_dir()):
        names = os.listdir(kind)
        files += [kind / x for x in rng.sample(names, min(len(names), max(1, n // 5)))]
    t0, nbytes = time.perf_counter(), 0
    for f in files:
        nbytes += len(f.read_bytes())
    dt = max(time.perf_counter() - t0, 1e-9)
    return {
        "files": len(files),
        "files_per_s": round(len(files) / dt, 1),
        "mib_per_s": round(nbytes / 2**20 / dt, 1),
    }


def reading(
    data_dir: Path, gpu_index: int, cpu_window_s: float = 0.0, probe_files: int = 0, images_subdir: str = ""
) -> dict:
    devs = gpu_reading()
    gpu = next((d for d in devs or [] if d["index"] == gpu_index), None)
    return {
        "gpu": gpu,
        "logical_threads": os.cpu_count(),
        "loadavg_1": round(os.getloadavg()[0], 2),
        "cpu_idle_fraction": cpu_idle_fraction(cpu_window_s),
        "cpu_window_s": cpu_window_s,
        "read_probe": read_probe(data_dir / images_subdir, probe_files) if images_subdir else None,
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
    idle = r.get("cpu_idle_fraction")
    if idle is not None:  # measured idle CPUs; loadavg is kept for the record only
        free_threads = round(r["logical_threads"] * idle, 2)
        basis = f"logical_threads x measured idle fraction over {r.get('cpu_window_s')} s"
    else:
        free_threads = round(r["logical_threads"] - r["loadavg_1"], 2)
        basis = "logical_threads - loadavg_1"
    check(
        "cpu_free_threads",
        free_threads,
        f">= {t.cpu_free_threads_min}",
        free_threads >= t.cpu_free_threads_min,
    )
    checks["cpu_free_threads"]["basis"] = basis
    probe = r.get("read_probe")
    warm = probe is not None and probe["files_per_s"] >= t.io_probe_min_files_per_s
    if probe is not None:
        check("io_read_probe_files_per_s", probe["files_per_s"], f">= {t.io_probe_min_files_per_s}", warm)
        checks["io_read_probe_files_per_s"]["advisory"] = (
            True  # a slow probe only re-enables the IO PSI checks
        )
    for name, src, key, lim in (
        ("cpu_psi_some_avg60", "cpu_psi", "some_avg60", t.cpu_psi_some_avg60_max),
        ("io_psi_some_avg60", "io_psi", "some_avg60", t.io_psi_some_avg60_max),
        ("io_psi_full_avg60", "io_psi", "full_avg60", t.io_psi_full_avg60_max),
    ):
        v = r[src].get(key)
        # PSI unavailable is treated as a failure: occupancy cannot be shown to be low.
        check(name, v, f"<= {lim}", v is not None and v <= lim)
        if src == "io_psi" and warm:
            checks[name]["advisory"] = True  # our reads come from the page cache (probe above)
    need = t.disk_floor_gib + t.run_output_gib
    for fs, free in r["free_gib"].items():
        check(f"disk_free_gib_{fs}", free, f">= {need}", free >= need)
    failing = sorted(k for k, c in checks.items() if not c["pass"] and not c.get("advisory"))
    advisory_failing = sorted(k for k, c in checks.items() if not c["pass"] and c.get("advisory"))
    return {
        "decision": "GO" if not failing else "NO-GO",
        "failing": failing,
        "advisory_failing": advisory_failing,
        "checks": checks,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--label", required=True, help="which launch this gate is for, e.g. smoke / baseline")
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--run-output-gib", type=float, default=Thresholds.run_output_gib)
    ap.add_argument(
        "--cpu-free-threads-min",
        type=float,
        default=Thresholds.cpu_free_threads_min,
        help="size to the job: loader workers + main + pin thread + headroom (default 16 = 12 workers)",
    )
    ap.add_argument("--cpu-window-s", type=float, default=5.0, help="idle-CPU sampling window (0 = loadavg)")
    ap.add_argument("--read-probe-files", type=int, default=0, help="random image files to read (0 = off)")
    ap.add_argument("--images-subdir", default="raw/extracted")
    ap.add_argument("--wait-max-s", type=int, default=0, help="poll until GO for up to this long (0 = once)")
    ap.add_argument("--poll-s", type=int, default=120)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    t = Thresholds(run_output_gib=a.run_output_gib, cpu_free_threads_min=a.cpu_free_threads_min)
    t0 = time.monotonic()
    attempts = []
    while True:
        r = reading(
            a.data_dir,
            a.gpu_index,
            a.cpu_window_s,
            a.read_probe_files,
            a.images_subdir if a.read_probe_files else "",
        )
        res = evaluate(r, t)
        attempts.append(
            {
                "at_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "decision": res["decision"],
                "failing": res["failing"],
                "advisory_failing": res["advisory_failing"],
                "loadavg_1": r["loadavg_1"],
                "cpu_idle_fraction": r["cpu_idle_fraction"],
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
        "final_reading": r,
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
