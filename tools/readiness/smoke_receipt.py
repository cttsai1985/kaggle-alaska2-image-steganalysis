"""Aggregate CUDA-smoke receipt and the epoch budget it fixes for the baseline.

Reads a smoke run directory (run_manifest.json, train_log.jsonl, predictions/*.json, *.verify.json),
the launch-gate JSON and an nvidia-smi sample CSV, and writes one aggregate JSON. Readiness only: the
smoke's numbers are never a model result (SMOKE_READINESS_ONLY_NOT_A_MODEL_RESULT).

Budget rule (phase-1 plan, made explicit about evaluation and startup):

    E = min(E_cap, floor((cap_s - margin_s - startup_s - eval_s) / epoch_s))

with epoch_s = fit_rows / train_img_s, eval_s = valid_rows / eval_img_s, and the slower of the epoch-average
and steady-state train rates. If E < 2 the full-data baseline is NO-GO.

    uv run python -m tools.readiness.smoke_receipt --run-dir RUN --gate GATE.json \
        --gpu-csv RUN/gpu_samples.csv \
        --wall-s 412 --out evidence/phase2/cuda_smoke.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

FULL_FIT_ROWS = 240_000
FULL_VALID_ROWS = 60_000


def steady_train_rate(log_rows: list[dict], batch: int) -> float | None:
    """img/s between the middle and the last logged step (cumulative img_per_s -> elapsed time)."""
    steps = [r for r in log_rows if "global_step" in r and "img_per_s" in r]
    if len(steps) < 3:
        return None
    a, b = steps[len(steps) // 2], steps[-1]
    t_a = a["global_step"] * batch / a["img_per_s"]
    t_b = b["global_step"] * batch / b["img_per_s"]
    return (b["global_step"] - a["global_step"]) * batch / (t_b - t_a)


def gpu_summary(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    mem, util = [], []
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            try:
                mem.append(int(row[1]))
                util.append(int(row[2]))
            except ValueError:
                continue
    if not mem:
        return None
    busy = [u for u in util if u > 0]
    return {
        "samples": len(mem),
        "device_memory_used_mib_max": max(mem),
        "device_memory_used_mib_min": min(mem),
        "utilization_pct_mean": round(sum(util) / len(util), 1),
        "utilization_pct_mean_nonzero": round(sum(busy) / len(busy), 1) if busy else 0.0,
        "note": "device totals (includes other tenants' ~893 MiB idle allocation); WSL hides per-process use",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--gate", type=Path, required=True)
    ap.add_argument("--gpu-csv", type=Path)
    ap.add_argument(
        "--wall-s", type=float, required=True, help="measured wall time of the whole smoke command"
    )
    ap.add_argument("--cap-s", type=float, default=10_800)
    ap.add_argument("--margin-s", type=float, default=900)
    ap.add_argument("--epochs-cap", type=int, default=6)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    run = a.run_dir
    m = json.loads((run / "run_manifest.json").read_text())
    log = [json.loads(x) for x in (run / "train_log.jsonl").read_text().splitlines() if x.strip()]
    pred_meta = json.loads(next((run / "predictions").glob("valid_epoch*[0-9].json")).read_text())
    ver = json.loads(next((run / "predictions").glob("valid_epoch*.verify.json")).read_text())
    gate = json.loads(a.gate.read_text())

    ep = m["epochs"][-1]
    batch = m["plan"]["batch_size"]
    avg_rate = ep["img_per_s"]
    steady = steady_train_rate(log, batch)
    train_rate = min(r for r in (avg_rate, steady) if r)
    eval_rows = pred_meta["membership"]["rows"]
    eval_rate = eval_rows / pred_meta["inference_wall_s"]
    startup_s = max(0.0, a.wall_s - ep["wall_s"] - pred_meta["inference_wall_s"])
    epoch_s = FULL_FIT_ROWS / train_rate
    eval_s = FULL_VALID_ROWS / eval_rate
    usable = a.cap_s - a.margin_s - startup_s - eval_s
    e = min(a.epochs_cap, math.floor(usable / epoch_s))
    losses = [r["loss"] for r in log if "loss" in r]

    out = {
        "kind": "cuda_smoke_receipt",
        "result_label": m["result_label"],
        "run_id": m["run_id"],
        "status": m["status"],
        "started_at": m["started_at"],
        "ended_at": m.get("ended_at"),
        "source": m["source"],
        "device": m["device"],
        "versions": m["versions"],
        "launch_gate": {
            "decision": gate["final"]["decision"],
            "failing": gate["final"]["failing"],
            "advisory_failing": gate["final"].get("advisory_failing", []),
            "checks": gate["final"]["checks"],
            "waited_s": gate["waited_s"],
            "attempts": len(gate["attempts"]),
        },
        "population": {
            "fit": "all 240,000 fit rows sampled (200 steps x 32 = 6,400 images seen)",
            "valid": m["population_audit"].get("population"),
            "valid_rows_evaluated": eval_rows,
        },
        "train": {
            "steps": ep["steps"],
            "images": ep["images"],
            "wall_s": ep["wall_s"],
            "img_per_s_epoch_average": avg_rate,
            "img_per_s_steady_second_half": round(steady, 1) if steady else None,
            "loss_first_logged": losses[0] if losses else None,
            "loss_last_logged": losses[-1] if losses else None,
            "loss_all_finite": all(math.isfinite(x) for x in losses),
            "train_loss_mean": ep["train_loss_mean"],
            "peak_vram_allocated_mib": ep.get("peak_vram_allocated_mib"),
            "peak_vram_reserved_mib": ep.get("peak_vram_reserved_mib"),
            "num_workers": m["config"]["train"]["num_workers"],
            "batch_size": batch,
            "amp_dtype": m["config"]["train"]["amp_dtype"],
            "channels_last": m["config"]["train"]["channels_last"],
        },
        "eval": {
            "rows": eval_rows,
            "wall_s": pred_meta["inference_wall_s"],
            "img_per_s": round(eval_rate, 1),
            "batch_size": m["config"]["eval"]["batch_size"],
            "amp_dtype": pred_meta["eval_amp_dtype"],
            "membership": pred_meta["membership"],
        },
        "checkpoint_reload_verify": {
            "passed": ver["passed"],
            "fresh_process": True,
            "observed": ver["observed"],
            "tolerance": ver["tolerance"],
            "checks": ver["checks"],
            "rows": ver["rows"],
        },
        "gpu_samples": gpu_summary(a.gpu_csv),
        "wall_s_total": a.wall_s,
        "startup_and_io_overhead_s": round(startup_s, 1),
        "baseline_budget": {
            "rule": "E = min(E_cap, floor((cap_s - margin_s - startup_s - eval_s) / epoch_s))",
            "cap_s": a.cap_s,
            "margin_s": a.margin_s,
            "startup_s": round(startup_s, 1),
            "train_img_per_s_used": round(train_rate, 1),
            "train_rate_basis": "slower of epoch average and steady second half",
            "epoch_s_full_fit": round(epoch_s, 1),
            "eval_s_full_valid": round(eval_s, 1),
            "usable_s": round(usable, 1),
            "epochs_cap": a.epochs_cap,
            "E": e,
            "decision": "GO" if e >= 2 else "NO-GO (E < 2): full-data baseline does not fit",
            "projected_wall_s": round(startup_s + e * epoch_s + eval_s, 1),
        },
        "redaction": "aggregates only; no image ids, PIDs, paths or env vars",
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out["train"] | out["eval"] | out["baseline_budget"], indent=2))


if __name__ == "__main__":
    main()
