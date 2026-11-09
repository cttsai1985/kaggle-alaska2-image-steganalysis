"""Redacted aggregate of one run directory for Git: identities, plan, per-epoch stats, wall time, verify.

Drops absolute paths, image ids and per-row values; keeps hashes, counts and timings.

    uv run python -m tools.runs.run_evidence --run-dir RUN --out evidence/phase2/baseline_run.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _load(p: Path) -> dict | None:
    return json.loads(p.read_text()) if p.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--note", action="append", default=[])
    a = ap.parse_args()
    run = a.run_dir
    m = json.loads((run / "run_manifest.json").read_text())
    preds = (
        sorted((run / "predictions").glob("valid_epoch*[0-9].json")) if (run / "predictions").exists() else []
    )
    pm = _load(preds[-1]) if preds else None
    ver = _load(preds[-1].with_name(preds[-1].stem + ".verify.json")) if preds else None
    out = {
        "kind": "run_evidence",
        "run_id": m["run_id"],
        "result_label": m["result_label"],
        "status": m["status"],
        "error": m.get("error"),
        "started_at": m["started_at"],
        "ended_at": m.get("ended_at"),
        "source": m["source"],
        "identities": m["identities"],
        "population_audit": m["population_audit"],
        "pretrained_weights": m.get("pretrained_weights"),
        "versions": m["versions"],
        "device": m["device"],
        "checkpoint_policy": m["checkpoint_policy"],
        "plan": m.get("plan"),
        "epochs": [{k: v for k, v in e.items() if k != "finished_at"} for e in m["epochs"]],
        "final_checkpoint": m.get("final_checkpoint"),
        "wall": _load(run / "wall.json"),
        "host_contention_at_launch": _load(run / "host_contention.json"),
        "launch_gate": (_load(run / "launch_gate.json") or {}).get("final"),
        "predictions": None
        if pm is None
        else {
            "file": preds[-1].name,
            "epoch": pm["epoch"],
            "predictions_sha256": pm["predictions_sha256"],
            "checkpoint_sha256": pm["checkpoint_sha256"],
            "model_sha256": pm["model_sha256"],
            "membership": pm["membership"],
            "inference_wall_s": pm["inference_wall_s"],
            "eval_amp_dtype": pm["eval_amp_dtype"],
            "decoder": pm["decoder"],
            "metrics_pooled": pm["metrics"]["pooled"],
            "four_class_accuracy": pm["metrics"]["four_class_accuracy"],
        },
        "verify_fresh_process": None
        if ver is None
        else {
            k: ver[k] for k in ("passed", "observed", "tolerance", "checks", "membership", "rows", "device")
        },
        "notes": a.note,
        "redaction": "no absolute paths, image ids, PIDs or env vars; run artifacts stay outside Git",
    }
    if m.get("config"):
        out["config_train"] = m["config"]["train"]
    if m.get("resumes"):
        out["resumes"] = m["resumes"]
    text = re.sub(r"\bpid \d+", "pid <redacted>", json.dumps(out, indent=2, sort_keys=True, default=str))
    if "/home/" in text:
        raise SystemExit("refusing to write: an absolute path survived redaction")
    a.out.write_text(text + "\n")
    print(
        json.dumps(
            {"status": out["status"], "predictions": out["predictions"] and out["predictions"]["file"]}
        )
    )


if __name__ == "__main__":
    main()
