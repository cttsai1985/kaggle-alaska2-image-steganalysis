"""Thin CLI: ``python -m alaska2 validate|train|evaluate|predict|verify``."""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

from alaska2.config import config_hash, load_config


def _overrides(a: argparse.Namespace) -> dict:
    return {
        "data.root": a.data_root,
        "train.device": a.device,
        "train.num_workers": a.num_workers,
        "eval.batch_size": a.eval_batch_size,
    }


def _runtime_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-root", help="override data.root (paths only; identities are still checked)")
    p.add_argument("--device", choices=["cpu", "cuda"])
    p.add_argument("--num-workers", type=int)
    p.add_argument("--eval-batch-size", type=int)


def cmd_validate(a: argparse.Namespace) -> int:
    from alaska2.inputs import decode_sample, prepare_inputs, rehash_sample

    cfg = load_config(a.config, a.set)
    inputs = prepare_inputs(cfg)
    out = {
        "kind": "validate",
        "config_sha256": config_hash(cfg),
        "purpose": cfg["purpose"],
        "identities": inputs.identities,
        "family_separation": {"families_crossing_roles": 0, "content_groups_crossing_roles": 0},
        "image_size_checked": cfg["data"]["image_size"],
        "population_audit": inputs.audit,
        "test_rows": inputs.test.height,
        "rehash_sample": rehash_sample(inputs, a.rehash_sample, int(cfg["train"]["seed"])),
        "decode_sample": decode_sample(inputs, cfg, a.decode_sample, int(cfg["train"]["seed"])),
    }
    text = json.dumps(out, indent=2, sort_keys=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n")
    print(text)
    return 0 if out["rehash_sample"]["mismatched"] == 0 else 2


def cmd_train(a: argparse.Namespace) -> int:
    from alaska2.config import data_paths
    from alaska2.evaluate import evaluate
    from alaska2.train import train

    cfg = load_config(a.config, a.set)
    run_dir = None
    if a.run_id:
        run_dir = data_paths(cfg)["runs_dir"] / a.run_id
    if a.run_dir:
        run_dir = Path(a.run_dir)
    run_dir = train(cfg, run_dir=run_dir, resume=Path(a.resume) if a.resume else None)
    print(f"run_dir={run_dir}")
    if cfg["train"].get("evaluate_at_end") and not a.no_evaluate:
        pred = evaluate(run_dir / "checkpoints" / "last.pt")
        print(f"predictions={pred}")
    return 0


def cmd_evaluate(a: argparse.Namespace, split: str) -> int:
    from alaska2.evaluate import evaluate

    path = evaluate(Path(a.checkpoint), Path(a.out_dir) if a.out_dir else None, _overrides(a), split=split)
    meta = json.loads(path.with_suffix(".json").read_text())
    print(
        json.dumps(
            {
                "predictions": str(path),
                "membership": meta["membership"],
                "result_label": meta["result_label"],
                "metrics": meta.get("metrics", {}).get("pooled"),
            },
            indent=2,
        )
    )
    return 0


def cmd_verify(a: argparse.Namespace) -> int:
    from alaska2.evaluate import verify

    res = verify(
        Path(a.predictions),
        Path(a.checkpoint) if a.checkpoint else None,
        _overrides(a),
        Path(a.out) if a.out else None,
    )
    print(json.dumps(res, indent=2, sort_keys=True))
    return 0 if res["passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="alaska2", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("validate", help="verify input identities, family separation and population counts")
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--rehash-sample", type=int, default=0, help="re-read N random files and compare sha256")
    p.add_argument("--decode-sample", type=int, default=0, help="decode N fit rows through the Dataset")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_validate)

    p = sub.add_parser("train", help="fixed-budget training; resumable from the last completed epoch")
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--run-id")
    p.add_argument("--run-dir")
    p.add_argument("--resume", metavar="RUN_DIR")
    p.add_argument("--no-evaluate", action="store_true")
    p.set_defaults(fn=cmd_train)

    for name, split in (("evaluate", "valid"), ("predict", "test")):
        p = sub.add_parser(
            name,
            help="holdout predictions + metrics" if split == "valid" else "Test/ predictions (no submission)",
        )
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--out-dir")
        _runtime_args(p)
        p.set_defaults(fn=lambda a, s=split: cmd_evaluate(a, s))

    p = sub.add_parser("verify", help="reload checkpoint and rebuild saved predictions within tolerance")
    p.add_argument("--predictions", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--out")
    _runtime_args(p)
    p.set_defaults(fn=cmd_verify)
    return ap


def _sigterm_as_interrupt(signum, frame):
    """``timeout`` (the wall-clock cap) sends SIGTERM; raise so the run manifest is closed as interrupted."""
    raise KeyboardInterrupt(f"signal {signum} (SIGTERM, e.g. the wall-clock cap)")


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    signal.signal(signal.SIGTERM, _sigterm_as_interrupt)
    return int(a.fn(a))


if __name__ == "__main__":
    sys.exit(main())
