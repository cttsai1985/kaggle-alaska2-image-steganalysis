"""Fixed-budget training with AMP, an explicit device and fully resumable checkpoints.

Checkpoint policy (declared before fitting): the **last completed epoch** is the model. Holdout labels are
never used to pick an epoch. Every epoch writes ``checkpoints/epoch_NNN.pt`` and ``last.pt`` atomically;
each holds model, optimizer, scheduler, grad-scaler and RNG state plus the run identities, so an interrupted
run resumes from the last completed epoch and reproduces the uninterrupted sequence of updates.
"""

from __future__ import annotations

import math
import shutil
import time
import traceback
from pathlib import Path

import torch
from torch import nn

from alaska2.config import config_hash, data_paths
from alaska2.constants import CLASSES, PURPOSE_LABELS
from alaska2.data import ImageDataset
from alaska2.identity import git_source, runtime_versions, sha256_file
from alaska2.inputs import prepare_inputs, seed_everything
from alaska2.model import build_model, load_checkpoint, state_dict_sha256
from alaska2.runtime import (
    autocast,
    make_loader,
    resolve_device,
    rng_state,
    save_atomic,
    set_rng_state,
    utc_now,
    write_json,
)
from alaska2.splits import IdentityError

CHECKPOINT_FORMAT = "alaska2-checkpoint-v1"
IDENTITY_KEYS = ("config_sha256", "input_manifest_sha256", "split_folds_sha256", "population_sha256")


class FaultInjected(RuntimeError):
    """Raised by ``train.fault_inject_after_epoch`` for a labeled interruption drill."""


def lr_lambda_factory(warmup: int, total: int):
    def f(step: int) -> float:
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return f


def default_run_id(cfg: dict) -> str:
    return f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{cfg['purpose']}-{config_hash(cfg)[:8]}"


def train(cfg: dict, run_dir: Path | None = None, resume: Path | None = None) -> Path:
    t = cfg["train"]
    device = resolve_device(t["device"])
    inputs = prepare_inputs(cfg)
    identities = {"config_sha256": config_hash(cfg), **inputs.identities}

    if resume is not None:
        run_dir = Path(resume)
        ckpt = load_checkpoint(run_dir / "checkpoints" / "last.pt")
        diff = [k for k in IDENTITY_KEYS if ckpt["identities"].get(k) != identities.get(k)]
        if diff:
            raise IdentityError(f"resume rejected: identities differ on {diff}")
        manifest = _read_manifest(run_dir)
        manifest["resumes"] = [*manifest.get("resumes", []), {"at": utc_now(), "from_epoch": ckpt["epoch"]}]
    else:
        run_dir = Path(run_dir) if run_dir else data_paths(cfg)["runs_dir"] / default_run_id(cfg)
        if run_dir.exists():
            raise FileExistsError(f"run dir exists, use a fresh run_id: {run_dir}")
        run_dir.mkdir(parents=True)
        ckpt = None
        manifest = {
            "run_id": run_dir.name,
            "purpose": cfg["purpose"],
            "result_label": PURPOSE_LABELS[cfg["purpose"]],
            "started_at": utc_now(),
            "source": git_source(),
            "identities": identities,
            "population_audit": inputs.audit,
            "versions": runtime_versions(),
            "classes": list(CLASSES),
            "checkpoint_policy": "last_completed_epoch (fixed budget; holdout labels never select an epoch)",
            "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
            "config_source": cfg.get("_source"),
            "epochs": [],
        }
    manifest.update(status="running", device=str(device))
    write_json(run_dir / "run_manifest.json", manifest)

    try:
        _fit(cfg, inputs, identities, device, run_dir, manifest, ckpt)
    except BaseException as exc:  # keep failed / interrupted runs visible
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt | FaultInjected) else "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["traceback_tail"] = traceback.format_exc().splitlines()[-6:]
        manifest["ended_at"] = utc_now()
        write_json(run_dir / "run_manifest.json", manifest)
        raise
    manifest.update(status="completed", ended_at=utc_now())
    write_json(run_dir / "run_manifest.json", manifest)
    return run_dir


def _read_manifest(run_dir: Path) -> dict:
    import json

    return json.loads((run_dir / "run_manifest.json").read_text())


def _fit(cfg, inputs, identities, device, run_dir, manifest, ckpt) -> None:
    t = cfg["train"]
    seed = int(t["seed"])
    seed_everything(seed)
    torch.backends.cudnn.benchmark = bool(t.get("cudnn_benchmark", False))
    if device.type == "cpu" and t.get("cpu_threads"):
        torch.set_num_threads(int(t["cpu_threads"]))

    fit = inputs.records.filter(inputs.records["role"] == "fit")
    ds = ImageDataset(fit, inputs.images_dir, cfg, train=True)
    batch = int(t["batch_size"])
    steps_per_epoch = len(ds) // batch
    if t.get("max_steps_per_epoch"):
        steps_per_epoch = min(steps_per_epoch, int(t["max_steps_per_epoch"]))
    if steps_per_epoch < 1:
        raise ValueError(f"fit set of {len(ds)} rows is smaller than one batch of {batch}")
    epochs = int(t["epochs"])
    total_steps = epochs * steps_per_epoch

    model, weights = build_model(cfg, pretrained=None if ckpt is None else False)
    model.to(device)
    if t.get("channels_last"):
        model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=float(t["lr"]), weight_decay=float(t["weight_decay"]))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(int(t["warmup_steps"]), total_steps))
    scaler = torch.amp.GradScaler(device.type, enabled=t["amp_dtype"] == "fp16")
    loss_fn = nn.CrossEntropyLoss(label_smoothing=float(t.get("label_smoothing", 0.0)))

    start_epoch, global_step = 0, 0
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        set_rng_state(ckpt["rng"])
        start_epoch, global_step = int(ckpt["epoch"]), int(ckpt["global_step"])
        weights = ckpt["pretrained_weights"]
    manifest["pretrained_weights"] = weights
    manifest["plan"] = {
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "fit_rows": len(ds),
        "batch_size": batch,
    }
    write_json(run_dir / "run_manifest.json", manifest)

    loader = make_loader(ds, cfg, batch, train=True, epoch=start_epoch)
    log_every = int(t.get("log_every", 50))
    log_path = run_dir / "train_log.jsonl"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(start_epoch, epochs):
        model.train()
        loader.sampler.set_epoch(epoch)
        t0, seen, loss_sum = time.perf_counter(), 0, 0.0
        for step, (x, y, _) in enumerate(loader):
            if step >= steps_per_epoch:
                break
            x = x.to(device, non_blocking=True)
            if t.get("channels_last"):
                x = x.contiguous(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with autocast(device, t["amp_dtype"]):
                loss = loss_fn(model(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss {loss.item()} at epoch {epoch} step {step}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            global_step += 1
            seen += x.shape[0]
            loss_sum += float(loss.detach()) * x.shape[0]
            if global_step % log_every == 0:
                _append_log(
                    log_path,
                    {
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "loss": float(loss.detach()),
                        "lr": sched.get_last_lr()[0],
                        "img_per_s": seen / (time.perf_counter() - t0),
                    },
                )
        wall = time.perf_counter() - t0
        rec = {
            "epoch": epoch + 1,
            "steps": steps_per_epoch,
            "images": seen,
            "train_loss_mean": loss_sum / seen,
            "wall_s": round(wall, 2),
            "img_per_s": round(seen / wall, 1),
            "finished_at": utc_now(),
        }
        if device.type == "cuda":
            rec["peak_vram_allocated_mib"] = round(torch.cuda.max_memory_allocated(device) / 2**20, 1)
            rec["peak_vram_reserved_mib"] = round(torch.cuda.max_memory_reserved(device) / 2**20, 1)
        state = {
            "format": CHECKPOINT_FORMAT,
            "epoch": epoch + 1,
            "global_step": global_step,
            "model": {k: v.detach().to("cpu") for k, v in model.state_dict().items()},
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict(),
            "rng": rng_state(),
            "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
            "identities": identities,
            "pretrained_weights": weights,
            "classes": list(CLASSES),
        }
        state["model_sha256"] = state_dict_sha256(state["model"])
        path = run_dir / "checkpoints" / f"epoch_{epoch + 1:03d}.pt"
        save_atomic(state, path)
        shutil.copyfile(path, path.with_name("last.pt.tmp"))
        (path.with_name("last.pt.tmp")).replace(path.with_name("last.pt"))
        rec.update(
            checkpoint=str(path.relative_to(run_dir)),
            checkpoint_sha256=sha256_file(path),
            model_sha256=state["model_sha256"],
        )
        manifest["epochs"] = [e for e in manifest["epochs"] if e["epoch"] != epoch + 1] + [rec]
        manifest["final_checkpoint"] = rec["checkpoint"]
        write_json(run_dir / "run_manifest.json", manifest)
        _append_log(log_path, {"event": "epoch_end", **rec})
        if t.get("fault_inject_after_epoch") and epoch + 1 == int(t["fault_inject_after_epoch"]):
            raise FaultInjected(f"fault drill: simulated interruption after epoch {epoch + 1}")


def _append_log(path: Path, rec: dict) -> None:
    import json

    with open(path, "a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
