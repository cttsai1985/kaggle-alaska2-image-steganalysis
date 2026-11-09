"""Tiny CPU end-to-end: train -> evaluate -> verify, resume drill and identity rejection.

Every number here is labeled FIXTURE_NOT_A_MODEL_RESULT; these tests prove plumbing, not model quality.
"""

import json

import polars as pl
import pytest

from alaska2.cli import main
from alaska2.constants import CLASSES
from alaska2.evaluate import evaluate, verify
from alaska2.model import load_checkpoint
from alaska2.splits import IdentityError
from alaska2.train import FaultInjected, train

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def trained(tmp_path_factory, fixture_data):
    from pathlib import Path

    from alaska2.config import load_config

    cfg = load_config(
        Path(__file__).resolve().parents[1] / "configs" / "fixture_cpu.toml",
        [f'data.root="{fixture_data["root"]}"'],
    )
    run = train(cfg, run_dir=tmp_path_factory.mktemp("runs") / "straight")
    return cfg, run


def test_run_manifest_records_identities(trained):
    _, run = trained
    m = json.loads((run / "run_manifest.json").read_text())
    assert m["status"] == "completed" and m["result_label"] == "FIXTURE_NOT_A_MODEL_RESULT"
    for k in ("config_sha256", "input_manifest_sha256", "split_folds_sha256", "population_sha256"):
        assert len(m["identities"][k]) == 64
    assert "git_sha" in m["source"] and m["versions"]["torch"] and m["versions"]["timm"]
    assert [e["epoch"] for e in m["epochs"]] == [1, 2] and m["final_checkpoint"].endswith("epoch_002.pt")
    assert m["checkpoint_policy"].startswith("last_completed_epoch")


def test_checkpoint_holds_full_training_state(trained):
    _, run = trained
    ck = load_checkpoint(run / "checkpoints" / "last.pt")
    for k in ("model", "optimizer", "scheduler", "scaler", "rng", "config", "identities", "model_sha256"):
        assert k in ck
    assert ck["epoch"] == 2 and ck["classes"] == list(CLASSES)
    assert set(ck["rng"]) >= {"python", "numpy", "torch"}


def test_evaluate_then_verify(trained):
    _, run = trained
    pred_path = evaluate(run / "checkpoints" / "last.pt")
    pred = pl.read_parquet(pred_path)
    meta = json.loads(pred_path.with_suffix(".json").read_text())
    want = {
        "image",
        "family_id",
        "kind",
        "label",
        "fold",
        "quality",
        "score",
        "model_sha256",
        *(f"p_{c}" for c in CLASSES),
        *(f"logit_{c}" for c in CLASSES),
    }
    assert want <= set(pred.columns)
    assert pred["fold"].unique().to_list() == [1]
    assert meta["membership"]["missing"] == 0 and meta["membership"]["rows"] == pred.height
    assert meta["result_label"] == "FIXTURE_NOT_A_MODEL_RESULT"
    assert (pred["score"] - (1 - pred["p_Cover"])).abs().max() < 1e-12
    assert set(meta["metrics"]["by_quality"]) <= {"75", "90", "95"}
    res = verify(pred_path)
    assert res["passed"], res
    assert res["observed"]["max_abs_prob_delta"] <= res["tolerance"]["prob_atol"]


def test_verify_detects_tampered_predictions(trained, tmp_path):
    _, run = trained
    pred_path = evaluate(run / "checkpoints" / "last.pt", out_dir=tmp_path)
    pred = pl.read_parquet(pred_path)
    pred.with_columns((pl.col("p_Cover") * 0.5).alias("p_Cover")).write_parquet(pred_path)
    res = verify(pred_path)
    assert not res["passed"]
    assert res["checks"]["predictions_sha256_matches"] is False
    assert res["observed"]["max_abs_prob_delta"] > 1e-3


def test_interrupted_run_resumes_to_identical_weights(trained, tmp_path, fixture_cfg):
    _, straight = trained
    drill_cfg = fixture_cfg("train.fault_inject_after_epoch=1")
    run = tmp_path / "drill"
    with pytest.raises(FaultInjected):
        train(drill_cfg, run_dir=run)
    m = json.loads((run / "run_manifest.json").read_text())
    assert m["status"] == "interrupted" and len(m["epochs"]) == 1  # kept visible, not deleted
    train(drill_cfg, resume=run)
    m = json.loads((run / "run_manifest.json").read_text())
    assert m["status"] == "completed" and len(m["resumes"]) == 1
    a = load_checkpoint(straight / "checkpoints" / "last.pt")
    b = load_checkpoint(run / "checkpoints" / "last.pt")
    assert a["model_sha256"] == b["model_sha256"]  # same update sequence as the uninterrupted run
    assert a["global_step"] == b["global_step"]


def test_persistent_worker_resume_replays_dropout_identically(fixture_cfg, tmp_path):
    """Workers > 0 with persistent workers and nonzero dropout: the loader's base-seed draw must not consume
    the global CPU RNG, or the resumed run's dropout masks diverge from the uninterrupted run."""
    extra = ("train.num_workers=2", "train.persistent_workers=true", "model.drop_rate=0.5")
    cfg = fixture_cfg(*extra)
    assert cfg["model"]["drop_rate"] > 0
    straight = train(cfg, run_dir=tmp_path / "straight_pw")
    drill_cfg = fixture_cfg(*extra, "train.fault_inject_after_epoch=1")
    run = tmp_path / "drill_pw"
    with pytest.raises(FaultInjected):
        train(drill_cfg, run_dir=run)
    train(drill_cfg, resume=run)
    a = load_checkpoint(straight / "checkpoints" / "last.pt")
    b = load_checkpoint(run / "checkpoints" / "last.pt")
    assert a["global_step"] == b["global_step"]
    assert a["model_sha256"] == b["model_sha256"]


def test_resume_rejects_changed_identity(trained, fixture_cfg):
    _, run = trained
    with pytest.raises(IdentityError, match="config_sha256"):
        train(fixture_cfg("train.lr=0.5"), resume=run)


def test_fresh_run_id_required(trained):
    cfg, run = trained
    with pytest.raises(FileExistsError):
        train(cfg, run_dir=run)


def test_evaluate_rejects_changed_split(trained, fixture_data, tmp_path):
    import shutil

    _, run = trained
    alt = tmp_path / "alt"
    shutil.copytree(fixture_data["root"], alt)
    folds = pl.read_parquet(alt / "manifest" / "legacy_family_folds.parquet")
    folds.with_columns(
        pl.when(pl.col("fold") == 1)
        .then(2)
        .when(pl.col("fold") == 2)
        .then(1)
        .otherwise(pl.col("fold"))
        .cast(pl.Int32)
        .alias("fold")
    ).write_parquet(alt / "manifest" / "legacy_family_folds.parquet")
    with pytest.raises(IdentityError):
        evaluate(run / "checkpoints" / "last.pt", out_dir=tmp_path / "p", overrides={"data.root": str(alt)})


def test_failed_run_manifest_is_kept(fixture_cfg, tmp_path):
    run = tmp_path / "fail"
    with pytest.raises(ValueError):
        train(fixture_cfg("train.batch_size=10000"), run_dir=run)
    m = json.loads((run / "run_manifest.json").read_text())
    assert m["status"] == "failed" and "smaller than one batch" in m["error"]


def test_cuda_request_without_gpu_is_an_error_not_a_fallback(monkeypatch):
    import torch

    from alaska2.runtime import resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="cuda"):
        resolve_device("cuda")


def test_cli_validate_train_evaluate_predict_verify(fixture_data, tmp_path, capsys):
    from pathlib import Path

    cfg = str(Path(__file__).resolve().parents[1] / "configs" / "fixture_cpu.toml")
    root = f'data.root="{fixture_data["root"]}"'
    assert (
        main(["validate", "--config", cfg, "--set", root, "--rehash-sample", "10", "--decode-sample", "4"])
        == 0
    )
    run = tmp_path / "cli"
    assert (
        main(["train", "--config", cfg, "--set", root, "--set", "train.epochs=1", "--run-dir", str(run)]) == 0
    )
    pred = run / "predictions" / "valid_epoch001.parquet"
    assert pred.exists()  # evaluate_at_end
    assert main(["verify", "--predictions", str(pred)]) == 0
    assert main(["predict", "--checkpoint", str(run / "checkpoints" / "last.pt")]) == 0
    test_pred = pl.read_parquet(run / "predictions" / "test_epoch001.parquet")
    assert test_pred.height == 5 and test_pred["kind"].unique().to_list() == ["Test"]
    assert main(["verify", "--predictions", str(run / "predictions" / "test_epoch001.parquet")]) == 0
    assert (
        main(
            [
                "evaluate",
                "--checkpoint",
                str(run / "checkpoints" / "last.pt"),
                "--out-dir",
                str(tmp_path / "ev"),
            ]
        )
        == 0
    )
    capsys.readouterr()


def test_training_loop_can_memorize_one_batch(fixture_cfg, tmp_path):
    """Optimizer/scheduler/loss wiring check: 8 fixed images, no augmentation, dropout or smoothing."""
    cfg = fixture_cfg(
        "split.max_fit_families=2",
        "train.epochs=8",
        "transforms.train=[]",
        "train.warmup_steps=0",
        "model.drop_rate=0.0",
        "train.label_smoothing=0.0",
    )
    run = train(cfg, run_dir=tmp_path / "memorize")
    losses = [e["train_loss_mean"] for e in json.loads((run / "run_manifest.json").read_text())["epochs"]]
    assert losses[-1] < 0.1 * losses[0], losses


# --- verify must enforce the declared population and canonical labels (review finding 1) -----------------


def _self_consistent(pred_path, frame):
    """Rewrite predictions and refresh the sidecar hash, so only membership/label checks can catch it."""
    from alaska2.identity import sha256_file
    from alaska2.runtime import write_json

    frame.write_parquet(pred_path)
    meta_path = pred_path.with_suffix(".json")
    meta = json.loads(meta_path.read_text())
    meta["predictions_sha256"] = sha256_file(pred_path)
    write_json(meta_path, meta)


def test_verify_rejects_fit_row_predictions(trained, tmp_path):
    from alaska2.evaluate import infer, load_model, prediction_frame
    from alaska2.identity import sha256_file
    from alaska2.inputs import prepare_inputs
    from alaska2.runtime import resolve_device

    cfg, run = trained
    ckpt_path = run / "checkpoints" / "last.pt"
    pred_path = evaluate(ckpt_path, out_dir=tmp_path)
    inputs = prepare_inputs(cfg)
    fit = inputs.records.filter(pl.col("role") == "fit")
    model, ckpt = load_model(ckpt_path, resolve_device("cpu"))
    logits = infer(model, fit, inputs.images_dir, cfg, resolve_device("cpu"))
    _self_consistent(pred_path, prediction_frame(fit, logits, ckpt, sha256_file(ckpt_path)))
    with pytest.raises(IdentityError, match="prediction IDs differ"):
        verify(pred_path)


def test_verify_rejects_missing_holdout_rows(trained, tmp_path):
    _, run = trained
    pred_path = evaluate(run / "checkpoints" / "last.pt", out_dir=tmp_path)
    pred = pl.read_parquet(pred_path)
    _self_consistent(pred_path, pred.head(pred.height - 4))
    with pytest.raises(IdentityError, match="4 missing"):
        verify(pred_path)


def test_verify_rejects_wrong_saved_labels(trained, tmp_path):
    _, run = trained
    pred_path = evaluate(run / "checkpoints" / "last.pt", out_dir=tmp_path)
    pred = pl.read_parquet(pred_path)
    flipped = pred.with_columns(pl.when(pl.col("label") == 0).then(1).otherwise(0).alias("label"))
    _self_consistent(pred_path, flipped)
    with pytest.raises(IdentityError, match="label"):
        verify(pred_path)


def test_verify_rejects_wrong_saved_fold(trained, tmp_path):
    _, run = trained
    pred_path = evaluate(run / "checkpoints" / "last.pt", out_dir=tmp_path)
    pred = pl.read_parquet(pred_path)
    _self_consistent(pred_path, pred.with_columns((pl.col("fold") + 1).cast(pred["fold"].dtype)))
    with pytest.raises(IdentityError, match="fold"):
        verify(pred_path)


def test_verify_reports_membership(trained, tmp_path):
    _, run = trained
    pred_path = evaluate(run / "checkpoints" / "last.pt", out_dir=tmp_path)
    res = verify(pred_path)
    assert res["passed"] and res["checks"]["saved_rows_match_canonical_inputs"]
    assert res["membership"]["missing"] == 0 and res["membership"]["rows"] == res["rows"]


# --- resume must not mix source revisions or runtime environments (review finding 3) ----------------------


def test_checkpoint_records_provenance(trained):
    _, run = trained
    ck = load_checkpoint(run / "checkpoints" / "last.pt")
    m = json.loads((run / "run_manifest.json").read_text())
    assert ck["provenance"]["source"] == m["source"] and ck["provenance"]["versions"] == m["versions"]
    assert len(ck["provenance"]["source"]["package_sha256"]) == 64


@pytest.mark.parametrize(
    "section,key,value",
    [("source", "git_sha", "0" * 40), ("source", "package_sha256", "f" * 64), ("versions", "torch", "0.0")],
)
def test_resume_rejects_changed_provenance(fixture_cfg, tmp_path, monkeypatch, section, key, value):
    import alaska2.train as train_mod
    from alaska2.identity import run_provenance

    cfg = fixture_cfg("train.fault_inject_after_epoch=1")
    run = tmp_path / "prov"
    with pytest.raises(FaultInjected):
        train(cfg, run_dir=run)
    changed = run_provenance()
    changed[section][key] = value
    monkeypatch.setattr(train_mod, "run_provenance", lambda: changed)
    with pytest.raises(IdentityError, match=f"{section}.{key}"):
        train(cfg, resume=run)
    m = json.loads((run / "run_manifest.json").read_text())
    assert m["status"] == "interrupted" and "resumes" not in m  # rejected before any continuation


def test_resume_records_each_continuation_provenance(fixture_cfg, tmp_path):
    cfg = fixture_cfg("train.fault_inject_after_epoch=1")
    run = tmp_path / "prov_ok"
    with pytest.raises(FaultInjected):
        train(cfg, run_dir=run)
    train(cfg, resume=run)
    m = json.loads((run / "run_manifest.json").read_text())
    assert m["resumes"][0]["provenance"]["source"] == m["source"]


@pytest.mark.parametrize("workers", [0, 2])
def test_sigterm_wall_clock_cap_closes_manifest_as_interrupted(fixture_data, tmp_path, workers):
    """`timeout` SIGTERMs the whole process group (main + DataLoader workers): status must be interrupted,
    not 'running' and not 'failed' from a worker that died first."""
    import os
    import signal
    import subprocess
    import sys
    import time
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    run = tmp_path / "capped"
    cmd = [
        sys.executable,
        "-m",
        "alaska2",
        "train",
        "--config",
        str(repo / "configs" / "fixture_cpu.toml"),
        "--set",
        f'data.root="{fixture_data["root"]}"',
        "--set",
        "train.epochs=500",
        "--set",
        f"train.num_workers={workers}",
        "--set",
        f"train.persistent_workers={'true' if workers else 'false'}",
        "--run-dir",
        str(run),
    ]
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
    }  # fixture is CPU-only: never open a context on the shared GPU
    proc = subprocess.Popen(
        cmd,
        cwd=repo,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        env=env,
    )
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:  # wait until training has started and written one epoch
        if (run / "checkpoints" / "last.pt").exists():
            break
        time.sleep(0.2)
    else:
        proc.kill()
        pytest.fail("training did not reach its first checkpoint")
    time.sleep(0.5)  # let the next epoch's workers start
    os.killpg(proc.pid, signal.SIGTERM)  # as `timeout` does: every process in the group
    try:
        _, err = proc.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        # Known limit with workers: they ignore SIGTERM, so interpreter exit waits on them until
        # `timeout --kill-after=60` sends SIGKILL. The manifest is already closed by then.
        os.killpg(proc.pid, signal.SIGKILL)
        _, err = proc.communicate(timeout=60)
        assert workers > 0, "only the worker case may need the SIGKILL follow-up"
    assert proc.returncode != 0
    m = json.loads((run / "run_manifest.json").read_text())
    assert m["status"] == "interrupted" and "SIGTERM" in m["error"], (m["status"], m.get("error"), err[-500:])
    assert m["epochs"] and m["final_checkpoint"] and m["terminated_by_signal"] == signal.SIGTERM
