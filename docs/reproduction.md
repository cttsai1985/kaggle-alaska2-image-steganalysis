# Reproduce the checks and model workflow

Use the maintained package for the current workflow. The original source remains in [legacy/](../legacy/README.md) for attribution and comparison. The commands below describe supported execution; they do not claim the incomplete baseline has finished.

## CPU checks without competition data

Use Python 3.12 and uv from the repository root:

```bash
uv sync --locked
CUDA_VISIBLE_DEVICES='' uv run --locked pytest -q -m 'not cuda'
uv run --locked ruff check src tests tools
uv run --locked ruff format --check src tests tools
```

The locked environment contains CUDA-enabled PyTorch wheels, even for the CPU tests. The test fixture creates synthetic JPEGs in temporary directories; it does not download competition images. Its metrics are labeled `FIXTURE_NOT_A_MODEL_RESULT`. [CPU CI](../.github/workflows/ci.yml) repeats these commands without a GPU.

## Authorized competition inputs

Keep competition data and output artifacts outside Git. The default recipe expects the following layout under `~/data/kaggle/alaska2-image-steganalysis`:

```text
raw/extracted/Cover/                 unchanged JPEGs
raw/extracted/JMiPOD/                corresponding embedded variants
raw/extracted/JUNIWARD/
raw/extracted/UERD/
raw/extracted/Test/                  unlabeled test JPEGs
manifest/files_manifest.parquet      verified input identities
manifest/legacy_family_folds.parquet frozen family assignments
runs/                               private checkpoints, predictions and logs
```

The [readiness report](phase1-readiness.md) records download/access checks, manifest generation, quality-metadata joins and split construction. A directory that merely resembles this layout is insufficient: the baseline config pins the manifest and fold content hashes.

Validate an existing authorized snapshot without starting CUDA training:

```bash
uv run --locked python -m alaska2 validate \
  --config configs/baseline_effb0_fold1.toml \
  --rehash-sample 2000 --decode-sample 64
```

Those are requested sample sizes, not a new validation result. To use another root, append `--set 'data.root="/absolute/path/to/alaska2"'`. The expected input identities remain enforced.

## CUDA training and budgets

CUDA is explicit; asking for it without an available GPU fails. The host is shared, so inspect the resource gate before any authorized execution:

```bash
uv run --locked python -m tools.readiness.launch_gate \
  --data-dir "$HOME/data/kaggle/alaska2-image-steganalysis" --label evaluation
```

The [fixed baseline specification](../evidence/phase2/baseline_spec.json) records the historical recipe and budget; its throughput-based times are estimates. The source TOML is a recipe template, while the specification records the actual overrides used for the interrupted run. Reusing the template alone does not reproduce that run's resolved config.

The bounded-run wrapper is `tools/runs/bounded_run.sh CONFIG RUN_ID CAP_S [KEY=VALUE ...]`. It wraps training/evaluation in a timeout and attempts an independent verify only after successful training and saved predictions. It does not run the resource gate itself. Use an approved fresh run identity and budget; an existing output directory is refused. No additional training budget is implied by this guide.

Resume uses `python -m alaska2 train --config CONFIG --resume RUN_DIR` and checks source, package, runtime, config, input and split provenance. The existing interrupted checkpoint belongs to its preserved recovery worktree. A new documentation commit changes Git provenance, so a resume from the latest checkout is not an equivalent execution. Never edit checkpoint provenance to bypass that rejection.

## Saved-model evaluation and verification

Given an appropriate retained checkpoint and an available execution budget:

```bash
uv run --locked python -m alaska2 evaluate --checkpoint /path/to/checkpoints/last.pt
uv run --locked python -m alaska2 verify --predictions /path/to/predictions/valid_epoch004.parquet
```

These commands use the checkpoint's recorded configuration and identities. Runtime-only flags can change the data path, device, loader-worker count or evaluation batch size; they do not override the evaluation population. For the interrupted baseline, report the last completed checkpoint as an incomplete-run evaluation, never the declared completed baseline.

`predict --checkpoint CHECKPOINT` produces test predictions in the package format, not a competition submission file. Submission schema/ID-order checks and ensemble provenance remain [phase-three work](https://github.com/cttsai1985/kaggle-alaska2-image-steganalysis/issues/4).

## Regenerate the interrupted-run aggregate

From the repository root, with the private run directory available:

```bash
python3 -m tools.runs.run_evidence --run-dir /path/to/retained-run \
  --out /path/to/baseline_interrupted.json
```

This tool uses only the standard library. It reads saved receipts, removes private paths and per-image rows, and refuses output if an absolute home path survives. The checked-in [aggregate](../evidence/phase2/baseline_interrupted.json) adds explanatory notes; the underlying fields come from those saved artifacts. Aggregation does not execute model inference or establish checkpoint validity by itself.

## Execution environments still outstanding

A learning notebook, Docker JupyterLab, dev container and portable offline Kaggle path have not been delivered. The original `legacy/docker/` files are historical source, not a verified current environment. Hosted execution, shared tracking/storage transfer and a release remain unverified for this model. [Delivery acceptance](https://github.com/cttsai1985/kaggle-alaska2-image-steganalysis/issues/5).

## Reproduce the verified incomplete-run fallback

`tools/runs/retained_gpu.sh RUNTIME_CHECKOUT DATA_ROOT OUTPUT_ROOT RUN_ID [CAP_S]` runs evaluation and fresh-process holdout/test verification without training. It uses the existing locked runtime through read-only mounts and requires the local CUDA base image. The current execution's exact image identity and runtime limits are in [the receipt](../evidence/phase2/retained_execution.json). The output directory is exclusively locked; a timeout cleans up only the container created by that invocation.

The launcher declares a finite evaluation allowance independently of the exhausted/incomplete historical training budget. Its resource and mount contract, input/output binding, incomplete-result label and timeout cleanup have regression tests. Source data and checkpoint files are read-only.

After evaluation, aggregate analysis can be recomputed from the saved predictions with `python -m tools.analysis.holdout_analysis`, then bound to the verified checkpoint with:

```bash
uv run --locked python -m tools.runs.retained_receipt \
  --run-dir /path/to/original-run --predictions-dir /path/to/output/predictions \
  --analysis /path/to/holdout_analysis.json --out /path/to/retained_evaluation.json
```

The collector requires baseline purpose, exact source/checkpoint/prediction/population identities, current independent verification, checkpoint-defined tolerances and agreement of the analysis with canonical metrics. It never promotes fixture/smoke evidence to a real-model result.
