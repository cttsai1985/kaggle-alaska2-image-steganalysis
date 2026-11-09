# Results and remaining evidence

The modern baseline has no reportable model score yet. The retained artifacts establish data/split integrity, CUDA readiness and interrupted training, with separate CPU recovery tests. They do not establish a completed baseline, a modern ensemble or a scored submission.

## Evaluation scope

**MEASURED input snapshot:** 305,000 JPEGs: 300,000 labeled variants from 75,000 complete families, plus 5,000 test images. The retained legacy family holdout assigns 240,000 files to fitting and 60,000 to validation, with no family or byte-identical group crossing those roles. This is a single holdout, not cross-validation. [Dataset receipt](../evidence/phase1/dataset_audit.json) · [Validation receipt](../evidence/phase2/validate_real_data.json).

| Evidence | Status | Meaning |
| --- | --- | --- |
| Historical leaderboard submissions | Not recovered | No public/private score or percentile is claimed. |
| CUDA smoke and independent reload | Passed readiness check | Explicit subset; not a model result. |
| Fixed baseline | Interrupted | Completed checkpoint retained, intended recipe incomplete. |
| Baseline holdout metrics and ROC/quality analysis | Not available | No saved baseline predictions to analyze. |
| Baseline checkpoint-to-prediction verification | Not available | No saved baseline predictions to reconstruct. |
| Modern test predictions / submission / leaderboard result | Not established | The prediction command is implemented; execution is not demonstrated. |
| Ensemble provenance and delivery recovery | Outstanding | See [issue #4](https://github.com/cttsai1985/kaggle-alaska2-image-steganalysis/issues/4). |

The [CUDA smoke receipt](../evidence/phase2/cuda_smoke.json) contains a subset score solely to exercise the metric and reload path. It must not be substituted for the baseline or compared with historical leaderboard scores. The [failed duplicate smoke](../evidence/phase2/failed_duplicate_smoke.json) is retained as a failed attempt.

## Interrupted training

The [baseline specification](../evidence/phase2/baseline_spec.json) declared the recipe before launch. **ESTIMATED before fitting:** the smoke-derived budget selected five epochs under a three-hour overall cap. Those projections are not measured baseline completion time.

**MEASURED from the retained run manifest:**

| Item | Observed value |
| --- | --- |
| Declared fit population | 240,000 files |
| Declared epochs | 5 |
| Completed epoch records | 4 |
| Final completed checkpoint | `checkpoints/epoch_004.pt` |
| Latest status | `interrupted` |
| Latest recorded error | SIGTERM, handled as `KeyboardInterrupt` |
| Accepted continuation | From epoch 4, with matching recorded source/package/runtime provenance |
| Saved holdout predictions | None |
| Fresh-process baseline verification | None |

[Generated aggregate receipt](../evidence/phase2/baseline_interrupted.json). Its epoch losses describe training only and are not validation metrics. Source provenance is recorded as dirty; the receipt retains its package-content hash rather than implying a clean checkout.

The transcript reports a host outage, but no dedicated outage-timing receipt or `wall.json` is present in the retained run directory. The manifest proves an accepted continuation and a later interruption. It does not by itself quantify outage duration, active compute across attempts or end-to-end budget compliance. The projected budget cannot be presented as any of those measurements.

**Decision:** retain the exact recovery worktree and checkpoints; do not silently restart under newer code, extend the training budget or call the run complete. The predeclared fallback permits evaluation of the last completed checkpoint, explicitly labeled incomplete. That evaluation, its resource allowance and an independent prediction reconstruction are the next model-evidence work. The [baseline issue](https://github.com/cttsai1985/kaggle-alaska2-image-steganalysis/issues/3) remains open.

## What the recovery tests establish

**MEASURED CPU fixture:** interrupted and uninterrupted training produce identical model hashes, including the persistent-worker/dropout case. Changed source or runtime provenance is rejected. The fixture's saved predictions reconstruct with zero observed probability and metric delta in the retained receipt. [CPU contracts](../evidence/phase2/cpu_contracts.json) · [Behavioral tests](../tests/test_e2e_cpu.py).

Those checks exercise synthetic data. They do not establish bit-for-bit replay of the interrupted CUDA run, recovery of evaluation/delivery stages or remote artifact integrity. The manifest, checkpoint, prediction and delivery states answer different completion questions.

## Outstanding work

| Issue | State supported by merged evidence | Remaining acceptance work |
| --- | --- | --- |
| [Readiness #2](https://github.com/cttsai1985/kaggle-alaska2-image-steganalysis/issues/2) | Complete | Readiness/data/split receipts and the bounded plan are merged in [PR #6](https://github.com/cttsai1985/kaggle-alaska2-image-steganalysis/pull/6). |
| [Baseline #3](https://github.com/cttsai1985/kaggle-alaska2-image-steganalysis/issues/3) | Partial | Produce explicitly scoped real-model predictions, reconstruct them independently, and explain ROC/cohort results; retain incomplete-budget status. |
| [Ensemble and delivery #4](https://github.com/cttsai1985/kaggle-alaska2-image-steganalysis/issues/4) | Outstanding | Audit cached prediction/OOF provenance, verify ensemble arithmetic and submission identity, and demonstrate recoverable artifact transfer. |
| [Walkthrough and delivery #5](https://github.com/cttsai1985/kaggle-alaska2-image-steganalysis/issues/5) | Partial | CPU CI and written decision stories exist; executed notebook, container/dev environment, hosted execution status, independent review and release evidence remain. |
| [Parent #1](https://github.com/cttsai1985/kaggle-alaska2-image-steganalysis/issues/1) | Open | Complete or explicitly disposition the remaining phase acceptance criteria. |

The README follows the same problem → findings → results → decisions → reproduction pattern as [Mercedes-Benz](https://github.com/cttsai1985/kaggle-mercedes-benz-solution), [Iceberg](https://github.com/cttsai1985/kaggle-statoil-iceberg-solution), [Home Credit](https://github.com/cttsai1985/kaggle-home-credit-solution) and [QUEST](https://github.com/cttsai1985/kaggle-google-quest-solution). Their completed-model and delivery claims are not imported into ALASKA2.
