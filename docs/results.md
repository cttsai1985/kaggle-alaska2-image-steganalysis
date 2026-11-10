# Results and evaluation limits

The retained model has a verified real-data result: **MEASURED weighted AUC 0.83540** on the complete family holdout. It is the predeclared fallback from an interrupted training run, not the intended completed recipe. No historical or modern leaderboard score is established.

## Evaluation scope

**MEASURED input snapshot:** 305,000 JPEGs: 300,000 labeled variants from 75,000 complete families, plus 5,000 test images. The retained legacy family assignments use 240,000 fit files and 60,000 holdout files from 15,000 families. No family or byte-identical group crosses those roles. This is a single holdout, not cross-validation. [Input audit](../evidence/phase1/dataset_audit.json) · [Split and input validation](../evidence/phase2/validate_real_data.json).

| Experiment | Scope | Result |
| --- | --- | --- |
| Historical submissions | Competition leaderboard | Scores and submission provenance have not been recovered. |
| Retained EfficientNet checkpoint | Complete family holdout; incomplete training | **MEASURED weighted AUC 0.83540; ordinary AUC 0.77024**. |
| Intended fixed-budget recipe | Declared training completion | Incomplete; no completed-recipe result is claimed. |
| CUDA smoke | Explicit family subset | Readiness only, not the baseline result. |
| CPU fixture | Synthetic images | Contract tests, not a model result. |

[Verified evaluation and analysis](../evidence/phase2/retained_evaluation.json) · [Test-prediction reconstruction](../evidence/phase2/retained_test_verification.json) · [Runtime/image receipt](../evidence/phase2/retained_execution.json).

## Where the pooled score hides weakness

**MEASURED from the same retained holdout predictions:**

| JPEG quality cohort | Files | Source families | Weighted AUC |
| --- | ---: | ---: | ---: |
| 75 | 20,000 | 5,000 | 0.86666 |
| 90 | 20,000 | 5,000 | 0.84809 |
| 95 | 20,000 | 5,000 | 0.76598 |

The cohort sizes and class proportions match, so unequal denominators do not explain the score ordering. However, each family belongs to one quality cohort. Image content and acquisition conditions can differ across those cohorts; this is not a matched experiment proving a causal effect of JPEG quality.

**MEASURED Cover-versus-kind weighted AUC:** JMiPOD 0.90974, JUNIWARD 0.61846, UERD 0.93095. Each comparison has 30,000 files from the same 15,000 families. The pooled score therefore should not stand in for equally effective detection of every embedding method. The decision is to retain method and quality slices alongside the pooled score, not to tune blend weights on this holdout.

**MEASURED operating points:** TPR is 0.31407 at FPR 0.001 and 0.37711 at FPR 0.01. Reaching TPR 0.8 requires FPR 0.4932 in this holdout. The weighted metric's low-TPR band is nearly saturated while substantial high-TPR shortfall remains. A broad detection-quality claim would conceal that behavior.

The exact metric and the legacy implementation both return **MEASURED 0.8354033190** on these predictions. Their known tie-fixture difference does not explain this model result. **MEASURED family-bootstrap 2.5/97.5 percentiles:** 0.83412–0.83663. This describes resampling of this fixed holdout, not training-seed stability, generalization across cameras or a leaderboard confidence interval. [Full cohort, ROC-region and bootstrap aggregates](../evidence/phase2/retained_evaluation.json).

## Interrupted training and independent reconstruction

The [baseline specification](../evidence/phase2/baseline_spec.json) declared the recipe before launch. **ESTIMATED before fitting:** smoke throughput selected five epochs under a three-hour overall cap. Those projections are not measured completion time.

**MEASURED:** the retained run has four completed epochs, ends `interrupted` after SIGTERM, and records an accepted continuation with matching source/package/runtime provenance. The fallback policy explicitly permits evaluating the last completed checkpoint while retaining the incomplete label. Its epoch losses describe training, not validation.

**MEASURED independent reconstruction:** all 60,000 holdout predictions and all 5,000 test predictions rebuild from the saved checkpoint with maximum probability delta 0.0. Holdout weighted-AUC delta is 0.0. Input/split/population, canonical membership/labels and exact checkpoint/prediction hashes pass verification. Saved score columns and metric metadata are checked against canonical probabilities and recomputed metrics. Replacing an input during verification is rejected.

The shell could not see CUDA, but the existing Docker GPU lane could. The initial container attempt exhausted its default shared memory and produced no saved prediction artifact. The corrected execution used explicit shared memory with source/data mounted read-only; both attempts remain in private logs. The maintained launcher also removes only its own container on timeout/failure. **MEASURED holdout inference time:** 362.45 seconds in that execution; this is not total training or end-to-end workflow time. [Execution receipt](../evidence/phase2/retained_execution.json).

No dedicated outage-timing receipt or complete cross-attempt `wall.json` was recovered from the original training run. Outage duration, total active training time and historical end-to-end budget compliance remain unverified. Independent model reload does not prove bit-for-bit replay of uninterrupted CUDA training, and does not establish remote artifact delivery.

## Completion status

| Issue | Evidence-backed status | Remaining work |
| --- | --- | --- |
| [Readiness #2](https://github.com/cttsai1985/kaggle-alaska2-image-steganalysis/issues/2) | Complete; merged in PR #6 | None for readiness. |
| [Baseline #3](https://github.com/cttsai1985/kaggle-alaska2-image-steganalysis/issues/3) | Verified incomplete-run fallback, as permitted by acceptance | Intended training remains incomplete; do not relabel this result. |
| [Ensemble/delivery #4](https://github.com/cttsai1985/kaggle-alaska2-image-steganalysis/issues/4) | Verified single-model provenance and filesystem recovery | Learned stacking and real ensemble gains remain deferred without full OOF evidence; remote services are unverified. |
| [Walkthrough #5](https://github.com/cttsai1985/kaggle-alaska2-image-steganalysis/issues/5) | Merged walkthrough and measured Lab isolation; v0.1.0 release published and downloaded assets verified | Hosted Kaggle and direct IDE attachment remain unverified. |
| [Parent #1](https://github.com/cttsai1985/kaggle-alaska2-image-steganalysis/issues/1) | Closed after all phase deliveries merged and v0.1.0 release readback | Writeup alignment and open portfolio requirements are tracked separately. |

[Prediction/OOF audit, official sample ordering and controlled real-model transfer recovery](consumer-contract.md) · [Aggregate receipt](../evidence/phase3/provenance_delivery.json).

The [executed walkthrough](../notebooks/01-method-and-rerun.ipynb) retains seven executed code cells, a cohort figure and an explicitly labeled CPU fixture. [Notebook execution](../evidence/phase4/notebook_execution.json), [actual Lab authentication/mount checks](../evidence/phase4/lab_validation.json) and [local offline-entry execution](../evidence/phase4/offline_entry.json) have separate receipts. These are engineering checks, not additional real-model evaluations.

## Additional fixed-model analysis

The [readout comparison](modeling-study.md) tests union probability against the
largest stego-class probability on the same complete holdout. The
[hard-case analysis](error-analysis.md) retains all evaluation rows and separates
confidence failures from weak ranking cohorts. Their hash-bound aggregate receipts
retain the original analysis source; these are additional diagnostics of the
same incomplete-run fallback, not newly trained models or leaderboard results.

The [portfolio audit](requirements-audit.md) distinguishes delivered writeup and
batch evidence from outstanding serving, canary and production requirements.
