# ALASKA2: Detecting Hidden Data in Images

An image can look ordinary while carrying an embedded message. The [ALASKA2 challenge](https://www.kaggle.com/c/alaska2-image-steganalysis) asks a model to distinguish an unchanged JPEG from versions produced by different steganography methods. This solution uses an ImageNet-pretrained EfficientNet and keeps every source image’s variants together during validation.

The first finding concerns the evaluation unit. **MEASURED:** a diagnostic split of individual files puts siblings of **59,460 of 60,000 validation images** in training. The original solution already avoided that boundary mistake by splitting source images before expanding their variants. The maintained pipeline preserves that design and also checks byte-identical images across families. This is evidence about split integrity, not a measured increase in model accuracy.

The retained EfficientNet checkpoint reaches **MEASURED weighted AUC 0.83540** on the complete family holdout. Detection is weaker for JUNIWARD and the quality-95 cohort than the pooled score suggests. This is a verified fallback from incomplete training: four epochs completed against a declared five-epoch recipe. [Recorded results and limits](docs/results.md).

[Problem and value](docs/problem-and-value.md) · [Results](docs/results.md) · [Run the solution](docs/reproduction.md)

## Results

| Experiment | Evaluation scope | Recorded outcome |
| --- | --- | --- |
| Historical solution | Competition submissions | Scores and submission provenance have not been recovered for this write-up. |
| Retained EfficientNet checkpoint | Complete single family holdout; incomplete training | **MEASURED weighted AUC 0.83540; AUC 0.77024**. |
| CUDA smoke | Explicit family subset | Readiness and checkpoint reload passed; not a model result. |
| CPU fixture | Synthetic images | Training, reload and recovery contracts tested; not a model result. |

No historical or modern leaderboard score is established by these receipts. A holdout score, once available, will remain separate from a scored competition submission. [Evidence and completion status](docs/results.md).

## Where the pooled result hides weakness

**MEASURED:** quality-cohort weighted AUC is **0.86666 / 0.84809 / 0.76598** at quality 75 / 90 / 95. Each cohort contains 20,000 files from 5,000 source families. Cover-versus-JUNIWARD weighted AUC is **0.61846**, versus **0.90974** for JMiPOD and **0.93095** for UERD.

Equal cohort sizes do not make the model equally effective across methods. The report keeps these slices visible instead of choosing weights against the reported holdout. Quality cohorts contain different source families, so their gap does not establish a causal compression effect. [Cohorts, operating points and limits](docs/results.md#where-the-pooled-score-hides-weakness).

## Keep source images together

The learning target belongs to a file, but the independence boundary is its source image. Cover and corresponding JMiPOD, JUNIWARD and UERD variants share image content. Randomly separating those files can put near-identical content on both sides of validation.

**MEASURED:** the retained family assignments have no families spanning folds. The audit also found a byte-identical Cover pair under different family names; both belong to the same fold. The pipeline checks both family membership and byte identity before fitting. Filename grouping alone would not detect that second case. [Split audit](evidence/phase1/split_audit.json) · [Dataset audit](evidence/phase1/dataset_audit.json).

That establishes the intended partition on this snapshot. It does not establish generalization to other cameras or embedding methods, and no matched model comparison has measured the effect of the diagnostic leaky split.

## Model and metric

EfficientNet predicts the Cover, JMiPOD, JUNIWARD and UERD classes; the competition probability is `1 - P(Cover)`. Training uses flips and right-angle rotations that permute pixels without interpolation. The native image dimensions, decoder, normalization, pretrained-weight revision and class order travel with the run.

The competition uses weighted AUC. Tests derived from the metric definition exposed a difference in the historical implementation when tied scores cross a weighting boundary. **MEASURED on an all-tied fixture:** the exact implementation returns **0.5857**, while the legacy implementation returns **0.5714**. These are metric-fixture values, not model scores. Continuous random-score fixtures agree between implementations; they also agree on the retained model’s predictions, so their fixture difference does not explain this score. [Metric tests](tests/test_metrics.py) · [Implementation decisions](docs/phase2-package.md).

## A checkpoint is not a completed evaluation

The interrupted baseline retains checkpoint hashes, completed-epoch statistics and an accepted continuation from the last completed epoch. The continuation records the same source, package and runtime identity. That supports a narrow operational decision: preserve valid training state and reject a resume under changed provenance.

**MEASURED:** independent checkpoint reload reconstructs all 60,000 holdout predictions and all 5,000 test predictions with maximum probability delta 0.0. Holdout metric delta is 0.0. The training manifest remains interrupted: successful evaluation preserves that incomplete status, and does not establish artifact delivery. CPU fixture tests show deterministic resumed weights; that training-replay equivalence has not been demonstrated for this interrupted CUDA run. [Verified fallback receipt](evidence/phase2/retained_evaluation.json) · [Recovery limits](docs/results.md#interrupted-training).

## Reproduce

The [reproduction guide](docs/reproduction.md) separates public CPU contract tests from authorized competition-data and CUDA execution. It includes the supported CLI, input manifests, resource checks and saved-checkpoint verification commands.

An executed learning notebook, Docker JupyterLab, dev container and hosted Kaggle walkthrough remain outstanding in [the delivery issue](https://github.com/cttsai1985/kaggle-alaska2-image-steganalysis/issues/5). Ensemble-input provenance and recoverable artifact delivery remain outstanding in [the ensemble issue](https://github.com/cttsai1985/kaggle-alaska2-image-steganalysis/issues/4). The package does not yet establish a verified modern ensemble or a deployed detection service.

The maintained implementation is in `src/alaska2/`; experiment recipes are in `configs/`, aggregate evidence in `evidence/`, and contract tests in `tests/`. The original solution and attribution are preserved in [legacy/](legacy/README.md). The [readiness report](docs/phase1-readiness.md) and [package report](docs/phase2-package.md) retain the detailed engineering record.

## License

Source code retains the [Apache License 2.0](LICENSE). Competition data follows its own terms.
