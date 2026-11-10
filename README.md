# ALASKA2: My Image Steganalysis Solution

My solution to the [Kaggle ALASKA2 competition](https://www.kaggle.com/competitions/alaska2-image-steganalysis), by **Chia-Ta Tsai**. The task is to predict whether a JPEG contains a hidden message, even when it looks unchanged. The original solution trains image classifiers, combines predictions with test-time augmentation and stacking, and exports competition submissions.

## Results

**MEASURED competition scores — weighted AUC ↑.** Public and private scores in each row come from the same historical submission.

| Model / submission | Public LB | Private LB | Public Top % |
| --- | ---: | ---: | ---: |
| Historical LightGBM stacker (2020-07-20) | 0.932 | 0.916 | 7.31% |
| Historical TTA/LightGBM stacker (2020-07-18) | 0.928 | 0.917 | 10.96% |
| Historical rank-averaged ensemble (2020-07-20) | 0.930 | 0.916 | 9.04% |
| Maintained EfficientNet baseline | Not submitted | Not submitted | Not applicable |

Public Top % is a **MEASURED score-equivalent comparison against the 2026-11-10 public leaderboard snapshot**, using the first tied position; it is not an awarded final rank. Private rank is unavailable. [Paired submission records](docs/historical-submissions.json) · [Reference population, ties and calculation](docs/leaderboard-context.json).

The maintained single-model baseline reaches **MEASURED holdout weighted AUC 0.83540** after four completed epochs of a planned five. This is a separate, incomplete-run fallback; it does not reproduce the historical ensemble’s leaderboard score. [Full results and limitations](docs/results.md).

## How my solution works

1. **Split by source image.** Keep each Cover image and its JMiPOD, JUNIWARD and UERD variants together, so shared content cannot cross the training/validation boundary.
2. **Train image classifiers.** Predict the four classes with pretrained convolutional models. Flips and right-angle rotations preserve the pixel grid without interpolation.
3. **Predict hidden-message probability.** Convert four-class predictions to `1 - P(Cover)`. The historical workflow also combines cached predictions using augmentation and stackers.
4. **Export predictions.** Produce a probability for each test image in the competition sample order.

The [original source](legacy/README.md) retains my training, inference, stacking and submission code. The maintained implementation lives in `src/alaska2/`, with recipes in `configs/` and regression checks in `tests/`. Historical stacking results belong to the recorded submissions; no new ensemble improvement is claimed.

## What the current evaluation shows

**MEASURED on the retained family holdout:** Cover-versus-JUNIWARD weighted AUC is **0.61846**, compared with **0.90974** for JMiPOD and **0.93095** for UERD. The pooled result hides a substantial weakness on JUNIWARD.

**MEASURED quality-cohort weighted AUC:** **0.86666 / 0.84809 / 0.76598** for JPEG quality **75 / 90 / 95**. These cohorts contain different source images, so the gap does not establish that compression causes the difference.

![Measured holdout performance across JPEG-quality cohorts and embedding methods](docs/figures/quality-and-method.png)

The original source-image split is preserved, with additional byte-identity checks. Independent checkpoint reload reproduces the saved holdout and test predictions. These checks support the retained baseline’s reproducibility; they do not establish a complete historical OOF stack or a completed modern training recipe. [Evaluation protocol, operating points and recovery limits](docs/results.md).

## Run the solution

- [Executed walkthrough](notebooks/01-method-and-rerun.ipynb): method, evaluation and a labeled CPU example.
- [Reproduction guide](docs/reproduction.md): locked local environment, Docker/Jupyter/dev container, competition inputs, training and saved-checkpoint prediction.
- [Offline Kaggle entry](docs/kaggle.md): packaging and local execution; hosted execution remains unverified.
- [Prediction and delivery contract](docs/consumer-contract.md): sample ordering, provenance and recoverable filesystem handoff.

Competition data and checkpoints are not bundled. Public synthetic fixtures verify execution contracts; their outputs are not competition results. The [engineering reports](docs/phase2-package.md) retain the implementation details, while [problem and value](docs/problem-and-value.md) explains the detection task and its limits.

## License

Source code retains the [Apache License 2.0](LICENSE). Competition data follows its own terms.
