# Phase 2, part A — modern package, legacy preservation and CPU evaluation contracts

Issue: #3 (parent #1). Builds on phase 1 (#2, PR #6). **No GPU training was run in this part.** The only GPU
use was a one-batch bf16 forward/backward availability check (`evidence/phase2/cuda_check.json`). The CUDA
smoke and the bounded real-data baseline are part B. Each one starts only after a fresh
`tools.readiness.launch_gate` returns GO.

Standard practice: a single maintained `src/` package with a locked environment, versioned configs, one
training loop, and contract tests that run without a GPU. The evaluation metric is checked against an
independent implementation before any score is reported. We differ in one respect. The original code is kept
byte-identical under `legacy/` and used as a comparison oracle in tests. It is not deleted. Risks the operator
would care about: the GPU is shared with a production container, so the package never chooses a device on its
own. `device = "cuda"` fails loudly when CUDA is missing and never falls back to CPU. The pretrained weights come
from the Hugging Face hub, pinned by commit and sha256, so a changed upstream file is rejected rather than used.

## 1. Layout

| Path | Content |
| --- | --- |
| `legacy/` | original `script/`, `alaska_utils/`, `configs/`, `docker/`, `0*_*.sh` — pure `git mv`, 100 % rename similarity, contents unchanged ([legacy/README.md](../legacy/README.md)); LICENSE stays at the root |
| `src/alaska2/` | `data`, `splits`, `metrics`, `model`, `train`, `evaluate` (evaluate / predict / verify), `inputs`, `identity`, `runtime`, `config`, `cli` |
| `configs/` | `baseline_effb0_fold1.toml` (phase-1 plan), `smoke_effb0.toml` (`extends` baseline), `fixture_cpu.toml` (tests) |
| `tests/` | `test_metrics.py`, `test_contracts.py`, `test_e2e_cpu.py`, `test_cuda.py` (marker `cuda`), phase-1 `test_readiness.py` |
| `evidence/phase2/` | `validate_real_data.json`, `cuda_check.json`, `cpu_contracts.json` (aggregates only) |

CLI: `uv run python -m alaska2 validate|train|evaluate|predict|verify` (also installed as `alaska2`).

## 2. Locked environment

`uv lock` resolves `torch 2.14.1+cu126`, `torchvision 0.29.1+cu126` (explicit PyTorch cu126 index), `timm 1.0.31`,
`opencv-python-headless 5.0.0`, `numpy 2.5.4`, `polars 2.1.0`, Python 3.12.13. We chose CUDA 12.6 wheels because
they are cu12x as planned, and the 595.71 driver (CUDA 13.2 driver API) runs older CUDA runtimes. On this host,
`uv run python -c "import torch;print(torch.cuda.is_available())"` prints `True`, with device `NVIDIA RTX A4500`,
capability 8.6 and cuDNN 91002. `.venv` takes 7.0 GB. Free disk on `/` was 123 GB at the start.

## 3. Data contract (`alaska2.data`, `alaska2.inputs`)

- **Manifest-driven.** Rows come from the phase-1 `files_manifest.parquet`. No directory globbing. The content
  hash is recomputed with the phase-1 definition and must equal `ab90caf0…c274`. On real data it does (`validate_real_data.json`).
- **Deterministic order.** Rows are ordered by `(family_id, label)`. The test `test_records_are_deterministic_and_in_class_order`
  shuffles the manifest and gets the same frame back.
- **Class order.** `Cover, JMiPOD, JUNIWARD, UERD`, the legacy order, so label = index. The config validator rejects
  any other order, and records are rejected if the manifest label disagrees with the order.
- **Decoding.** OpenCV `imread` then BGR→RGB, as in legacy `trainer.py:404`. The phase-1 probe measured 491 img/s
  for OpenCV and 192 img/s for Pillow on one process. Pillow stays available as `decoder = "pillow"`. The two
  decoders use different IDCTs and agree within 2 grey levels on a test image, so the decoder and its version are
  recorded with every prediction file.
- **No resize.** All 305,000 files are 512×512 RGB (phase-1 manifest). `check_image_size` rejects other sizes, and
  the Dataset asserts the decoded shape. The legacy `Resize(512, 512)` was a no-op on this data and is dropped.
- **Normalization.** `(x / 255 − mean) / std` with ImageNet mean `(0.485, 0.456, 0.406)` and std
  `(0.229, 0.224, 0.225)`. This is the same math as the legacy albumentations `Normalize` defaults (tested) and
  matches the timm `efficientnet_b0.ra_in1k` pretrained config (recorded per run). On 64 real images the
  normalized tensors have mean 0.0019 and std 1.0205.

### Augmentation: flips and rot90 only

Training uses the legacy recipe: `HorizontalFlip(p=.5)`, `VerticalFlip(p=.5)`, `RandomRotate90(p=.5, k ∈ 0..3)`,
applied in that order. Validation applies no transform. Why these are suitable for steganalysis:

- They are elements of the square's symmetry group D4. They move pixels and never change a value, so there is no
  interpolation, no colour change and no recompression. Low-amplitude embedding changes survive exactly. Tests
  check that the multiset of pixel values is unchanged and that all 8 symmetries occur.
- On a 512×512 image (a multiple of 8) they map the 8×8 JPEG block grid onto itself. The test fills each 8×8 block
  with one constant value and checks that every output block is still constant.
- Anything that resamples or changes values is rejected by the config validator. That covers resize, crop at an
  offset, colour jitter, blur and JPEG re-encoding, because each one would change or destroy the stego signal.

Augmentation randomness is a pure function of `(seed, epoch, row)`. It does not depend on the worker count, the
worker order or the resume point. The shuffle is `torch.Generator(seed + epoch)`. The DataLoader also gets its own
`torch.Generator(seed + epoch)`, so drawing the worker base seed never consumes the global CPU RNG. Without it, a
resumed persistent-worker loader would take one extra draw and change later CPU dropout masks.

## 4. Split contract (`alaska2.splits`)

- Folds are **loaded, never recomputed**, from phase-1 `legacy_family_folds.parquet`. The phase-1 content hash
  `7260e8bd…7d02` is required. A mismatch raises `IdentityError`.
- `role = valid` for `holdout_fold = 1` and `fit` for folds 2–5. Every train row must get exactly one fold (an `m:1` join with a row-count check).
- Before any fit, `assert_family_separation` checks that **0 families** and **0 byte-identical files** (the
  content-linked case from phase 1) appear in both roles. Real data passes: fit is 240,000 rows / 60,000 families
  and valid is 60,000 rows / 15,000 families. Each kind has 15,000 holdout rows, each quality 75/90/95 has 20,000,
  and all 15,000 holdout families are complete.
- An optional family subset (`max_fit_families`, `max_valid_families`) is deterministic, chosen by
  sha256(seed:family). It is recorded as `population = "explicit_family_subset"` and never reported as full data.
  The smoke uses 400 holdout families.
- A `population_sha256` over `(relpath, role)` identifies the exact fit/valid membership of every run.

## 5. Metric contract (`alaska2.metrics`)

The competition weighted AUC is implemented from the published definition. The ROC is split into TPR bands
`[0, 0.4]` and `[0.4, 1]`, weighted 2 and 1, and normalized by 1.4. The implementation does not use sklearn's ROC
code. It builds exact ROC vertices, where a tie group becomes a diagonal segment, and integrates
`clip(TPR, lo, hi) − lo` over FPR in closed form on each linear segment.

| Test | Expected (derived by hand) |
| --- | --- |
| perfect ranking | 1.0; band areas 0.4 / 0.6 |
| reversed | 0.0 |
| all tied (diagonal ROC) | (2·0.32 + 0.18) / 1.4 = **0.5857**, not 0.5; plain AUC 0.5 |
| ROC vertex exactly at TPR 0.4 | 0.8 / 1.4; legacy agrees |
| tie segment crossing TPR 0.4 | band areas 0.38 / 0.38 → 1.14 / 1.4 |
| legacy `eval_metrics.py`, continuous random scores (n = 1k / 20k / 60k) | equal within 1e-12 |
| sklearn `roc_auc_score` vs our one-band AUC | equal within 1e-12 (also with ties) |
| monotone transform / row permutation | unchanged |
| `score = 1 − P(Cover)` from 4-class softmax | = sum of stego probabilities; raising the Cover logit lowers the score, raising a stego logit raises it |

**Finding: the legacy metric differs from the definition when scores are tied.** The legacy code (copied from
the Kaggle notebook it cites) keeps only the ROC points inside each band and pads to the band edge. It does not
interpolate the segment that crosses TPR 0.4. With continuous scores the ROC is a staircase, and the two
implementations agree exactly. With ties, a diagonal segment can cross the edge. All-tied scores give 0.5714
(legacy) against 0.5857 (exact). Scores rounded to one decimal on 20,000 rows differ by less than 1e-3. Consequence:
reported scores use the exact implementation. Softmax outputs are effectively continuous, so the two should
agree on real predictions, and phase 2B will report both.

`evaluation_report` adds the pooled weighted AUC with per-band contributions, plain AUC, cohorts by quality and
Cover-vs-each-kind cohorts. Every cohort carries its row, family, positive and negative counts.

## 6. Model and training (`alaska2.model`, `alaska2.train`)

| Item | Value | Legacy | Why changed |
| --- | --- | --- | --- |
| Framework | plain PyTorch loop | PyTorch Lightning + Catalyst sampler | overlapping legacy implementations; obsolete APIs are not ported |
| Backbone | timm `efficientnet_b0`, `timm/efficientnet_b0.ra_in1k` @ `1b5383e5…f26a`, sha256 `d5698997…a2cd` | `efficientnet_pytorch` b2/b5/b6, timm others | one modest backbone (phase-1 plan); weights pinned and verified (the loaded backbone tensors equal the pinned file) |
| Head / loss | 4 classes; CE with `label_smoothing = 0.05` | `LabelSmoothing(.05)` | same formula (tested within 1e-6) |
| Dropout | 0.2 | EfficientNet-PyTorch b0 default 0.2 | kept |
| Optimizer / schedule | AdamW 5e-4, wd 1e-2, 500 warmup steps, cosine | AdamW + `ReduceLROnPlateau` (8 configs) or `OneCycleLR` (7) | plateau schedule and `ModelCheckpoint(top-3)` / `EarlyStopping` read **holdout labels**; fixed budget instead |
| Sampler | seeded epoch permutation | `BalanceClassSampler(downsampling)` | classes are balanced (25 % each), so downsampling is a no-op |
| Precision / device | bf16 autocast, channels_last, explicit `train.device` | `.cuda()` calls | explicit device, no silent fallback |
| Checkpoint policy | **last completed epoch** | best-of-3 by holdout weighted AUC | no epoch selection on holdout labels |

Each epoch atomically writes `checkpoints/epoch_NNN.pt` and `last.pt`. Each file holds the model, optimizer,
scheduler, GradScaler and RNG states (Python, NumPy, torch, CUDA), the resolved config, all identities and the
`model_sha256`. That hash covers the weight content (sorted keys, dtype, shape and bytes), not the pickle layout.
Checkpoints load with `weights_only=True`.

`run_manifest.json` holds: run_id, purpose and result label, status, source git sha and dirty flag, config sha256,
input-manifest sha256, split-folds sha256, population sha256, manifest-parquet sha256, pretrained weight identity,
runtime versions, device, the plan (epochs, steps, rows), per-epoch loss, img/s and peak VRAM, and each checkpoint's
sha256. A fresh run_id is required, and an existing run dir is refused. Failed and interrupted runs keep their
manifest, with `status = failed` or `interrupted`, the error and the traceback tail.

**Resume** (`train --resume RUN_DIR`) restarts from the last completed epoch. It is rejected when the config,
input-manifest, folds or population hash differs from the checkpoint. It is also rejected when the source or
runtime provenance differs. Each checkpoint stores `git_sha`, `dirty`, `package_sha256` (a hash of every
`alaska2/*.py`) and the runtime versions, so one run never mixes code revisions or environments. Each accepted
continuation records its own provenance in `run_manifest.resumes`. A labeled fault drill
(`train.fault_inject_after_epoch`) stops after epoch 1. Resuming then gives a `model_sha256` and `global_step`
**equal** to an uninterrupted 2-epoch run (CPU, fixture). The equality holds with 0 workers and with 2 persistent
workers at dropout 0.5. This tests **interrupted-training recovery only**.
Stage and artifact-delivery recovery is not tested here (phase 3).

## 7. Evaluate, predict, verify (`alaska2.evaluate`)

- `evaluate` writes `predictions/valid_epochNNN.parquet`. Each row has image, family_id, kind, label, fold, quality,
  relpath, logits, `p_Cover…p_UERD`, `score = 1 − p_Cover`, `model_sha256` and `checkpoint_sha256`. A JSON
  sidecar holds the metrics, identities, decoder, transforms, normalization, versions and the prediction file's
  sha256. Exact membership is asserted: the `(image, kind)` set must equal the declared holdout population, with no
  duplicates.
- `predict` does the same for `Test/` with no metrics and **no submission file**.
- `evaluate` and `verify` use the checkpoint's own config. Only runtime keys can change: data root, device,
  workers and eval batch. Inputs whose identities differ from training are rejected. The test swaps folds 1 and 2
  and the evaluation is refused.
- `verify` rebuilds the model from the checkpoint alone and checks the weight hash. Before inference it asserts
  that the saved `(image, kind)` set equals the declared population exactly (`role = valid`, or the full `Test/`
  set). It also rejects any saved label, fold, family_id or quality that differs from the canonical records.
  Metrics are rebuilt from the canonical labels. Fit-row predictions, missing holdout rows, flipped labels and a
  shifted fold are each rejected in a test. It then re-runs inference on exactly the saved rows in saved order and
  compares against the declared tolerance:
  `prob_atol = 1e-3` and `metric_atol = 1e-4` for the GPU baseline, `1e-6` and `1e-9` for the CPU fixture. It also
  checks the checkpoint and prediction file hashes. Tampering with `p_Cover` is detected in a test.
- Fixture CLI run (CPU, **FIXTURE_NOT_A_MODEL_RESULT**): train (2 epochs) plus evaluate took 9.5 s. Verify passed
  with observed max |Δp| = 0.0, Δ weighted AUC = 0.0 and holdout membership 16/16 rows (0 missing, 0 extra).

Result labels are attached to every manifest and prediction file: `FIXTURE_NOT_A_MODEL_RESULT`,
`SMOKE_READINESS_ONLY_NOT_A_MODEL_RESULT`, `FAULT_DRILL_NOT_A_MODEL_RESULT`, `REAL_DATA_BASELINE`. A fixture or
smoke number is never a model result.

## 8. Evidence (this part)

| File | Content |
| --- | --- |
| `validate_real_data.json` | `uv run python -m alaska2 validate --config configs/baseline_effb0_fold1.toml --rehash-sample 2000 --decode-sample 64`: manifest and folds hashes equal phase 1; 0 families or content groups cross roles; 2,000 files rehashed with 0 mismatches; 64 real images decoded to finite `3×512×512` float32 tensors |
| `cuda_check.json` | `torch.cuda.is_available() = True`; one-batch bf16 forward/backward test passed; GPU at 893 MiB / 3 % before and after |
| `cpu_contracts.json` | `pytest -m "not cuda"`: 89 passed (61 new + 28 phase 1); ruff check and format clean; fixture CLI run, resume drills and review fixes |

**Not measured yet (part B):** GPU training throughput, peak VRAM, the epoch count E fixed from the smoke
(`E = min(6, floor((10800 − 900) / epoch_s))`), and any real-data model score.

## Reproduce

```bash
uv sync
uv run ruff check src tests tools && uv run ruff format --check src tests tools
uv run pytest -q -m "not cuda"          # CPU contracts, fixture end-to-end (~30-100 s depending on host load)
uv run pytest -q -m cuda                # one-batch CUDA availability check (not training)
uv run python -m alaska2 validate --config configs/baseline_effb0_fold1.toml --rehash-sample 2000 \
  --decode-sample 64 --out evidence/phase2/validate_real_data.json
```
