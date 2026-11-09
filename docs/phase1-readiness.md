# Phase 1 — P620 readiness receipt, dataset manifest and resource plan

Issue: #2 (parent #1). Original baseline: `e00f0b8` (unchanged). No training was run in this phase.

**Verdict: conditional.** Data, GPU access and the split are ready. **Launch is NO-GO until the host-occupancy gate
(section 7) passes on a fresh reading.** The last gate reading (2026-11-09T16:45:07Z) was NO-GO on CPU and IO load
from other host tenants. GPU and disk passed. The phase-2 CUDA smoke (≤ 15 min) and the bounded baseline (≤ 3 h)
each start only after their own fresh gate returns GO.

All numbers below are quoted from the committed JSON under `evidence/phase1/`. Images, per-file rows, the manifest
parquet and the fold parquet stay outside Git under the data directory.

| Evidence file | Produced by |
| --- | --- |
| `host_receipt.json` | `uv run python -m tools.readiness.host_receipt` |
| `download_receipt.json` | download log, zip listing, `sha256sum`, `zipfile.testzip`, extracted-vs-listing compare |
| `dataset_audit.json` | `uv run python -m tools.readiness.build_manifest` |
| `split_audit.json` | `uv run python -m tools.readiness.legacy_split` |
| `decode_probe.json` | `uv run --group probe python -m tools.readiness.decode_probe` |
| `launch_gate.json` | `uv run python -m tools.readiness.launch_gate` (snapshot at review time; NO-GO) |

## 1. Host receipt (redacted)

Captured 2026-11-09T16:34:53Z. No env vars, tokens, usernames, hostnames, PIDs or GPU UUIDs are recorded.

| Item | Value |
| --- | --- |
| OS | Ubuntu 24.04.5 LTS on WSL2, kernel `6.18.54.1-microsoft-standard-WSL2` |
| CPU | AMD Ryzen Threadripper PRO 5965WX, 36 logical threads visible |
| RAM | 165.1 GiB total, 113.6 GiB available |
| Disk (`/`, same fs as data) | 129.8 GiB free of 392.7 GiB |
| GPU | 1 × NVIDIA RTX A4500, 20470 MiB; 890 MiB used, 2 % util at capture; driver 595.71, CUDA (driver API) 13.2 |
| Toolchain | Python 3.12.13 (uv-managed), uv 0.10.4, Docker 29.8.1 |
| Docker GPU | `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L` → exit 0, `GPU 0: NVIDIA RTX A4500` |
| Kaggle | CLI 2.2.5; competition file list OK (paginated, 20 rows on first page, more pages); quality dataset file list OK (1 file) |
| Host load at capture | loadavg 57.6 / 50.6 / 48.0; CPU pressure some avg60 = 30.8 %; IO pressure some avg60 = 10.5 % |

The host is shared. A production GPU container (`numerai-tournament-prod-gpu-worker-1`) is up but was not using the
GPU at capture. Per-process GPU attribution is hidden under WSL2, so `memory.used`/`utilization.gpu` is the
occupancy signal used by the launch gate.

**Failed attempts (kept visible).** The first Docker GPU check timed out at 180 s with its container stuck in
`Created`. A manual retry printed the GPU but did not exit within 170 s. Both ran during heavy host IO from the
concurrent 30 GB download (IO pressure "full" ≈ 72 %). The stuck container was ours (matching timestamp and command)
and we removed it. After the download finished, the check passed in 14.6 s and then in 1.4 s. The receipt tool now
names its container, keeps partial output on timeout and removes only that named container. Engineering consequence:
a GPU container start has no fixed latency on this host, so the phase-2 launch must allow for startup time.

**Kaggle file-list pagination.** The first receipt counted the CLI's page-token line as a file ("21 files"). The
listing is paginated: 20 rows per page, plus a token. The parser now records only data rows and a `more_pages`
flag, and never the token. File-list success shows access only. Download completeness is checked in section 2.

## 2. Download, extraction and dataset manifest identity

| Item | Value |
| --- | --- |
| Competition zip | 32,203,919,104 bytes (= Kaggle-reported size), sha256 `34022f655637be63fb32a7ea63cca4d44cbd1286e049f9f0bb90d106a85c6b20` |
| Zip integrity | `testzip` first bad member: none; 0 unsafe member paths; download log `exit=0` |
| Zip listing | 305,001 members: Cover/JMiPOD/JUNIWARD/UERD 75,000 each, Test 5,000, `sample_submission.csv`; 32,267,099,326 uncompressed bytes |
| Extraction | `unzip -q -n`, exit 0, 2026-11-09T16:08:02Z → 16:17:29Z |
| Extracted vs listing | 305,001 = 305,001 files, 32,267,099,326 = 32,267,099,326 bytes, 0 missing, 0 extra, 0 size mismatches → **exact match**; the zip was then deleted to preserve disk |
| Manifest content hash | `ab90caf00e5b21bde712cbb0fccd9aa5a6a1d41dff743934f186ae0806f5c274` (sha256 over sorted `relpath\tsha256` lines, 305,000 image files). Re-verified on review by `--reuse-manifest`, which now **rehashes every file**: 305,000 rehashed, 0 changed, same hash |
| Quality CSV | `image_quality.csv` sha256 `c826e8f3177e2866184abf6df515ad51a9ac0f0ddf49edf0b48bbd60f21d31a8`, 305,000 rows, 0 duplicate `(image, kind)` keys; the split's quality join is enforced 1:1 (`validate="1:1"` plus an unchanged row count: 300,000 train rows in, 300,000 out) |
| Sample submission | sha256 `e16c82f3…14be25`; columns `Id, Label`; 5,000 rows; 0 duplicate Ids; **Id set equals the 5,000 `Test/` file names** (0 missing either way); Label constant (1 unique value) |

Membership is derived from the real files, not from a numeric range or README counts:

- 305,000 JPEG files; 0 open errors; 0 non-JPEG; all 512×512 RGB.
- 75,000 train families, **all 75,000 complete with exactly one file of each of the 4 kinds**; 0 incomplete.
- Family IDs run from 1 to 80,005 with **5,005 gaps**. A contiguous range `1..75000` would be wrong.
- Test has 5,000 images: 0 Test names overlap train family IDs, and 0 Test files are byte-identical to any train file.
- Byte-identical groups: 244. Of these, 243 are a stego file identical to its own Cover. One is a **Cover identical to
  another family's Cover** (two family IDs, same bytes). Their stego variants differ.
- **Stego files byte-identical to their own Cover: 250** (JMiPOD 8, JUNIWARD 29, UERD 213), in 243 families.
  246 of the 250 are at quality 75. These rows have a stego label but no byte change. No model can separate them from
  the Cover, so they set a small floor on achievable error. They are kept in the data and counted per fold.

## 3. Observation unit: the source-image family

**The unit of independence is the source-image family:** one Cover plus its JMiPOD, JUNIWARD and UERD variants,
which share a filename stem. The three stego files are small embedding perturbations of the same Cover. They share
almost all content, the JPEG quality and the quantization tables. If one variant is in training and another is in
validation, the validation row measures memorized content, not steganalysis on an unseen image. The weighted AUC
also scores the Cover against its own stego variants, so the validation negatives and positives must come from
families the model has not seen.

Checked against the actual membership and the original code:

- The legacy splitter (`alaska_utils/data_utils.py::split_train_valid_data`) sorts by image name, keeps the first
  row per base image, runs `StratifiedKFold(n_splits=5)` (no shuffle) on `quality`, and expands each fold's families
  back to all four variants. `groups=df[image]` is passed but `StratifiedKFold` ignores it; family separation comes
  from **dedup before expansion**, not from `groups`. `script/trainer.py:732` and `script/stacker.py:271` use fold 1.
- Reproduced assignment: fold-assignment content sha256
  `7260e8bdb592cdd970d4cddace4321e68e8e1c81a085ab1ac7b223ca627b8d02`. 75,000 families, **0 families in more than one
  fold**, 0 rows without a fold. Each fold has 15,000 families = 60,000 rows (15,000 of each kind) and
  20,000 rows each of q75/q90/q95.
- The one cross-family identical-Cover pair is invisible to a filename-based split. In the legacy folds both
  families fall in the same fold (`of_which_split_across_folds = 0`), so no byte-identical content crosses folds.
  Engineering consequence: any new splitter must keep this content-linked check, not only a filename check.
- Stego-identical-to-Cover rows per legacy fold: 23 / 44 / 58 / 68 / 57. Fold 1 has 23 (JUNIWARD 6, UERD 17, all q75).

**DIAGNOSTIC ONLY, not an evaluation:** a row-level `StratifiedKFold(5, shuffle=True, random_state=0)` on label,
fold 1. Of 60,000 validation rows, **59,460 have a same-family sibling in training**. Of 45,000 validation stego
rows, **35,766 have their own Cover in training**. A row-level split would mostly score memorized images. It is
never used for a reported number.

Decision: keep the legacy family boundary (dedup → fold → expand). Assert 0 families across folds and 0
content-linked groups across folds before any fit.

## 4. JPEG-quality metadata: coverage and agreement

- Coverage: 305,000 / 305,000 files have a CSV row; 0 CSV rows without a file; join rows = file rows (1:1).
- Independent check: the IJG quality was recovered from each file's luminance quantization table
  (`tools/readiness/jpeg_quality.py`) for all 305,000 files. **CSV vs table: 305,000 compared, 0 mismatches.**
  Table quality: 75 → 101,667, 90 → 101,666, 95 → 101,667. Quality is constant within every family.
- Balance: train 25,000 per kind per quality; Test 1,667 / 1,666 / 1,667.

Consequences: the legacy quality-stratified split is reproducible exactly. Quality can also be derived from the
files, so the CSV is not a hard dependency, and the derived value must agree with it. A quality-cohort evaluation
is possible on the holdout (20,000 rows / 5,000 families per quality in fold 1). Quality is treated as an
evaluation axis. It is not assumed to be a useful predictive feature.

## 5. One testable evaluation question

**Is a fold-1 weighted-AUC result spread across the JPEG-quality cohorts (75/90/95), or concentrated in one?**

Cheapest adequate evidence: the bounded baseline's saved fold-1 held-out probabilities (60,000 rows), joined to the
manifest's table-derived quality. Per-cohort competition weighted AUC and plain AUC with counts, plus a
family-cluster bootstrap interval. This costs no extra GPU time. Alternative explanations to separate: cohort
difficulty; the 23 label-noise rows in fold 1 (all q75); and sampling noise at 5,000 families per cohort. The
weighted AUC (TPR 0–0.4 weighted 2, 0.4–1 weighted 1) will be verified independently in phase 2 before use.
Low-FPR behavior will not be called "measured" unless the ROC analysis supports it.

## 6. Loader sizing: measured CPU JPEG decode and read throughput

2,000 images, kind-balanced, seed 20261109; full decode to 512×512×3 uint8, 1 thread per process; warm cache.
Host loadavg ≈ 40–57 during the probe, so these are throughputs under contention.

| Backend | 1 proc | 4 | 8 | 16 workers (img/s) |
| --- | --- | --- | --- | --- |
| OpenCV 5.0.0 `imread` | 491 | 899 | 1,349 | 1,213 |
| Pillow 13.0.0 `convert("RGB")` | 192 | 631 | 953 | 1,058 |

Raw reads (disjoint sample, no decode): the first pass with 1 thread was **101 files/s (10.5 MB/s)** for files
that were not cached. With 16 threads it was 4,050 files/s. Cached second passes reached 4,900–8,300 files/s. An
earlier probe run during heavy IO pressure measured 22 MB/s for a first read. Decision: use OpenCV decode with
12 workers; never use single-threaded cold reads. The 32 GB dataset fits in page cache, given 113 GiB of available
RAM.

## 7. Resource plan (GO/NO-GO)

Standard practice: a fixed-budget, single-backbone baseline on a grouped holdout, smoke-tested on the GPU first,
with the budget decided before fitting, and launched only when the shared host is idle enough. We differ in one
respect: the legacy fold 1 is the holdout, not 5-fold CV.
That reproduces the original evaluation boundary within the 3 h budget. It is reported as a **single holdout, never
as CV**. Risks the operator would care about: the GPU is shared with a production container; host CPU load is high
(loadavg ~57 on 36 threads at the receipt) and would starve the loader, so the gate below blocks launch while it
lasts; GPU container startup latency varies (see section 1);
pretrained weights download from the Hugging Face hub needs network access.

| Item | Fixed value |
| --- | --- |
| Assigned GPU | GPU 0, NVIDIA RTX A4500 (the only GPU); measured free VRAM 19,580 MiB at capture |
| Launch gate (every launch) | `tools.readiness.launch_gate`, run fresh **immediately before the smoke and again before the baseline**; exit 0 = GO, exit 3 = NO-GO. All must pass: GPU other usage ≤ 2,048 MiB and util ≤ 20 %; CPU free threads (logical threads − loadavg-1) ≥ 16 (the job needs ~14: 12 loader workers + main + pin thread); CPU PSI `some` avg60 ≤ 10 %; IO PSI `some` avg60 ≤ 20 % and `full` avg60 ≤ 10 %; free disk ≥ 40 GiB floor + 5 GiB run outputs on `/` and the data fs. Missing PSI counts as failing. On NO-GO, wait (`--wait-max-s`, `--poll-s`) and never stop, renice or touch other jobs or containers. One job at a time. Thresholds are our proposal, not an operator ruling. |
| VRAM cap | ≤ 16 GB of our usage; the smoke records peak allocated/reserved; halve batch if peak reserved > 14 GB |
| Backbone | timm `efficientnet_b0`, ImageNet-pretrained (`efficientnet_b0.ra_in1k`), current timm API |
| Head / target | 4-class softmax in legacy class order (Cover, JMiPOD, JUNIWARD, UERD); score = 1 − p(Cover), as in legacy `generate_submission` |
| Input | 512×512 native RGB, no resize; train augmentation: random h/v flip + rot90 |
| Batch / precision | batch 32, AMP bf16 autocast, channels_last |
| Optimizer | AdamW lr 5e-4, weight decay 1e-2, cosine schedule, 500 warmup steps, seed 20261109 |
| Loader | OpenCV decode, 12 workers (cap 16), pinned memory, persistent workers |
| Split | **full data, single holdout = legacy fold 1**: train 240,000 rows / 60,000 families; valid 60,000 rows / 15,000 families. Not CV. Test/ is not used for selection; no Kaggle submission. |
| CUDA smoke | ≤ 15 min wall, separate run_id: ~200 train + ~50 valid steps on real data. Records img/s, peak VRAM, loss finiteness, and a checkpoint save + independent reload with logits equal on a fixed batch. Readiness only, not a model result. |
| Baseline | fresh run_id, `nohup` + log, **hard wall clock ≤ 3 h** (`timeout 10800`), one run. Epochs fixed before fitting from the smoke throughput: `E = min(6, floor((10800 s − 900 s margin) / measured epoch_s))`. If E < 2, NO-GO on full data; a subset run would be explicitly labeled as such. Checkpoint every epoch; held-out predictions are saved. |
| Second model | only if it fits an additional 2 h after the baseline wall time is measured; otherwise deferred and reported as deferred |
| Disk | 129.8 GiB free (floor 40 GB). Outputs go to `~/data/kaggle/alaska2-image-steganalysis/runs/<run_id>/`; b0 checkpoints are ~20 MB each |

**Gate state at review time: NO-GO** (`evidence/phase1/launch_gate.json`, two readings 75 s apart, ending
2026-11-09T16:45:07Z, after our own rehash job had finished):

| Criterion | Value | Limit | Pass |
| --- | --- | --- | --- |
| GPU other usage | 893 MiB | ≤ 2,048 MiB | yes |
| GPU util | 2 % | ≤ 20 % | yes |
| CPU free threads | 5.3 | ≥ 16 | **no** |
| CPU PSI some avg60 | 10.05 % | ≤ 10 % | **no** |
| IO PSI some avg60 | 19.3 % | ≤ 20 % | yes |
| IO PSI full avg60 | 16.2 % | ≤ 10 % | **no** |
| Free disk `/` (= data fs) | 122.4 GiB | ≥ 45 GiB | yes |

This load is not ours: no job from this run was active during the readings. **Phase 2 waits for a GO** from a fresh
gate run before the smoke, and again before the baseline, and records each gate JSON with that run. GPU, Docker GPU,
Kaggle access, complete data, quality metadata and the reproducible legacy split are all verified. **Not yet
measured:** GPU training throughput and peak VRAM (the smoke measures both), and the weighted-AUC implementation
(verified in phase 2).

## Reproduce

```bash
D=~/data/kaggle/alaska2-image-steganalysis
uv run python -m tools.readiness.build_manifest --data-dir $D/raw/extracted \
  --quality-csv $D/quality/image_quality.csv --sample-submission $D/raw/extracted/sample_submission.csv \
  --out-dir $D/manifest --audit-json evidence/phase1/dataset_audit.json   # add --reuse-manifest to rehash-verify
uv run python -m tools.readiness.legacy_split --manifest $D/manifest/files_manifest.parquet \
  --quality-csv $D/quality/image_quality.csv --out-dir $D/manifest --audit-json evidence/phase1/split_audit.json
uv run python -m tools.readiness.host_receipt --data-dir $D --out evidence/phase1/host_receipt.json
uv run --group probe python -m tools.readiness.decode_probe --data-dir $D/raw/extracted \
  --manifest $D/manifest/files_manifest.parquet --out evidence/phase1/decode_probe.json
uv run python -m tools.readiness.launch_gate --data-dir $D --label smoke --out evidence/phase1/launch_gate.json
uv run ruff check tools tests && uv run pytest -q
```
