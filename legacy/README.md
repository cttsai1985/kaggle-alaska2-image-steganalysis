# Legacy solution (original baseline `e00f0b8`)

This directory holds the original ALASKA2 solution by Chia-Ta Tsai, moved here unchanged in phase 2
(pure `git mv`, 100 % rename similarity; file contents are byte-identical to `e00f0b8`). The repository
[LICENSE](../LICENSE) covers these files. Competition: <https://www.kaggle.com/c/alaska2-image-steganalysis>.

| Path | Original role |
| --- | --- |
| `script/trainer.py`, `script/starter.py` | training / inference (overlapping implementations, direct `.cuda()`) |
| `script/stacker.py`, `script/meta_reader.py` | cached-probability joins and meta-model fitting |
| `alaska_utils/data_utils.py` | dedup-then-expand family split, submission (`1 - P(Cover)`) |
| `alaska_utils/eval_metrics.py` | competition weighted AUC (used as a comparison reference in `tests/test_metrics.py`) |
| `configs/*.py` | per-backbone training configs (flip/rot90 augmentations, `Normalize`) |
| `docker/`, `0*_*.sh` | unpinned container build and data download (`--force`) scripts |

These files are kept for provenance and comparison. They are not run, linted or packaged; the maintained
code is `src/alaska2/` (see `docs/phase2-package.md`).
