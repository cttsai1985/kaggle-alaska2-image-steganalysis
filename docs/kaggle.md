# Offline Kaggle path

Hosted Kaggle execution is **unverified**. The [entry notebook](../notebooks/kaggle/rerun.ipynb)
provides a portable route to the same maintained CPU walkthrough. The locally executed
[learning notebook](../notebooks/01-method-and-rerun.ipynb) has saved outputs and an
[execution receipt](../evidence/phase4/notebook_execution.json); these do not prove hosted execution.

From clean committed source, run `python -m tools.package_offline --out /outside/repo/capsule`.
The capsule contains the committed source ZIP, lock, receipt and SHA256SUMS. It excludes untracked
runtime inputs and ignored secrets. It does **not** include Python, compiled dependencies, real
competition data, pretrained weights or the retained checkpoint.

1. Verify `sha256sum -c SHA256SUMS` before uploading the public source capsule as a Kaggle input.
2. Attach it to the entry notebook, set `INPUT` to its actual mount path and disable Internet.
3. Supply Python 3.12 and the exact dependency versions from `uv.lock`. The entry fails explicitly
   if Kaggle's installed versions differ. For an offline runtime, prepare a Linux/Python-3.12 wheelhouse
   or shared uv cache in an online environment using this lock, attach it separately, and run
   `uv sync --offline --locked --group lab --group analysis` against that cache from the extracted source directory. The entry selects `SOURCE/.venv/bin/python`
   when present and checks/executes with that interpreter, rather than requiring a kernel switch. The CUDA wheel index
   is pinned in the project even when the fixture uses CPU; do not silently substitute another build.
4. Run top to bottom and retain `execution.json`, `executed.ipynb` and the figure/HTML artifacts from
   `/kaggle/working/alaska2`. A hosted reproduction claim requires those actual saved outputs and
   a runtime/source readback; the checked-in entry has no hosted outputs.

The source checksum is an integrity check. Obtain the capsule and published checksums together from
the trusted repository release. No service credentials or submission API are used. A real-model
rerun requires separately authorized competition inputs and checkpoint provenance from the runbook.
