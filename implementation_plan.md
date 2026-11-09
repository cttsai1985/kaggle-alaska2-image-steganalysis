# Finish the executed ALASKA2 walkthrough and delivery

Standard practice: one maintained package across CLI, notebook and container, with an executed teaching artifact and a release built from merged source. ALASKA2 differs because the retained real model is an explicitly incomplete-run fallback; preserve that scope everywhere and do not claim hosted Kaggle execution without evidence.

## Scope

- Environment: `.gitignore`, `.dockerignore`, `pyproject.toml`, `uv.lock`, `Dockerfile`, `compose.yml`, `compose.gpu.yml`, `docker/jupyter_server_config.py`, `.devcontainer/devcontainer.json`, `tools/lab.py`.
- Walkthrough: `tools/fixtures.py`, `tests/conftest.py`, `tools/notebooks/build_walkthrough.py`, `tools/notebooks/execute_walkthrough.py`, `notebooks/01-method-and-rerun.ipynb`, `notebooks/kaggle/rerun.ipynb`.
- Portable delivery: `tools/package_offline.py`, source/runtime metadata helpers only if a failing behavior test proves missing provenance in the standalone image.
- Gates: `tests/test_lab.py`, `tests/test_notebook.py`, `tests/test_release.py`, `.github/workflows/ci.yml`; extend existing contract tests only for directly changed source provenance.
- Documentation: `README.md`, `docs/reproduction.md`, `docs/results.md`, `docs/kaggle.md`, `docs/consumer-contract.md`; retain historical reports.
- Evidence: `evidence/phase4/notebook_execution.json`, `lab_validation.json`, `offline_bundle.json`, aggregate walkthrough figure(s). Release assets/checksums are produced outside Git from merged source.

## Steps and verification

1. Add locked optional notebook/release groups; use the existing shared caches or container layers, never a worktree data/artifact copy.
2. Build a pinned Python container and loopback-only Lab with nonempty token auth, read-only data, writable outputs, explicit optional GPU access. Run the real HTTP auth and mount checks.
3. Generate and execute the notebook top-to-bottom: measured family-split/model-cohort story, actual retained-model reconstruction and transfer story, then a CPU fixture train/evaluate/verify using the maintained package. Inspect saved figures and outputs.
4. Package a wheel/source/offline Kaggle path with identity/checksum checks. Hosted execution remains explicitly unverified unless it actually runs.
5. Run CPU CI including analysis/notebook contracts; obtain independent review, merge, then build source/lock/evidence/checksum release assets from clean master. Close #5 with live release evidence, then reconcile/close parent #1.

## Risks and limits

No new training search or Kaggle submission. Existing real-model results and filesystem-only delivery limits remain. Never expose Lab beyond loopback, publish private rows/checkpoints/tokens, change Docker credentials, deploy shared services or imply a production detector. Verify installed/runtime artifacts rather than configuration alone.
