# Land recovered Claude ALASKA2 work

Standard practice: validate the existing stacked PRs and merge them in dependency order. This repository has no automated CI, so add a CPU regression check before landing.

## Scope

Existing implementation: the complete tracked diff of phase1/readiness and phase2/package against master. No changes to their Python implementation, configs, or historical evidence are planned.

New files: `.github/workflows/ci.yml`, `implementation_plan.md`.

## Steps

1. Locate Claude's transcript, PRs, commits, and recovery worktree; compare uncommitted recovery edits against the committed package branch.
2. Run Ruff and all non-CUDA tests against phase2/package. Add CI that repeats these checks using the locked environment.
3. Merge PR #6, retarget PR #7 to master, then merge #7 after CI passes. Preserve branch history with merge commits.
4. Fast-forward the clean main checkout and verify remote master includes both PR heads. Remove clean merged worktrees; preserve resume-ca123aa because its exact revision and dirty package content are needed for checkpoint provenance.

## Risks and boundaries

No new GPU jobs, external dataset mutations, or checkpoint changes. Existing baseline artifacts show an interrupted run after epoch 4 and no prediction artifacts; merging tooling does not constitute completion of the baseline or phases 3–4. CI downloads the locked CUDA wheel but runs only CPU tests.

## Verification

`uv run --locked ruff check src tests tools`
`uv run --locked ruff format --check src tests tools`
`CUDA_VISIBLE_DEVICES='' uv run --locked pytest -q -m 'not cuda'`

Success: both PRs merged to master; CPU regression workflow passes; the recovered baseline worktree remains intact.
