# Align ALASKA2 with the competition portfolio

Standard practice: a solution-first narrative with paired leaderboard scores, source-backed Top %, a controlled modeling comparison, hard-case analysis and a usable prediction path. We differ because ALASKA2 already has an evaluated incomplete-run fallback and must preserve that scope while adopting the existing portfolio convention.

## Scope
- Presentation/evidence: README.md; docs/{results,problem-and-value,modeling-study,error-analysis,requirements-audit,portfolio-standard,serving,reproduction}.md; docs/{leaderboard-context,historical-submissions,public-profile,modeling-study,error-analysis,canary-validation}.json; docs/figures/; notebooks/01-method-and-rerun.ipynb.
- Reusable analysis: tools/analysis/{modeling_study,error_analysis}.py; tools/notebooks/build_walkthrough.py; scripts/{check_portfolio,analyze_hard_cases}.py (thin/shared entry points only).
- Actual model interface: src/alaska2/serving.py; tests/test_serving.py; tests/test_modeling_study.py; tests/test_portfolio.py.
- Environment/release: pyproject.toml, uv.lock, src/alaska2/__init__.py, Dockerfile, .github/workflows/ci.yml.
- Metadata after merge: rename GitHub repository to kaggle-alaska2-solution, update local origin and reviewed public description; retain local checkout/mount/service identities. Publish a corrected canary from merged source and verify downloaded assets. Preserve previous release bytes/history.

## Sequence and proof
1. Recover paired historical submissions through authenticated read-only Kaggle API. Use dated public leaderboard export, exact Decimal comparisons, unique scored teams and tie intervals; never assign a percentile to holdout/private populations.
2. Compare fixed union-of-stego probability vs maximum stego-class probability on identical saved holdout predictions. Declare before measuring; no refit, learned weights, threshold tuning, row removal or default change. Analyze fixed hard cohorts and input→representation→prediction failures with aggregate-only output.
3. Put leaderboard table immediately after short opening, then modeling decision/error insight and inference. Keep hardware/provenance mechanics in runbook. Map all R01–R26 honestly; preserve open production requirements.
4. Add and verify actual model→MLflow package→fresh-workspace HTTP parity on private real inputs, with invalid-request rejection and cleanup; no shared-service deployment or retraining.
5. Adopt shared scaffold/checker from infra; execute notebook and relevant tests, obtain review and current-source CI, merge, rename coherently, release and read back artifacts. Remove only merged/unoccupied branches and worktrees.

## Boundaries
Keep raw rows/images/checkpoints/credentials/private logs outside Git. Preserve original source, weak results, incomplete training and historical receipts. No new Kaggle submission or training/model search. CPU analysis and bounded private-input inference only; shared live services/volumes/ports/credentials remain untouched.
