# Portfolio requirements audit

Standard: [version 1.1](portfolio-standard.md). This PR aligns the **writeup**
with the reproduction portfolio. It does not establish complete production or
portfolio acceptance: the open requirements below remain open. Historical model
and release receipts retain their original source scope.

| ID | Priority | Status | Evidence / applicability |
| --- | --- | --- | --- |
| R01 | P0 | Mapped | This matrix retains every shared requirement; open items remain visible. |
| R02 | P0 | Delivered in scope | [README](../README.md): problem, measured decisions and executable batch path. |
| R03 | P0 | Partial | [Paired scores and public reference](leaderboard-context.json) appear in README; GitHub About alignment remains open. |
| R04 | P0 | Delivered in scope | [Results](results.md) distinguish weighted-AUC direction, historical submissions and family holdout. |
| R05 | P0 | Delivered in scope | [Reference evidence](leaderboard-context.json): score-equivalent context, ties and date; no awarded/private rank claim. |
| R06 | P0 | Delivered in scope | [Controlled readout comparison](modeling-study.md), full population and fixed-model variation. |
| R07 | P0 | Delivered in scope | [Hard-case analysis](error-analysis.md), representation boundary and competing explanations. |
| R08 | P0 | Partial | [Evaluation](results.md): family/byte separation and exact reconstruction; full CV and training-seed sensitivity unperformed. |
| R09 | P0 | Delivered in scope | [Diagnostic](error-analysis.json) retains all rows; no refit, threshold tuning or promotion. |
| R10 | P0 | Delivered in scope | [Retained execution](../evidence/phase2/retained_execution.json) and independent checkpoint reconstruction. |
| R11 | P0 | Open | Native-to-MLflow-to-fresh-workspace HTTP parity and invalid-request tests are not delivered. |
| R12 | P0 | Delivered in scope | [Batch input/output contract](serving.md): native JPEG decoding, normalization and class order. |
| R13 | P0 | Partial | [Runbook](reproduction.md), executed local notebook/container/offline paths; hosted Kaggle execution unverified. |
| R14 | P0 | Open | Shared-platform integration for online serving remains unverified; this update adds no services. |
| R15 | P0 | Open | Online deployment UX and lifecycle verification remain outstanding. |
| R16 | P0 | Preserved | No service, volume, port, credential or repository rename is performed in this writeup update. |
| R17 | P0 | Delivered in scope | Public analysis receipts contain aggregate results and hashes; private rows/images/checkpoints remain external. |
| R18 | P0 | Partial | Analysis receipts retain original source/input hashes; they are historical executions, not reruns of this PR. New CI verifies code/contracts. |
| R19 | P0 | Open | Existing v0.1.0 release remains intact. Coordinated corrected canary release is not delivered by this PR. |
| R20 | P0 | Delivered in scope | README preserves Chia-Ta Tsai attribution, original solution and archived source; opens with the JPEG task. |
| R21 | P0 | Delivered in scope | [Problem/value](problem-and-value.md) links the competition and separates proposed benefits from measured detection. |
| R22 | P1 | Partial | Batch prediction/recovery exists; production research/online serving remains open. |
| R23 | P1 | Open | Hosting, ownership, SLOs, auth/TLS, capacity, monitoring and off-host restore/rollout evidence remain outstanding. |
| R24 | P1 | Delivered in recorded scope | [Filesystem delivery/recovery](consumer-contract.md); remote-service transfer remains unverified. |
| R25 | P1 | Delivered in scope | [Error slices and next hypothesis](error-analysis.md), reusing saved holdout predictions without training. |
| R26 | P0 | Partial | Existing repository/package/release identities are retained; verified PR/worktree cleanup follows merge. Rename is not part of this update. |
