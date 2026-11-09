# Prediction provenance and artifact handoff

An evaluated checkpoint and a completed transfer answer different questions. The [real-model handoff receipt](../evidence/phase3/provenance_delivery.json) records both, with the training result still labeled incomplete.

## What a prediction must prove

Before export or composition, the package checks the actual producing checkpoint's weight/content hash, input/split/population identities, class order, canonical image/family/label/fold/quality membership and exact row ordering. Every row must name the same producing model and checkpoint as its bound independent-reconstruction receipt. Probabilities must be finite, normalized and within range; the competition score must equal `1 - P(Cover)`.

For labeled validation, the expected fold must be the producing model's held-out fold. Neither a cached filename nor a plausible score establishes that relationship. Missing proof is rejected rather than inferred.

**MEASURED:** the retained model covers 60,000 held-out files out of 300,000 labeled files, all from its producing model's declared holdout. This is a single holdout, not a full OOF matrix. The test prediction file covers the actual supplied test manifest; its CSV preserves all 5,000 sample IDs exactly once and in sample order. No Kaggle upload was performed.

## Why learned stacking remains deferred

The original `generate_stacking_data_split` concatenates cached probability frames and uses architecture/metric fragments from filenames without producing-fold/model proofs. `run_stacker` also has a path that fits and scores on the same validation rows. Those source paths cannot establish an independent stacker score. The [audit receipt](../evidence/phase3/provenance_delivery.json) pins the unchanged original source blob; historical caches were not supplied with verifiable provenance.

A fixed probability mean is implemented and independently checked on synthetic, verified-model fixtures. It requires distinct producing models on the same population and class order. Its output does not carry one constituent model's logits. No real ensemble gain or complementary-error finding is claimed: only one retained real model is available. Learned stacking needs leakage-safe OOF features and a separate/nested assessment; the current holdout is not reused to fit and report a meta-model.

## Stable identity and transfer attempts

The offline bundle binds the original logical run, input/split/config identities, retained model/checkpoint, predictions and CSV checksum. Its manifest identifies every member by relative path, size and SHA-256. Packaging rejects conflicting manifests, path escapes, symlinks and unowned output directories.

The filesystem destination owns the exclusive import claim and authoritative SQLite identity/attempt ledger. A caller cannot bypass that ownership by selecting another local state directory. That local directory records events only. A live importer cannot be taken over; after it stops, an incomplete attempt requires explicit recovery under the same immutable identity. Published destination bytes are read and checked again before an existing completion record is reused.

**MEASURED controlled real-model filesystem drill:** the bundle contains 10 verified files. Interrupted-copy, corrupt-copy and post-publish failures each retained the local scientific artifacts and recovered on transfer attempt 2 under the same experiment ID. A separate controlled quality rejection retained the experiment but returned a model/policy decision rather than a dependency retry. These are injected transfer conditions around a real retained model, not observed service outages or a production deployment.

The [receipt](../evidence/phase3/provenance_delivery.json) explicitly records `transport = filesystem` and `remote_service_verified = false`. No MLflow/S3 delivery, tracker availability or shared-service operation is claimed. An offline verified bundle remains valid without those services.

## Canonical platform contract proposal

Standard practice: declare shared-service names explicitly and validate the contract before provisioning. We differ from the original issue's central-registry assumption because the [current platform registry](https://github.com/cttsai1985/ml-infrastructure/blob/a6bd181adb9bf7c2b0e59e975cb6caadd643ef33/src/mlops_platform/registry.py) accepts a client-supplied contract rather than a per-project table.

The versioned [proposal](../configs/platform-contract.json) declares `kaggle/alaska2` for both experiment name and artifact namespace; the stable project ID is `alaska2`. A regression test pins these names. This is an explicit proposal, not evidence of provisioning. No private namespace or fallback endpoint was created. Actual shared-service import must use an approved reachable endpoint and validate the existing lifecycle/artifact namespace through the platform registry. Names do not grant access.

The shared platform owns deployment and service operation. This application owns scientific identity, verified outputs, portable offline artifacts and its consumer recovery contract.
