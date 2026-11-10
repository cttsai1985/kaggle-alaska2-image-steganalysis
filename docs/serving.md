# Saved-checkpoint input and prediction contract

The maintained interface is batch prediction from authorized native JPEGs,
not an HTTP service. The checkpoint carries model configuration, class order,
normalization and provenance. The decoder retains native dimensions; model
probabilities are ordered Cover, JMiPOD, JUNIWARD, UERD. The hidden-message
score is `1 - P(Cover)`.

Use the existing [reproduction guide](reproduction.md) to evaluate/reload the
checkpoint and generate predictions. `predict --checkpoint CHECKPOINT` writes
the package prediction format. The verified export path produces the competition
CSV in supplied sample-ID order and retains its producer receipt.
[Prediction and recovery contract](consumer-contract.md) documents validation,
identity checks, immutable delivery and rejection of stale evidence.

**MEASURED existing evidence:** independent checkpoint reconstruction matches
all 60,000 holdout and 5,000 test predictions exactly, as recorded in
[the retained execution receipt](../evidence/phase2/retained_execution.json).
This proves the recorded batch/reload scope, not HTTP parity.

MLflow export, fresh-workspace HTTP parity, invalid HTTP request rejection and
online operations are open requirements. No canary or production-service claim
is made for this writeup update. See [the requirements audit](requirements-audit.md).
