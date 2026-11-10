# Concentrated confidence errors and weak ranking cohorts

## Diagnostic and population

The [analyzer](../tools/analysis/error_analysis.py) selects the largest binary
log-loss errors separately within Cover and stego, using a fixed fraction of
0.05 and canonical retained-row order to break ties. The selection diagnoses
confidence failures; it is not a decomposition of weighted AUC and is not a new
unbiased benchmark. Every row stays in the original evaluation.

**MEASURED:** the original holdout contains 60,000 files from 15,000 families.
The hard cohort contains 750 Cover and 2,250 stego files. The independently
reconciled full-population weighted AUC remains 0.8354033190.
The [aggregate receipt](error-analysis.json) retains exact producer identities,
analyzer hash, denominators and the unmodified evaluation scope. No private
images, case indexes or row predictions are published.

## Input, representation and failure

Native JPEGs retain their dimensions and use the checkpoint's normalization.
The four-class network produces Cover/JMiPOD/JUNIWARD/UERD probabilities;
`1 - P(Cover)` becomes the binary hidden-message probability.

**MEASURED hard stego counts:** JUNIWARD contributes 2,179 files, JMiPOD 44 and
UERD 27, from equal method populations of 15,000 files each. JUNIWARD's
Cover-versus-method weighted AUC is 0.61846, compared with 0.90974 and 0.93095
for JMiPOD and UERD. Both ranking and confidence diagnostics therefore identify
JUNIWARD as a weakness, without proving which image features cause it.

**MEASURED:** quality cohorts 75/90/95 each contain 20,000 files and contribute
1,308/1,400/292 hard files. Quality 95 has the weakest ranking AUC, 0.76598,
yet the fewest extreme confidence failures. Confidence errors and ranking
separation answer different questions. Different source images populate the
quality cohorts; compression is not isolated as a causal treatment.

## Decision and next experiment

Keep method and quality slices beside the pooled metric. Do not delete hard
rows or retune thresholds and blend weights against these labels. Aggregate
probabilities cannot establish a particular texture, camera or image-content
mechanism, and this analysis includes no per-image visual finding.

A next controlled experiment would compare independently trained, matched
family-separated models for JUNIWARD, retaining all validation rows and the
same JPEG preprocessing. Report method and quality results together to test
whether a gain extends beyond a single cohort. That experiment is proposed,
not executed; no new training or Kaggle submission is part of this writeup.
