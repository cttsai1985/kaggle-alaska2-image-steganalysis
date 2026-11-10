# Predict the union of embedding methods

## Question and controlled comparison

A JPEG may contain JMiPOD, JUNIWARD or UERD embedding. Does taking the largest
stego-class probability preserve the ranking produced by `1 - P(Cover)`?
The maximum is a plausible shortcut when a single embedding class dominates,
but it discards probability assigned to the other embedding classes.

The two readouts use the same saved checkpoint, native JPEG preprocessing,
labels and complete family holdout. No fitting, row removal, weight search or
threshold tuning occurs. The [aggregate receipt](modeling-study.json) binds the
checkpoint, predictions, split and analyzer hashes; the
[rerunnable analyzer](../tools/analysis/modeling_study.py) verifies those inputs.

## Measured effect and variation

**MEASURED:** the population contains 60,000 files from 15,000 source families.
Competition weighted AUC is higher-is-better.

| Readout | Holdout weighted AUC | Difference from union |
| --- | ---: | ---: |
| Union, `1 - P(Cover)` | 0.8354033190 | 0 |
| Largest stego-class probability | 0.7674848725 | -0.0679184466 |

**MEASURED treatment activity:** every score differs and the ranking changes.
The alternative loses in every recorded quality and Cover-versus-method slice;
quality-slice deltas range from -0.05411 to -0.04891. The paired family-bootstrap
2.5/97.5 percentiles for the pooled delta are -0.06889 and -0.06674
(MEASURED using 64 shared resamples, fixed seed 20261109). This is uncertainty
under resampling one fixed holdout, not training-seed stability or full CV.

## Decision and remaining question

Retain the existing union-probability readout. The default does not change.
This diagnostic can lose in either direction; the two formulas can reorder
predictions, and no constant forces the union to win the measured comparison.
The result supports this checkpoint's inference policy, not a claim that it is
optimal for every model or embedding distribution.

The remaining question is whether independently trained models reduce the
JUNIWARD weakness under family-separated validation. That needs a separately
declared training/OOF experiment; the reported holdout is not a tuning set.
The retained training run remains an incomplete-run fallback.
