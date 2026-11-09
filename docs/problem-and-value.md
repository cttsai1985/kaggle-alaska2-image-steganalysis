# Problem and value

The task is to assign a probability that a JPEG contains embedded data. An image's visible subject is not the target: unchanged Cover images and their stego variants share the same source content. The [competition](https://www.kaggle.com/c/alaska2-image-steganalysis) supplies examples from JMiPOD, JUNIWARD and UERD.

A possible use is to prioritize files for a trained analyst's review. A probability could order a queue; confirming embedded content would require separate analysis. This repository evaluates the competition workload and has not measured an operational review benefit.

## What would establish usefulness?

A deployment assessment would need a representative stream with independently established labels, the prevalence of embedded files, an acceptable review budget, and the cost of false alarms and missed cases. It would measure recall at that review budget, false-alarm burden, calibration and changes across acquisition conditions. These are proposed assessment criteria, not measurements from this project.

The current evaluation question is narrower: does a fixed model recipe produce a score traceable to an independent source-image holdout, and is its performance broad across JPEG-quality and stego-method cohorts? The [predeclared baseline specification](../evidence/phase2/baseline_spec.json) records that question and the controls before training.

## Why grouping and cohort evidence come first

**MEASURED:** the downloaded snapshot contains 75,000 complete source families and 5,000 test images. The family holdout contains 60,000 files from 15,000 families, with 20,000 files in each quality cohort. [Dataset audit](../evidence/phase1/dataset_audit.json) · [Split audit](../evidence/phase1/split_audit.json).

A file-level split would allow source content into both roles. The diagnostic quantifies that exposure; it does not measure the resulting score bias. The maintained design retains the original family boundary and asserts byte separation as an additional check.

**MEASURED:** 250 stego files are byte-identical to their own Cover. Different labels therefore do not always imply different pixels. The audit keeps these cases visible instead of silently removing difficult or contradictory examples. Their effect on the model's held-out errors remains unmeasured because the baseline has not produced verified predictions.

JPEG-quality metadata supports a cohort analysis, but its completeness alone does not make it a useful predictive feature. The pending analysis will retain cohort counts alongside scores. Model selection, operational threshold selection and a production benefit remain separate questions.
