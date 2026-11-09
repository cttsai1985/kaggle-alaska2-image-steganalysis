"""Fixed contracts shared by every module."""

# Legacy class order (alaska_utils/data_utils.py::configure_arguments): label = index.
CLASSES: tuple[str, ...] = ("Cover", "JMiPOD", "JUNIWARD", "UERD")
COVER_INDEX = 0
TEST_KIND = "Test"

# Competition weighted-AUC parameters (competition evaluation page; legacy README).
TPR_THRESHOLDS: tuple[float, ...] = (0.0, 0.4, 1.0)
TPR_WEIGHTS: tuple[float, ...] = (2.0, 1.0)

# Result labels: a number carrying one of these is never a model result on real data.
PURPOSE_LABELS = {
    "fixture": "FIXTURE_NOT_A_MODEL_RESULT",
    "smoke": "SMOKE_READINESS_ONLY_NOT_A_MODEL_RESULT",
    "drill": "FAULT_DRILL_NOT_A_MODEL_RESULT",
    "baseline": "REAL_DATA_BASELINE",
}
