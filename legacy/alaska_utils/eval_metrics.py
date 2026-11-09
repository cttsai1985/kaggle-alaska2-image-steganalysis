import warnings
from typing import List

import numpy as np
from sklearn import metrics

warnings.filterwarnings("ignore")


# competition metrics
def alaska_weighted_auc(
        y_true: np.array, y_valid: np.array, tpr_thresholds: List[float] = [0.0, 0.4, 1.0],
        weights: List[float] = [2, 1]):
    """
    https://www.kaggle.com/anokas/weighted-auc-metric-updated
    """
    # size of subsets
    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])

    # The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.
    normalization = np.dot(areas, weights)

    def compute_sub_metrics(tpr_min: float, tpr_max: float, fpr_arr: np.array, tpr_arr: np.array) -> float:
        mask = (tpr_min <= tpr_arr) & (tpr_arr <= tpr_max)

        if not mask.any():  # at least one sample
            return 0.

        fpr_sel = fpr_arr[mask]
        fpr_sel = np.concatenate([fpr_sel, [fpr_sel[-1], 1.]])
        tpr_sel = np.concatenate([tpr_arr[mask], [tpr_max, tpr_max]])
        return metrics.auc(fpr_sel, tpr_sel - tpr_min)  # normalize such that curve starts at y=0

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_valid, pos_label=1)
    sub_metrics = [compute_sub_metrics(
        tpr_min=a, tpr_max=b, fpr_arr=fpr, tpr_arr=tpr) for a, b in zip(tpr_thresholds[:-1], tpr_thresholds[1:])]
    return np.dot(sub_metrics, weights) / normalization
