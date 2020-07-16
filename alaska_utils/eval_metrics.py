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

    def compute_sub_metrics(y_min: float, y_max: float, fpr_arr: np.array, tpr_arr: np.array) -> float:
        mask = (y_min < tpr_arr) & (tpr_arr < y_max)

        if not len(fpr[mask]):
            return 0.

        x_padding = np.linspace(fpr_arr[mask][-1], 1, 100)

        x = np.concatenate([fpr_arr[mask], x_padding])
        y = np.concatenate([tpr_arr[mask], [y_max] * len(x_padding)])
        return metrics.auc(x, y - y_min)  # normalize such that curve starts at y=0

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_valid, pos_label=1)
    sub_metrics = [compute_sub_metrics(
        y_min=a, y_max=b, fpr_arr=fpr, tpr_arr=tpr) for a, b in zip(tpr_thresholds[:-1], tpr_thresholds[1:])]
    competition_metric = (np.array(sub_metrics) * weights).sum() / normalization
    return competition_metric
