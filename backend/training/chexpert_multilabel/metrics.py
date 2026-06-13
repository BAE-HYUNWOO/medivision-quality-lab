"""Validation metrics for multi-label classification."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    """Compute per-label AUROC/AP where possible.

    AUROC is undefined when a label has only one class in the validation subset.
    In that case, the metric is returned as NaN for that label.
    """
    per_label: dict[str, dict[str, float | int]] = {}
    aucs: list[float] = []
    aps: list[float] = []

    for idx, label in enumerate(labels):
        truth = y_true[:, idx]
        prob = y_prob[:, idx]
        positives = int(truth.sum())
        negatives = int(len(truth) - positives)

        if len(np.unique(truth)) < 2:
            auc = math.nan
            ap = math.nan
        else:
            auc = float(roc_auc_score(truth, prob))
            ap = float(average_precision_score(truth, prob))
            aucs.append(auc)
            aps.append(ap)

        per_label[label] = {
            "auroc": auc,
            "average_precision": ap,
            "positives": positives,
            "negatives": negatives,
        }

    return {
        "mean_auroc": float(np.mean(aucs)) if aucs else math.nan,
        "mean_average_precision": float(np.mean(aps)) if aps else math.nan,
        "per_label": per_label,
    }
