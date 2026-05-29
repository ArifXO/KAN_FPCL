"""Multilabel AUC/mAP utilities (R7, R9, R10)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def multilabel_auc(
    scores: np.ndarray,
    labels: np.ndarray,
) -> dict[str, object]:
    """Compute per-class and aggregate AUC + mAP for multilabel data.

    Args:
        scores: Float array [N, C] — predicted probabilities per class.
        labels: Binary int array [N, C].

    Returns:
        dict with keys: macro_auroc, micro_auroc, per_class_auroc, mAP.

    Raises:
        ValueError: If any class has no positive examples (silent NaN avoided).
    """
    if scores.shape != labels.shape:
        raise ValueError(
            f"scores and labels must have the same shape, "
            f"got {scores.shape} vs {labels.shape}"
        )

    n_classes = labels.shape[1]
    zero_pos_classes = [c for c in range(n_classes) if labels[:, c].sum() == 0]
    if zero_pos_classes:
        raise ValueError(
            f"Classes {zero_pos_classes} have 0 positive examples. "
            "AUROC is undefined. Filter or merge these classes first."
        )
    zero_neg_classes = [c for c in range(n_classes) if labels[:, c].sum() == labels.shape[0]]
    if zero_neg_classes:
        raise ValueError(
            f"Classes {zero_neg_classes} have 0 negative examples (all positive) in val set. "
            "AUROC is undefined. Filter or merge these classes first."
        )

    # With a single class, roc_auc_score(average=None) returns a scalar rather
    # than an array; wrap so per-class handling stays uniform.
    per_class_auroc = np.atleast_1d(roc_auc_score(labels, scores, average=None))
    macro_auroc = float(np.mean(per_class_auroc))
    micro_auroc = float(roc_auc_score(labels, scores, average="micro"))
    map_score = float(average_precision_score(labels, scores, average="macro"))

    return {
        "macro_auroc": macro_auroc,
        "micro_auroc": micro_auroc,
        "per_class_auroc": per_class_auroc.tolist(),
        "mAP": map_score,
    }
