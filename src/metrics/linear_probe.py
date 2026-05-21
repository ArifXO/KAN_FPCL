"""Linear probe evaluation via sklearn LogisticRegression (Stage 3)."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

from src.metrics.auc import multilabel_auc


def linear_probe(
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    val_emb: np.ndarray,
    val_labels: np.ndarray,
    max_iter: int = 1000,
    C: float = 1.0,
) -> dict[str, object]:
    """Fit a one-vs-rest logistic regression probe and evaluate on val split.

    Args:
        train_emb: [N_train, D] float32 embeddings.
        train_labels: [N_train, C] binary labels.
        val_emb: [N_val, D] float32 embeddings.
        val_labels: [N_val, C] binary labels.
        max_iter: Max iterations for LogisticRegression solver.
        C: Inverse regularisation strength.

    Returns:
        dict from multilabel_auc plus 'n_train', 'n_val', 'probe_C'.
    """
    if train_emb.shape[1] != val_emb.shape[1]:
        raise ValueError(
            f"Embedding dims must match: train {train_emb.shape[1]} "
            f"vs val {val_emb.shape[1]}"
        )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_emb)
    x_val = scaler.transform(val_emb)

    clf = OneVsRestClassifier(
        LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs"),
        n_jobs=-1,
    )
    clf.fit(x_train, train_labels)
    scores = clf.predict_proba(x_val)

    metrics = multilabel_auc(scores, val_labels)
    metrics["n_train"] = int(train_emb.shape[0])
    metrics["n_val"] = int(val_emb.shape[0])
    metrics["probe_C"] = float(C)
    return metrics
