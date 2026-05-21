"""kNN evaluation with cosine similarity and weighted voting (Stage 3)."""

from __future__ import annotations

import numpy as np

from src.metrics.auc import multilabel_auc


def knn_eval(
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    val_emb: np.ndarray,
    val_labels: np.ndarray,
    k: int = 20,
) -> dict[str, object]:
    """Evaluate embeddings with cosine kNN and multilabel weighted voting.

    Each query's score for class c is the sum of cosine similarities over
    its k nearest neighbours that are positive for c, divided by k. This
    gives a soft score in [0, 1] suitable for AUROC computation.

    Args:
        train_emb: [N_train, D] — L2-normalised embeddings recommended.
        train_labels: [N_train, C] binary int labels.
        val_emb: [N_val, D].
        val_labels: [N_val, C] binary int labels.
        k: Number of nearest neighbours.

    Returns:
        dict from multilabel_auc plus 'k', 'n_train', 'n_val'.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if k > train_emb.shape[0]:
        raise ValueError(
            f"k={k} exceeds training set size {train_emb.shape[0]}"
        )

    # L2-normalise for cosine similarity via dot product.
    def _l2(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)
        return x / norms

    tr = _l2(train_emb.astype(np.float32))
    vl = _l2(val_emb.astype(np.float32))

    # [N_val, N_train] cosine similarities.
    sims = vl @ tr.T

    # Top-k indices per query.
    topk_idx = np.argpartition(sims, -k, axis=1)[:, -k:]  # [N_val, k]

    n_val, n_classes = val_labels.shape[0], train_labels.shape[1]
    scores = np.zeros((n_val, n_classes), dtype=np.float32)

    for i in range(n_val):
        nn_idx = topk_idx[i]
        nn_sims = sims[i, nn_idx].clip(min=0.0)  # [k]
        nn_labels = train_labels[nn_idx].astype(np.float32)  # [k, C]
        weight_sum = nn_sims.sum() + 1e-12
        scores[i] = (nn_sims[:, None] * nn_labels).sum(axis=0) / weight_sum

    metrics = multilabel_auc(scores, val_labels)
    metrics["k"] = k
    metrics["n_train"] = int(train_emb.shape[0])
    metrics["n_val"] = int(val_emb.shape[0])
    return metrics
