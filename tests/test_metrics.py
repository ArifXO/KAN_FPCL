"""Tests for Stage 3 metrics: linear probe, kNN, AUC (R3, R7, R9)."""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics import multilabel_auc, linear_probe, knn_eval


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_clustered(n: int = 200, n_classes: int = 4, dim: int = 32, seed: int = 0):
    """Embeddings where each class cluster is well-separated.

    Each class has an orthogonal center direction. Samples are evenly split
    across both halves so [:n//2] and [n//2:] each contain positives for all
    classes — required to avoid zero-positive AUROC errors.
    """
    assert dim >= n_classes, "dim must be >= n_classes for orthogonal centers"
    rng = np.random.default_rng(seed)
    half = n // 2
    per_class_per_half = half // n_classes

    # Orthogonal class centers in R^dim (unit vectors scaled by 10).
    centers = np.eye(n_classes, dim, dtype=np.float32) * 10.0

    emb = np.zeros((n, dim), dtype=np.float32)
    labels = np.zeros((n, n_classes), dtype=np.int32)

    for half_start in (0, half):
        for c in range(n_classes):
            s = half_start + c * per_class_per_half
            e = s + per_class_per_half
            emb[s:e] = rng.standard_normal((per_class_per_half, dim)).astype(np.float32) * 0.1 + centers[c]
            labels[s:e, c] = 1

    return emb, labels


def _make_random(n: int = 200, n_classes: int = 4, dim: int = 32, seed: int = 1):
    """Random embeddings + random balanced labels."""
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal((n, dim)).astype(np.float32)
    labels = rng.integers(0, 2, size=(n, n_classes)).astype(np.int32)
    # Ensure at least one positive per class in every split
    for c in range(n_classes):
        labels[c * 2, c] = 1
        labels[c * 2 + 1, c] = 1
        labels[n // 2 + c * 2, c] = 1
        labels[n // 2 + c * 2 + 1, c] = 1
    return emb, labels


# ---------------------------------------------------------------------------
# multilabel_auc
# ---------------------------------------------------------------------------


def test_auc_clustered_gt_09():
    emb, labels = _make_clustered(n=200, n_classes=4)
    half = 100
    tr_emb, tr_lbl = emb[:half], labels[:half]
    vl_emb, vl_lbl = emb[half:], labels[half:]

    # Use probe scores as a proxy for "embeddings tell you labels"
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    clf = OneVsRestClassifier(LogisticRegression(max_iter=500))
    clf.fit(tr_emb, tr_lbl)
    scores = clf.predict_proba(vl_emb)

    out = multilabel_auc(scores, vl_lbl)
    assert out["macro_auroc"] > 0.9, f"Expected >0.9, got {out['macro_auroc']:.4f}"


def test_auc_random_near_05():
    rng = np.random.default_rng(42)
    n, n_classes = 400, 4
    emb, labels = _make_random(n=n, n_classes=n_classes)
    half = n // 2
    # Random scores — should be near 0.5
    scores = rng.random((half, n_classes)).astype(np.float32)
    vl_lbl = labels[half:]
    out = multilabel_auc(scores, vl_lbl)
    assert 0.3 < out["macro_auroc"] < 0.7, (
        f"Random scores should give AUROC near 0.5, got {out['macro_auroc']:.4f}"
    )


def test_auc_zero_positives_raises():
    scores = np.random.rand(50, 3).astype(np.float32)
    labels = np.ones((50, 3), dtype=np.int32)
    labels[:, 1] = 0  # class 1 has no positives
    with pytest.raises(ValueError, match="0 positive examples"):
        multilabel_auc(scores, labels)


def test_auc_all_positive_raises():
    scores = np.random.rand(50, 3).astype(np.float32)
    labels = np.ones((50, 3), dtype=np.int32)  # all positive
    with pytest.raises(ValueError, match="0 negative"):
        multilabel_auc(scores, labels)


def test_auc_shape_mismatch_raises():
    with pytest.raises(ValueError, match="same shape"):
        multilabel_auc(np.zeros((10, 3)), np.zeros((10, 4)))


def test_auc_returns_required_keys():
    rng = np.random.default_rng(0)
    scores = rng.random((20, 2)).astype(np.float32)
    labels = np.array([[1, 0]] * 10 + [[0, 1]] * 10, dtype=np.int32)
    out = multilabel_auc(scores, labels)
    for key in ("macro_auroc", "micro_auroc", "per_class_auroc", "mAP"):
        assert key in out, f"missing key {key}"


# ---------------------------------------------------------------------------
# linear_probe
# ---------------------------------------------------------------------------


def test_linear_probe_clustered_gt_09():
    emb, labels = _make_clustered(n=400, n_classes=4)
    half = 200
    out = linear_probe(emb[:half], labels[:half], emb[half:], labels[half:])
    assert out["macro_auroc"] > 0.9, f"Expected >0.9, got {out['macro_auroc']:.4f}"


def test_linear_probe_random_near_05():
    emb, labels = _make_random(n=400, n_classes=4)
    half = 200
    out = linear_probe(emb[:half], labels[:half], emb[half:], labels[half:])
    assert 0.3 < out["macro_auroc"] < 0.75, (
        f"Random probe should be near 0.5, got {out['macro_auroc']:.4f}"
    )


def test_linear_probe_returns_required_keys():
    emb, labels = _make_clustered(n=100, n_classes=2)
    out = linear_probe(emb[:50], labels[:50], emb[50:], labels[50:])
    for key in ("macro_auroc", "micro_auroc", "per_class_auroc", "mAP", "n_train", "n_val", "probe_C"):
        assert key in out, f"missing key {key}"


def test_linear_probe_dim_mismatch_raises():
    with pytest.raises(ValueError, match="Embedding dims must match"):
        linear_probe(
            np.zeros((10, 32)), np.zeros((10, 2), dtype=np.int32),
            np.zeros((10, 64)), np.zeros((10, 2), dtype=np.int32),
        )


# ---------------------------------------------------------------------------
# kNN
# ---------------------------------------------------------------------------


def test_knn_identical_k1_auroc_perfect():
    """k=1 with val==train should recover labels perfectly -> AUROC=1.0."""
    rng = np.random.default_rng(7)
    n, n_classes, dim = 60, 3, 16
    emb = rng.standard_normal((n, dim)).astype(np.float32)
    labels = np.zeros((n, n_classes), dtype=np.int32)
    for c in range(n_classes):
        labels[c * 20:(c + 1) * 20, c] = 1

    # Val is same as train — k=1 nearest is itself => perfect score.
    out = knn_eval(emb, labels, emb, labels, k=1)
    assert out["macro_auroc"] > 0.99, f"Expected ~1.0, got {out['macro_auroc']:.4f}"


def test_knn_clustered_gt_09():
    emb, labels = _make_clustered(n=400, n_classes=4)
    half = 200
    out = knn_eval(emb[:half], labels[:half], emb[half:], labels[half:], k=10)
    assert out["macro_auroc"] > 0.9, f"Expected >0.9, got {out['macro_auroc']:.4f}"


def test_knn_returns_required_keys():
    emb, labels = _make_clustered(n=100, n_classes=2)
    out = knn_eval(emb[:50], labels[:50], emb[50:], labels[50:], k=5)
    for key in ("macro_auroc", "micro_auroc", "per_class_auroc", "mAP", "k", "n_train", "n_val"):
        assert key in out, f"missing key {key}"


def test_knn_invalid_k_raises():
    emb = np.zeros((10, 8), dtype=np.float32)
    lbl = np.ones((10, 2), dtype=np.int32)
    with pytest.raises(ValueError, match="k must be positive"):
        knn_eval(emb, lbl, emb, lbl, k=0)
    with pytest.raises(ValueError, match="exceeds training set size"):
        knn_eval(emb, lbl, emb, lbl, k=100)
