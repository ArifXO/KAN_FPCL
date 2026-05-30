"""Tests for multilabel mask utilities (R3 — masks for SupCon Oracle)."""

from __future__ import annotations

import pytest
import torch

from src.losses.multilabel_masks import (
    build_self_view_positive_mask,
    build_supcon_oracle_positive_mask,
    expand_labels_for_two_views,
    multilabel_jaccard_matrix,
    multilabel_overlap_matrix,
)


# ---------------------------------------------------------------------------
# expand_labels_for_two_views
# ---------------------------------------------------------------------------


def test_expand_labels_doubles_rows_in_order():
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    out = expand_labels_for_two_views(labels)
    assert out.shape == (6, 2)
    assert torch.equal(out[:3], labels)
    assert torch.equal(out[3:], labels)


def test_expand_labels_invalid_shape_raises():
    with pytest.raises(ValueError, match=r"\[B, C\]"):
        expand_labels_for_two_views(torch.zeros(3))


# ---------------------------------------------------------------------------
# multilabel_overlap_matrix
# ---------------------------------------------------------------------------


def test_overlap_matrix_shape_and_diagonal():
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    labels_2v = expand_labels_for_two_views(labels)
    M = multilabel_overlap_matrix(labels_2v)
    assert M.shape == (4, 4)
    # Diagonal = label cardinality.
    assert torch.allclose(M.diagonal(), labels_2v.sum(dim=-1))


def test_overlap_matrix_counts_intersections():
    labels = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    labels_2v = expand_labels_for_two_views(labels)
    M = multilabel_overlap_matrix(labels_2v)
    # |{1,1,0} ∩ {1,0,1}| = 1, symmetric.
    assert M[0, 1].item() == 1.0
    assert M[1, 0].item() == 1.0


# ---------------------------------------------------------------------------
# multilabel_jaccard_matrix
# ---------------------------------------------------------------------------


def test_jaccard_self_is_one_for_nonempty():
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    labels_2v = expand_labels_for_two_views(labels)
    J = multilabel_jaccard_matrix(labels_2v)
    assert torch.allclose(J.diagonal(), torch.ones(4))


def test_jaccard_known_values():
    labels = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    labels_2v = expand_labels_for_two_views(labels)
    J = multilabel_jaccard_matrix(labels_2v)
    # |∩|=1, |∪|=3 -> 1/3
    assert pytest.approx(1 / 3, rel=1e-5) == J[0, 1].item()


def test_jaccard_all_zero_rows_no_nan():
    # A row of all-zero labels: 0/0 must be 0, not NaN.
    labels = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    labels_2v = expand_labels_for_two_views(labels)
    J = multilabel_jaccard_matrix(labels_2v)
    assert torch.isfinite(J).all()


# ---------------------------------------------------------------------------
# build_self_view_positive_mask
# ---------------------------------------------------------------------------


def test_self_view_mask_structure():
    M = build_self_view_positive_mask(3)
    assert M.shape == (6, 6)
    assert M.dtype == torch.bool
    for i in range(3):
        assert M[i, i + 3].item()
        assert M[i + 3, i].item()
    assert not M.diagonal().any().item()
    assert M.sum().item() == 6


def test_self_view_mask_invalid_b_raises():
    with pytest.raises(ValueError, match="B must be positive"):
        build_self_view_positive_mask(0)


# ---------------------------------------------------------------------------
# build_supcon_oracle_positive_mask
# ---------------------------------------------------------------------------


def test_any_overlap_mask_shape_and_diag():
    labels = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    M = build_supcon_oracle_positive_mask(labels, positive_mode="any_overlap")
    assert M.shape == (6, 6)
    assert M.dtype == torch.bool
    assert not M.diagonal().any().item()


def test_any_overlap_positives_correct():
    labels = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    M = build_supcon_oracle_positive_mask(
        labels, positive_mode="any_overlap", include_self_view_positive=True
    )
    # Image 0 ([1,0]) overlaps image 1 ([1,1]) at index 1.
    assert M[0, 1].item()
    # Image 2 ([0,0]) overlaps nothing label-wise.
    assert not M[2, 0].item()
    assert not M[2, 1].item()
    # Self-view positives always present.
    assert M[0, 3].item() and M[3, 0].item()
    assert M[2, 5].item() and M[5, 2].item()


def test_no_overlap_pairs_are_false():
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    M = build_supcon_oracle_positive_mask(
        labels, positive_mode="any_overlap", include_self_view_positive=False
    )
    # Disjoint labels — only self-view would have made them positives, and
    # self-view is OFF, so the cross-image entries must be False.
    assert not M[0, 1].item()
    assert not M[0, 3].item()  # self-view OFF


def test_jaccard_threshold_filters():
    labels = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])  # Jaccard = 1/3
    M_lax = build_supcon_oracle_positive_mask(
        labels, positive_mode="jaccard", min_jaccard=0.0,
        include_self_view_positive=False,
    )
    M_strict = build_supcon_oracle_positive_mask(
        labels, positive_mode="jaccard", min_jaccard=0.5,
        include_self_view_positive=False,
    )
    assert M_lax[0, 1].item()
    assert not M_strict[0, 1].item()


def test_all_zero_label_row_no_nan_no_overlap_positives():
    labels = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    M = build_supcon_oracle_positive_mask(
        labels, positive_mode="jaccard", min_jaccard=0.0,
        include_self_view_positive=True,
    )
    # Zero-label anchor: no cross-image positives, only its self-view.
    assert not M[0, 1].item()
    assert M[0, 2].item()  # self-view


def test_invalid_positive_mode_raises():
    with pytest.raises(ValueError, match="positive_mode"):
        build_supcon_oracle_positive_mask(
            torch.zeros(2, 2), positive_mode="bogus",
        )


def test_invalid_min_jaccard_raises():
    with pytest.raises(ValueError, match="min_jaccard"):
        build_supcon_oracle_positive_mask(
            torch.zeros(2, 2), positive_mode="jaccard", min_jaccard=1.0,
        )
