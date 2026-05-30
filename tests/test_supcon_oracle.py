"""Tests for MultilabelSupConOracleLoss (R3, R7, R9 — oracle baseline)."""

from __future__ import annotations

import pytest
import torch

from src.losses import MultilabelSupConOracleLoss


_REQUIRED_KEYS = {
    "loss",
    "num_oracle_positives_mean",
    "num_oracle_positives_min",
    "num_oracle_positives_max",
    "positive_pair_fraction",
    "dropped_anchor_fraction",
    "pos_sim_mean",
    "neg_sim_mean",
    "temperature",
    "positive_mode",
    "min_jaccard",
}


def _z_and_labels(
    batch: int = 4, dim: int = 8, n_classes: int = 3, seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    v1 = torch.randn(batch, dim, generator=g)
    v2 = v1 + 0.1 * torch.randn(batch, dim, generator=g)
    z = torch.cat([v1, v2], dim=0).requires_grad_(True)
    labels = (torch.rand(batch, n_classes, generator=g) > 0.5).float()
    return z, labels


# ---------------------------------------------------------------------------
# Constructor validation (R9)
# ---------------------------------------------------------------------------


def test_temperature_validation():
    with pytest.raises(ValueError, match="temperature must be > 0"):
        MultilabelSupConOracleLoss(temperature=0.0)


def test_positive_mode_validation():
    with pytest.raises(ValueError, match="positive_mode"):
        MultilabelSupConOracleLoss(positive_mode="bogus")


def test_min_jaccard_validation():
    with pytest.raises(ValueError, match="min_jaccard"):
        MultilabelSupConOracleLoss(min_jaccard=-0.1)
    with pytest.raises(ValueError, match="min_jaccard"):
        MultilabelSupConOracleLoss(min_jaccard=1.0)


def test_no_positive_policy_validation():
    with pytest.raises(ValueError, match="no_positive_policy"):
        MultilabelSupConOracleLoss(no_positive_policy="bogus")


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_labels_required():
    loss_fn = MultilabelSupConOracleLoss()
    z, labels = _z_and_labels()
    # Forward requires labels — missing positional arg.
    with pytest.raises(TypeError):
        loss_fn(z)


def test_label_shape_mismatch_raises():
    loss_fn = MultilabelSupConOracleLoss()
    z, labels = _z_and_labels(batch=4)
    with pytest.raises(ValueError, match=r"\[B, C\] with B=4"):
        loss_fn(z, torch.zeros(3, 5))


def test_invalid_z_shape_raises():
    loss_fn = MultilabelSupConOracleLoss()
    with pytest.raises(ValueError, match=r"\[2B, D\]"):
        loss_fn(torch.randn(8), torch.zeros(4, 3))


# ---------------------------------------------------------------------------
# Dict / shape contract (R7)
# ---------------------------------------------------------------------------


def test_returns_required_dict_keys():
    loss_fn = MultilabelSupConOracleLoss(positive_mode="any_overlap")
    z, labels = _z_and_labels()
    out = loss_fn(z, labels)
    missing = _REQUIRED_KEYS - set(out.keys())
    assert not missing, f"missing keys: {missing}"


def test_loss_is_finite_scalar():
    loss_fn = MultilabelSupConOracleLoss()
    z, labels = _z_and_labels()
    out = loss_fn(z, labels)
    loss = out["loss"]
    assert loss.dim() == 0
    assert torch.isfinite(loss).item()


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_gradient_flows_to_input():
    z, labels = _z_and_labels()
    out = MultilabelSupConOracleLoss(temperature=0.1)(z, labels)
    out["loss"].backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert z.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Positive-mask behavior
# ---------------------------------------------------------------------------


def test_self_view_positives_always_included():
    """Even with disjoint labels, the augmented view of the same image
    must be treated as a positive when include_self_view_positive=True."""
    z = torch.randn(8, 16, requires_grad=True)
    labels = torch.eye(4, 3)[:, :3]  # disjoint single-label rows
    out = MultilabelSupConOracleLoss(
        positive_mode="any_overlap", include_self_view_positive=True,
    )(z, labels)
    # n_positives_min should be at least 1 (the self-view partner).
    assert out["num_oracle_positives_min"].item() >= 1.0


def test_any_overlap_positives_correct():
    """With identical labels everywhere, all cross-image pairs are positives."""
    z = torch.randn(8, 16)
    labels = torch.ones(4, 3)  # all positive on all classes
    out = MultilabelSupConOracleLoss(positive_mode="any_overlap")(z, labels)
    # Each anchor: 7 cross-image positives + 1 self-view = 7 in 2B-1=7 slots.
    assert out["num_oracle_positives_mean"].item() == 7.0


def test_jaccard_threshold_filters_positives():
    z = torch.randn(8, 16)
    # Two distinct labels per row: rows 0,1 share class 0; rows 2,3 share class 1.
    labels = torch.tensor(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    out_lax = MultilabelSupConOracleLoss(
        positive_mode="jaccard", min_jaccard=0.0,
        include_self_view_positive=False,
    )(z, labels)
    out_strict = MultilabelSupConOracleLoss(
        positive_mode="jaccard", min_jaccard=0.99,
        include_self_view_positive=False,
    )(z, labels)
    # Lax threshold picks up the partner-label pairs.
    assert out_lax["num_oracle_positives_mean"].item() > 0
    # Strict threshold keeps only identical-label pairs; here all pairs in
    # each cluster are identical (Jaccard=1) so we still have positives.
    assert out_strict["num_oracle_positives_mean"].item() > 0


def test_all_zero_labels_handled_safely():
    """All-zero label rows must not produce NaN — they fall through to the
    self-view positive under the default policy."""
    z = torch.randn(8, 16)
    labels = torch.zeros(4, 3)
    out = MultilabelSupConOracleLoss(
        positive_mode="any_overlap",
        no_positive_policy="self_view_only",
    )(z, labels)
    assert torch.isfinite(out["loss"]).item()
    assert out["dropped_anchor_fraction"].item() == 0.0


# ---------------------------------------------------------------------------
# no_positive_policy
# ---------------------------------------------------------------------------


def test_no_positive_policy_self_view_only_keeps_anchors():
    z = torch.randn(8, 16)
    labels = torch.zeros(4, 3)  # no label-overlap positives anywhere
    out = MultilabelSupConOracleLoss(
        positive_mode="any_overlap",
        include_self_view_positive=True,
        no_positive_policy="self_view_only",
    )(z, labels)
    assert out["dropped_anchor_fraction"].item() == 0.0


def test_no_positive_policy_drop_anchor_drops_anchors():
    z = torch.randn(8, 16)
    labels = torch.zeros(4, 3)
    out = MultilabelSupConOracleLoss(
        positive_mode="any_overlap",
        include_self_view_positive=False,
        no_positive_policy="drop_anchor",
    )
    # All anchors get dropped -> RuntimeError per R9 (no silent NaN).
    with pytest.raises(RuntimeError, match="every anchor was dropped"):
        out(z, labels)


def test_drop_anchor_partial_drop():
    """Mix of label-having and label-less anchors with drop_anchor policy:
    only label-less anchors are dropped, label-having anchors contribute."""
    z = torch.randn(8, 16)
    labels = torch.tensor(
        [[1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]
    )
    out = MultilabelSupConOracleLoss(
        positive_mode="any_overlap",
        include_self_view_positive=False,
        no_positive_policy="drop_anchor",
    )(z, labels)
    assert torch.isfinite(out["loss"]).item()
    assert out["dropped_anchor_fraction"].item() > 0
