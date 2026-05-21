"""Tests for Stage 8 embedding geometry metrics."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.metrics import (
    alignment,
    effective_rank,
    off_diagonal_covariance_norm,
    per_dim_std,
    uniformity,
)


def test_alignment_identical_vectors_is_zero():
    z = F.normalize(torch.randn(16, 8), dim=-1)
    assert torch.equal(alignment(z, z), torch.tensor(0.0))


def test_uniform_distribution_has_lower_uniformity_loss_than_clustered():
    generator = torch.Generator().manual_seed(0)
    uniform = F.normalize(torch.randn(256, 16, generator=generator), dim=-1)
    center = F.normalize(torch.randn(1, 16, generator=generator), dim=-1)
    clustered = F.normalize(
        center + 0.02 * torch.randn(256, 16, generator=generator),
        dim=-1,
    )

    assert uniformity(uniform) < uniformity(clustered)


def test_effective_rank_rank_one_is_one():
    z = F.normalize(torch.ones(32, 12), dim=-1)
    assert effective_rank(z).item() == pytest.approx(1.0, abs=1e-5)


def test_effective_rank_orthogonal_rows_matches_dimension():
    dim = 8
    z = torch.eye(dim)
    assert effective_rank(z).item() == pytest.approx(float(dim), rel=1e-5)


def test_per_dim_std_has_expected_shape_and_finite_values():
    z = F.normalize(torch.randn(64, 10), dim=-1)
    std = per_dim_std(z)
    assert std.shape == (10,)
    assert torch.isfinite(std).all()
    assert (std > 0).all()


def test_off_diagonal_covariance_zero_for_balanced_sign_grid():
    values = torch.tensor([-1.0, 1.0])
    z = torch.cartesian_prod(values, values, values, values) / 2.0
    assert off_diagonal_covariance_norm(z).item() == pytest.approx(0.0, abs=1e-7)


def test_geometry_validation_errors_are_descriptive():
    with pytest.raises(ValueError, match="same shape"):
        alignment(torch.randn(3, 4), torch.randn(3, 5))
    with pytest.raises(ValueError, match="at least 2 rows"):
        uniformity(torch.randn(1, 4))
    with pytest.raises(ValueError, match="positive finite"):
        uniformity(torch.randn(3, 4), t=0)
    with pytest.raises(ValueError, match="all-zero"):
        effective_rank(torch.zeros(3, 4))
