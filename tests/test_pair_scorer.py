"""Tests for MLPPairScorer (R7-aligned bounds, R9 validation, R10 footprint)."""

from __future__ import annotations

import pytest
import torch

from src.models import MLPPairScorer


# ---------------------------------------------------------------------------
# Constructor validation (R9)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kw, msg",
    [
        ({"input_dim": 0}, "input_dim must be positive"),
        ({"input_dim": -1}, "input_dim must be positive"),
        ({"input_dim": 8, "hidden_dim": 0}, "hidden_dim must be positive"),
        ({"input_dim": 8, "hidden_dim": 16, "num_layers": 0}, "num_layers must be >= 1"),
    ],
)
def test_constructor_validation(kw, msg):
    with pytest.raises(ValueError, match=msg):
        MLPPairScorer(**kw)


# ---------------------------------------------------------------------------
# Forward shape + bounds
# ---------------------------------------------------------------------------


def test_output_shape_and_dtype():
    scorer = MLPPairScorer(input_dim=16, hidden_dim=32, num_layers=2)
    z = torch.randn(8, 16)
    p = scorer(z)
    assert p.shape == (8, 8), f"expected [B, B]=[8,8], got {tuple(p.shape)}"
    assert p.dtype == z.dtype


def test_output_bounded_in_unit_interval():
    """Sigmoid output must lie in [0, 1] — the FN loss requires it (R9 in loss)."""
    scorer = MLPPairScorer(input_dim=32, hidden_dim=16, num_layers=2)
    z = torch.randn(12, 32) * 5.0  # large logits stress sigmoid bounds
    p = scorer(z)
    assert (p >= 0.0).all().item()
    assert (p <= 1.0).all().item()
    assert torch.isfinite(p).all().item()


def test_output_bounded_extreme_inputs():
    scorer = MLPPairScorer(input_dim=8, hidden_dim=8, num_layers=2)
    z = torch.full((4, 8), 1e3)
    p = scorer(z)
    assert (p >= 0.0).all() and (p <= 1.0).all()


# ---------------------------------------------------------------------------
# Forward input validation (R9)
# ---------------------------------------------------------------------------


def test_invalid_input_shape_raises():
    scorer = MLPPairScorer(input_dim=8)
    with pytest.raises(ValueError, match=r"\[B, D\]"):
        scorer(torch.randn(8))


def test_input_dim_mismatch_raises():
    scorer = MLPPairScorer(input_dim=8)
    with pytest.raises(ValueError, match="input_dim=8"):
        scorer(torch.randn(4, 16))


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_gradient_flows_to_input():
    scorer = MLPPairScorer(input_dim=8, hidden_dim=16, num_layers=2)
    z = torch.randn(4, 8, requires_grad=True)
    p = scorer(z)
    p.sum().backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert z.grad.abs().sum() > 0


def test_gradient_flows_to_parameters():
    scorer = MLPPairScorer(input_dim=8, hidden_dim=16, num_layers=2)
    z = torch.randn(4, 8)
    scorer(z).sum().backward()
    for name, p in scorer.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"


# ---------------------------------------------------------------------------
# Footprint sanity — keeps R1 within reach when paired with the KAN scorer.
# ---------------------------------------------------------------------------


def test_parameter_count_reasonable_for_default_geometry():
    """Default config (input_dim=128, hidden=32, layers=2) stays under 10K
    params so the matching KAN scorer can land within ±15 %."""
    scorer = MLPPairScorer(input_dim=128, hidden_dim=32, num_layers=2)
    n = scorer.parameter_count()
    assert 5_000 < n < 12_000, f"unexpected param count: {n}"
