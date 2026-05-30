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
    # detach_inputs=False here so we can verify the underlying op is
    # differentiable wrt z. The default (True) deliberately severs this
    # path in train_fn.py to keep scorer gradients off the encoder.
    scorer = MLPPairScorer(input_dim=8, hidden_dim=16, num_layers=2, detach_inputs=False)
    z = torch.randn(4, 8, requires_grad=True)
    p = scorer(z)
    p.sum().backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert z.grad.abs().sum() > 0


def test_detach_inputs_blocks_gradient():
    """detach_inputs=True is the deployment default — z must NOT receive grad."""
    scorer = MLPPairScorer(input_dim=8, hidden_dim=16, num_layers=2, detach_inputs=True)
    z = torch.randn(4, 8, requires_grad=True)
    scorer(z).sum().backward()
    assert z.grad is None or float(z.grad.abs().sum()) == 0.0


def test_init_pfn_prior_initial_mean():
    """R-D acceptance: initial mean p_fn ≈ init_pfn_prior on random z."""
    torch.manual_seed(0)
    for prior in (0.05, 0.1, 0.2):
        scorer = MLPPairScorer(input_dim=32, hidden_dim=16, init_pfn_prior=prior)
        z = torch.randn(64, 32)
        p = scorer(z)
        assert abs(p.mean().item() - prior) < 0.05, (
            f"prior={prior} expected mean ≈ {prior}, got {p.mean().item():.4f}"
        )


def test_init_pfn_prior_invalid_raises():
    with pytest.raises(ValueError, match="init_pfn_prior must be in"):
        MLPPairScorer(input_dim=8, init_pfn_prior=0.0)
    with pytest.raises(ValueError, match="init_pfn_prior must be in"):
        MLPPairScorer(input_dim=8, init_pfn_prior=1.0)


def test_gradient_flows_to_parameters():
    # Scorer params must always get grad; works regardless of detach_inputs.
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


# ---------------------------------------------------------------------------
# Symmetry
# ---------------------------------------------------------------------------


def test_mlp_scorer_symmetry():
    scorer = MLPPairScorer(input_dim=32, hidden_dim=16)
    z = torch.randn(6, 32)
    p = scorer(z)
    assert torch.allclose(p, p.T, atol=1e-6), (
        f"MLPPairScorer is not symmetric: max diff={(p - p.T).abs().max()}"
    )
