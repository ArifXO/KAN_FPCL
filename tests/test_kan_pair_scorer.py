"""Tests for KANPairScorer (Stage 7 — H3).

Verifies:
  - Output shape and range [0, 1]
  - R1 parameter parity within ±15% of MLPPairScorer anchor
  - Gradient flow through all parameters
  - Drop-in interchangeability with FNWeightedInfoNCELoss
"""

import torch
import pytest

from src.models.pair_scorer import KANPairScorer, MLPPairScorer
from src.losses.fn_weighted_infonce import FNWeightedInfoNCELoss

B, D = 4, 128


def test_output_shape():
    scorer = KANPairScorer(input_dim=D)
    z = torch.randn(B, D)
    p_fn = scorer(z)
    assert p_fn.shape == (B, B), f"Expected ({B}, {B}), got {p_fn.shape}"
    assert p_fn.min() >= 0.0 and p_fn.max() <= 1.0, (
        f"p_fn out of [0, 1]: min={p_fn.min():.4f}, max={p_fn.max():.4f}"
    )


def test_param_parity():
    kan = KANPairScorer(input_dim=D)
    mlp = MLPPairScorer(input_dim=D, hidden_dim=32)
    kan_params = kan.parameter_count()
    mlp_params = mlp.parameter_count()
    ratio = abs(kan_params - mlp_params) / mlp_params
    assert ratio < 0.15, (
        f"R1 parity violated: KAN={kan_params}, MLP={mlp_params}, "
        f"diff={ratio:.1%} (must be < 15%)"
    )


def test_gradient_flows():
    scorer = KANPairScorer(input_dim=D)
    z = torch.randn(B, D, requires_grad=True)
    p_fn = scorer(z)
    loss = p_fn.sum()
    loss.backward()
    for name, param in scorer.named_parameters():
        assert param.grad is not None, f"No gradient for param: {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient in: {name}"


def test_interchangeable():
    """KANPairScorer is a drop-in for MLPPairScorer in FNWeightedInfoNCELoss."""
    scorer = KANPairScorer(input_dim=D)
    loss_fn = FNWeightedInfoNCELoss(temperature=0.1, max_fn_weight=0.5)

    z = torch.randn(2 * B, D)
    z = torch.nn.functional.normalize(z, dim=-1)

    z_view1 = z[:B]
    p_fn = scorer(z_view1)
    assert p_fn.shape == (B, B)

    out = loss_fn(z, p_fn)
    assert "loss" in out, "FNWeightedInfoNCELoss output missing 'loss' key"
    assert torch.isfinite(out["loss"]), f"Loss is not finite: {out['loss']}"


def test_kan_scorer_symmetry():
    scorer = KANPairScorer(input_dim=D)
    z = torch.randn(6, D)
    p = scorer(z)
    assert torch.allclose(p, p.T, atol=1e-6), (
        f"KANPairScorer is not symmetric: max diff={(p - p.T).abs().max()}"
    )
