"""Tests for residual FastKAN warp invariants."""

import pytest
import torch
import torch.nn.functional as F

from src.models.kan import ResidualFastKANWarp


def test_identity_at_init_exact_when_alpha_fixed_zero():
    warp = ResidualFastKANWarp(
        input_dim=8,
        hidden_dim=4,
        alpha_init=0.0,
        learnable_alpha=False,
    )
    z = torch.randn(5, 8)

    out = warp(z)
    expected = F.normalize(z, dim=-1, eps=1e-12)

    assert torch.equal(out, expected)


def test_gradient_flows_through_alpha_and_kan_params():
    warp = ResidualFastKANWarp(
        input_dim=6,
        hidden_dim=5,
        alpha_init=0.1,
        learnable_alpha=True,
        clamp_alpha=True,
        clamp_max=0.2,
    )
    z = torch.randn(4, 6, requires_grad=True)
    target = torch.randn(4, 6)

    loss = (warp(z) * target).sum()
    loss.backward()

    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert warp.alpha_raw.grad is not None
    assert torch.isfinite(warp.alpha_raw.grad).all()

    kan_grad_total = 0.0
    for name, param in warp.kan.named_parameters():
        assert param.grad is not None, f"no grad for kan.{name}"
        assert torch.isfinite(param.grad).all(), f"non-finite grad for kan.{name}"
        kan_grad_total += float(param.grad.abs().sum())
    assert kan_grad_total > 0.0


def test_alpha_clamped_to_configured_range():
    warp = ResidualFastKANWarp(
        input_dim=4,
        hidden_dim=3,
        alpha_init=1.5,
        clamp_alpha=True,
        clamp_max=0.2,
    )
    assert warp.alpha.item() == pytest.approx(0.2)

    with torch.no_grad():
        warp.alpha_raw.fill_(-1.0)
    assert warp.alpha.item() == pytest.approx(0.0)


def test_output_always_l2_normalized():
    warp = ResidualFastKANWarp(
        input_dim=10,
        hidden_dim=6,
        alpha_init=0.2,
        learnable_alpha=True,
    )
    z = torch.randn(7, 10) * 25.0

    out = warp(z)

    assert out.shape == (7, 10)
    assert torch.allclose(out.norm(dim=-1), torch.ones(7), atol=1e-5)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"input_dim": 0, "hidden_dim": 4}, "input_dim"),
        ({"input_dim": 4, "hidden_dim": 0}, "hidden_dim"),
        ({"input_dim": 4, "hidden_dim": 3, "alpha_init": float("inf")}, "finite"),
        ({"input_dim": 4, "hidden_dim": 3, "learnable_alpha": "yes"}, "bool"),
        ({"input_dim": 4, "hidden_dim": 3, "clamp_alpha": 1}, "bool"),
        ({"input_dim": 4, "hidden_dim": 3, "clamp_max": 0.0}, "clamp_max"),
    ],
)
def test_residual_warp_validation_errors(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ResidualFastKANWarp(**kwargs)


def test_residual_warp_rejects_wrong_input_shape():
    warp = ResidualFastKANWarp(input_dim=4, hidden_dim=3)
    with pytest.raises(ValueError, match="expected input shape"):
        warp(torch.randn(2, 5))
