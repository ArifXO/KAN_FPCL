"""Tests for residual FastKAN warp invariants."""

import pytest
import torch
import torch.nn.functional as F

from src.models import MLPHead, ProjectorWithWarp
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


# ---------------------------------------------------------------------------
# return_edges support
# ---------------------------------------------------------------------------


def test_residual_warp_return_edges_active_path():
    warp = ResidualFastKANWarp(
        input_dim=8, hidden_dim=4, num_centers=6, alpha_init=0.1, learnable_alpha=True
    )
    z = torch.randn(5, 8)

    out, phi = warp(z, return_edges=True)

    assert out.shape == (5, 8)
    assert torch.allclose(out.norm(dim=-1), torch.ones(5), atol=1e-5)
    # KAN last-layer edges: [batch, output_dim=input_dim, input_dim=hidden_dim].
    assert phi.shape == (5, 8, 4)
    assert torch.isfinite(phi).all()


def test_residual_warp_return_edges_zeros_when_bypassed():
    warp = ResidualFastKANWarp(
        input_dim=8, hidden_dim=4, alpha_init=0.0, learnable_alpha=False
    )
    z = torch.randn(3, 8)

    out, phi = warp(z, return_edges=True)

    assert torch.equal(out, F.normalize(z, dim=-1, eps=1e-12))
    assert phi.shape == (3, 8, 4)
    assert torch.count_nonzero(phi) == 0


# ---------------------------------------------------------------------------
# ProjectorWithWarp composite head (H1 fix: 128-D output, not 512-D)
# ---------------------------------------------------------------------------


def _mlp_projector(out_dim: int = 128) -> MLPHead:
    return MLPHead(
        in_dim=512,
        hidden_dim=512,
        output_dim=out_dim,
        use_batch_norm=True,
        l2_normalize=False,
    )


def test_composite_output_dim_equals_warp_width_not_512():
    head = ProjectorWithWarp(
        _mlp_projector(128),
        ResidualFastKANWarp(input_dim=128, hidden_dim=16, num_centers=8),
    )
    h = torch.randn(4, 512)

    z = head(h)

    assert z.shape == (4, 128)
    assert torch.allclose(z.norm(dim=-1), torch.ones(4), atol=1e-5)


def test_composite_return_edges_come_from_warp():
    head = ProjectorWithWarp(
        _mlp_projector(128),
        ResidualFastKANWarp(
            input_dim=128, hidden_dim=16, num_centers=8, alpha_init=0.1
        ),
    )
    h = torch.randn(4, 512)

    z, phi = head(h, return_edges=True)

    assert z.shape == (4, 128)
    assert phi.shape == (4, 128, 16)


def test_composite_warp_none_equals_projector_alone():
    projector = MLPHead(
        in_dim=512, hidden_dim=32, output_dim=128, use_batch_norm=False
    )
    head = ProjectorWithWarp(projector, warp=None)
    h = torch.randn(3, 512)

    assert torch.equal(head(h), projector(h))


def test_composite_alpha_property_exposes_warp_alpha():
    warp = ResidualFastKANWarp(input_dim=128, hidden_dim=16, alpha_init=0.05)
    with_warp = ProjectorWithWarp(_mlp_projector(128), warp)
    without_warp = ProjectorWithWarp(_mlp_projector(128), warp=None)

    assert with_warp.alpha is not None
    assert torch.equal(with_warp.alpha, warp.alpha)
    assert without_warp.alpha is None
