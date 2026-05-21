"""Tests for FastKAN projection modules."""

import pytest
import torch

from src.models import MLPHead
from src.models.kan import FastKANLayer, FastKANProjector


def test_fastkan_layer_output_shape_and_edges():
    layer = FastKANLayer(input_dim=6, output_dim=4, num_centers=5)
    x = torch.randn(3, 6)

    y = layer(x)
    y_with_edges, phi = layer(x, return_edges=True)

    assert y.shape == (3, 4)
    assert y_with_edges.shape == (3, 4)
    assert phi.shape == (3, 4, 6)
    assert torch.allclose(y, y_with_edges)


def test_fastkan_layer_phi_matches_definition():
    layer = FastKANLayer(
        input_dim=2,
        output_dim=1,
        num_centers=3,
        grid_min=-1.0,
        grid_max=1.0,
        use_base_linear=False,
    )
    with torch.no_grad():
        layer.bandwidth.fill_(1.0)
        layer.rbf_weight.copy_(
            torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
        )

    x = torch.tensor([[0.0, 1.0]])
    y, phi = layer(x, return_edges=True)

    rbf = torch.exp(-((x.unsqueeze(-1) - layer.centers) / layer.bandwidth) ** 2)
    expected_phi = torch.einsum("bic,oic->boi", rbf, layer.rbf_weight)
    assert torch.allclose(phi, expected_phi)
    assert torch.allclose(y, expected_phi.sum(dim=-1))


def test_rbf_centers_are_fixed_buffers():
    layer = FastKANLayer(input_dim=4, output_dim=3)
    expected_params = (3 * 4 * 8) + 1 + (4 * 3 + 3)
    assert layer.parameter_count() == expected_params
    assert "centers" not in dict(layer.named_parameters())
    assert "centers" in dict(layer.named_buffers())

    before = layer.centers.detach().clone()
    opt = torch.optim.SGD(layer.parameters(), lr=0.1)
    loss = layer(torch.randn(5, 4)).sum()
    loss.backward()
    opt.step()

    assert torch.allclose(layer.centers, before)


def test_fastkan_projector_l2_normalizes_outputs():
    head = FastKANProjector(input_dim=8, hidden_dim=5, output_dim=3, normalize=True)
    z = head(torch.randn(4, 8))
    norms = z.norm(dim=-1)
    assert z.shape == (4, 3)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_fastkan_projector_return_edges_and_num_edges():
    head = FastKANProjector(
        input_dim=8,
        hidden_dim=5,
        output_dim=3,
        num_centers=4,
        normalize=False,
    )
    x = torch.randn(4, 8)

    z = head(x)
    z_with_edges, phi = head(x, return_edges=True)

    assert z.shape == (4, 3)
    assert z_with_edges.shape == (4, 3)
    assert phi.shape == (4, 3, 5)
    assert head.num_edges == 15
    assert torch.allclose(z, z_with_edges)


def test_fastkan_projector_parameter_count_matches_mlp_baseline():
    mlp = MLPHead(in_dim=512, hidden_dim=512, output_dim=128, use_batch_norm=True)
    kan = FastKANProjector(input_dim=512, hidden_dim=57, output_dim=128)

    mlp_params = mlp.parameter_count()
    kan_params = kan.parameter_count()
    relative_delta = abs(kan_params - mlp_params) / mlp_params

    assert kan_params == 328507
    assert relative_delta <= 0.15


def test_fastkan_projector_gradient_flow():
    head = FastKANProjector(
        input_dim=6,
        hidden_dim=4,
        output_dim=3,
        num_centers=5,
        normalize=False,
    )
    x = torch.randn(3, 6, requires_grad=True)

    loss = head(x).pow(2).mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for name, param in head.named_parameters():
        assert param.grad is not None, f"no grad for {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite grad for {name}"


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"input_dim": 0, "output_dim": 3}, "input_dim"),
        ({"input_dim": 4, "output_dim": 0}, "output_dim"),
        ({"input_dim": 4, "output_dim": 3, "num_centers": 0}, "num_centers"),
        ({"input_dim": True, "output_dim": 3}, "input_dim"),
        ({"input_dim": 4, "output_dim": 3, "grid_min": "bad"}, "grid_min"),
        ({"input_dim": 4, "output_dim": 3, "grid_min": 2.0}, "grid_min"),
        (
            {"input_dim": 4, "output_dim": 3, "rbf_init_scale": float("inf")},
            "finite",
        ),
        ({"input_dim": 4, "output_dim": 3, "rbf_init_scale": 0.0}, "rbf_init"),
        ({"input_dim": 4, "output_dim": 3, "base_activation": "bad"}, "base"),
        ({"input_dim": 4, "output_dim": 3, "base_activation": 1}, "base"),
    ],
)
def test_fastkan_layer_validation_errors(kwargs, match):
    with pytest.raises(ValueError, match=match):
        FastKANLayer(**kwargs)


def test_fastkan_projector_validation_errors():
    with pytest.raises(ValueError, match="hidden_dim"):
        FastKANProjector(input_dim=4, hidden_dim=0, output_dim=3)


def test_fastkan_layer_rejects_wrong_input_shape():
    layer = FastKANLayer(input_dim=4, output_dim=3)
    with pytest.raises(ValueError, match="expected input shape"):
        layer(torch.randn(2, 5))
