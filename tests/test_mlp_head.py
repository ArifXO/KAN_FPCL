"""Tests for MLPHead projection module (Stage 2)."""

import torch

from src.models import MLPHead


def test_output_shape():
    head = MLPHead(in_dim=512, hidden_dim=256, output_dim=128)
    x = torch.randn(16, 512)
    z = head(x)
    assert z.shape == (16, 128)


def test_l2_normalize_produces_unit_vectors():
    head = MLPHead(in_dim=512, hidden_dim=256, output_dim=128, l2_normalize=True)
    head.eval()
    z = head(torch.randn(8, 512))
    norms = z.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_no_l2_when_disabled():
    head = MLPHead(in_dim=512, hidden_dim=256, output_dim=128, l2_normalize=False)
    head.eval()
    z = head(torch.randn(8, 512))
    # Not all unit norm (with overwhelming probability).
    assert not torch.allclose(z.norm(dim=-1), torch.ones(8), atol=1e-2)


def test_parameter_count_matches_manual():
    """Verify parameter_count() against a hand calculation."""
    in_dim, hidden, out_dim = 32, 64, 16
    head = MLPHead(in_dim=in_dim, hidden_dim=hidden, output_dim=out_dim,
                   use_batch_norm=True, l2_normalize=True)
    # Linear1: in*hidden + hidden = 32*64 + 64 = 2112
    # BN: 2 * hidden = 128 (weight + bias; running stats are buffers, not params)
    # Linear2: hidden*out + out = 64*16 + 16 = 1040
    expected = (in_dim * hidden + hidden) + (2 * hidden) + (hidden * out_dim + out_dim)
    assert head.parameter_count() == expected, (
        f"expected {expected} params, got {head.parameter_count()}"
    )


def test_parameter_count_without_bn():
    head = MLPHead(in_dim=32, hidden_dim=64, output_dim=16, use_batch_norm=False)
    expected = (32 * 64 + 64) + (64 * 16 + 16)
    assert head.parameter_count() == expected


def test_forward_with_batch_norm_requires_batch_gt1():
    """BatchNorm1d needs >1 in training mode; eval mode handles single sample."""
    head = MLPHead(in_dim=8, hidden_dim=16, output_dim=4, use_batch_norm=True)
    head.eval()
    z = head(torch.randn(1, 8))
    assert z.shape == (1, 4)


def test_gradient_flow():
    head = MLPHead(in_dim=16, hidden_dim=32, output_dim=8)
    x = torch.randn(4, 16, requires_grad=True)
    z = head(x)
    z.sum().backward()
    assert x.grad is not None
    for name, p in head.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
