"""Tests for src.losses.edge_features (Stage 7.5 — H4)."""

from __future__ import annotations

import pytest
import torch

from src.losses.edge_features import edge_fingerprint, edge_pair_similarity


def test_edge_fingerprint_shape_and_l2_norm():
    phi = torch.randn(6, 8, 32)            # 6 samples, O=8, I=32 -> O*I=256
    f = edge_fingerprint(phi)
    assert f.shape == (6, 256)
    norms = f.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(6), atol=1e-5)


def test_edge_fingerprint_pads_when_small():
    phi = torch.randn(4, 4, 8)             # O*I=32 < 256 -> right-pad with zeros
    f = edge_fingerprint(phi)
    assert f.shape == (4, 256)
    # The padded tail must contain only zeros (L2-norm is unaffected because
    # `pooled` is non-zero and normalization happens after pad).
    # We can't check exact zeros after L2 normalization because padding 0s
    # become 0/(non-zero norm) = 0 — confirm directly.
    assert (f[:, 32:] == 0).all(), "padded entries must be exact zero after L2-norm."


def test_edge_fingerprint_invalid_shape_raises():
    with pytest.raises(ValueError, match=r"\[B, O, I\]"):
        edge_fingerprint(torch.randn(6, 8))


def test_edge_fingerprint_zero_input_no_nan():
    """STABILITY (Pitfall 3): zero edges must not produce NaN."""
    phi = torch.zeros(4, 8, 32)
    f = edge_fingerprint(phi)
    assert torch.isfinite(f).all()
    assert (f.norm(dim=-1) <= 1.0 + 1e-6).all()
    # All-zero output is expected (eps-clamp inside F.normalize).
    assert (f == 0).all()


def test_edge_pair_similarity_symmetric():
    f = torch.nn.functional.normalize(torch.randn(5, 256), dim=-1)
    s = edge_pair_similarity(f)
    assert s.shape == (5, 5)
    assert torch.allclose(s, s.t(), atol=1e-6)


def test_edge_pair_similarity_identical_is_one():
    f = torch.randn(3, 256)
    f = torch.nn.functional.normalize(f, dim=-1)
    s = edge_pair_similarity(f)
    diag = s.diagonal()
    assert torch.allclose(diag, torch.ones(3), atol=1e-5)


def test_edge_pair_similarity_bounded_in_unit_range():
    f = torch.randn(7, 256)
    s = edge_pair_similarity(f)
    assert (s >= -1.0 - 1e-5).all() and (s <= 1.0 + 1e-5).all()


def test_edge_pair_similarity_invalid_shape_raises():
    with pytest.raises(ValueError, match=r"\[B, D\]"):
        edge_pair_similarity(torch.randn(5))
