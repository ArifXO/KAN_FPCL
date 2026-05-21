"""Tests for FN-weighted InfoNCE loss (R3, R7, R9 — Stage 6 / H2)."""

from __future__ import annotations

import pytest
import torch

from src.losses import FNWeightedInfoNCELoss, InfoNCELoss


def _two_view_batch(batch: int = 4, dim: int = 8, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v1 = torch.randn(batch, dim, generator=g)
    v2 = v1 + 0.1 * torch.randn(batch, dim, generator=g)
    return torch.cat([v1, v2], dim=0)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_temperature_validation():
    with pytest.raises(ValueError, match="temperature must be > 0"):
        FNWeightedInfoNCELoss(temperature=0.0)
    with pytest.raises(ValueError, match="temperature must be > 0"):
        FNWeightedInfoNCELoss(temperature=-0.1)


def test_max_fn_weight_validation():
    with pytest.raises(ValueError, match="max_fn_weight must be in"):
        FNWeightedInfoNCELoss(max_fn_weight=-0.1)
    with pytest.raises(ValueError, match="max_fn_weight must be in"):
        FNWeightedInfoNCELoss(max_fn_weight=1.5)


# ---------------------------------------------------------------------------
# Input validation (R9: descriptive errors)
# ---------------------------------------------------------------------------


def test_invalid_z_shape_raises():
    loss_fn = FNWeightedInfoNCELoss()
    with pytest.raises(ValueError, match=r"\[2B, D\]"):
        loss_fn(torch.randn(8), torch.zeros(4, 4))
    with pytest.raises(ValueError, match="2B with B>=2"):
        loss_fn(torch.randn(3, 8), torch.zeros(1, 1))


def test_p_fn_shape_mismatch_raises():
    loss_fn = FNWeightedInfoNCELoss()
    z = _two_view_batch(batch=4, dim=8)
    with pytest.raises(ValueError, match=r"\[B, B\] with B=4"):
        loss_fn(z, torch.zeros(3, 3))


def test_p_fn_nan_raises():
    loss_fn = FNWeightedInfoNCELoss()
    z = _two_view_batch(batch=4, dim=8)
    p = torch.zeros(4, 4)
    p[0, 1] = float("nan")
    with pytest.raises(ValueError, match="p_fn contains NaN"):
        loss_fn(z, p)


def test_p_fn_out_of_bounds_raises():
    loss_fn = FNWeightedInfoNCELoss()
    z = _two_view_batch(batch=4, dim=8)
    p_lo = torch.zeros(4, 4)
    p_lo[0, 1] = -0.01
    with pytest.raises(ValueError, match=r"p_fn must lie in \[0, 1\]"):
        loss_fn(z, p_lo)

    p_hi = torch.zeros(4, 4)
    p_hi[0, 1] = 1.01
    with pytest.raises(ValueError, match=r"p_fn must lie in \[0, 1\]"):
        loss_fn(z, p_hi)


# ---------------------------------------------------------------------------
# R7: required dict keys
# ---------------------------------------------------------------------------


def test_returns_required_dict_keys():
    loss_fn = FNWeightedInfoNCELoss(temperature=0.1)
    z = _two_view_batch(batch=4, dim=8)
    p = torch.rand(4, 4) * 0.3
    out = loss_fn(z, p)
    for key in (
        "loss",
        "pos_sim_mean",
        "neg_sim_mean",
        "p_fn_mean",
        "p_fn_max",
        "downweighted_fraction",
        "temperature",
    ):
        assert key in out, f"missing key {key}"
    assert out["loss"].dim() == 0
    assert torch.isfinite(out["loss"]).item()


# ---------------------------------------------------------------------------
# Equivalence to InfoNCE when p_fn = 0 everywhere
# ---------------------------------------------------------------------------


def test_p_fn_zero_equals_infonce():
    """R3: p_fn=0 means no downweighting -> exact InfoNCE."""
    z = _two_view_batch(batch=4, dim=8, seed=11)
    infonce = InfoNCELoss(temperature=0.1, normalize_embeddings=True)
    fn_loss = FNWeightedInfoNCELoss(
        temperature=0.1, normalize_embeddings=True, max_fn_weight=1.0
    )
    p_zero = torch.zeros(4, 4)
    a = infonce(z)["loss"]
    b = fn_loss(z, p_zero)["loss"]
    assert torch.allclose(a, b, atol=1e-5), f"InfoNCE={a.item()}, FN={b.item()}"


# ---------------------------------------------------------------------------
# R3 positive/negative/FN cases
# ---------------------------------------------------------------------------


def test_r3_positive_only_loss_near_zero():
    """When views are identical (perfect positives), loss → 0."""
    base = torch.randn(8, 16)
    z = torch.cat([base, base.clone()], dim=0)
    p_zero = torch.zeros(8, 8)
    out = FNWeightedInfoNCELoss(temperature=0.05)(z, p_zero)
    assert out["loss"].item() < 0.05


def test_r3_negative_only_loss_finite_and_positive():
    """Adversarial alignment (positives orthogonal to their match) -> loss >> 0."""
    n = 4
    v1 = torch.eye(n, n)
    v2 = torch.roll(v1, shifts=1, dims=0)
    z = torch.cat([v1, v2], dim=0)
    p_zero = torch.zeros(n, n)
    out = FNWeightedInfoNCELoss(temperature=0.1)(z, p_zero)
    assert torch.isfinite(out["loss"]).item()
    assert out["loss"].item() > 1.0


def test_r3_fn_case_marking_true_positive_increases_loss():
    """R3 FN-case: marking a real positive as a negative (p_fn=0 -> still in
    denominator at full weight) yields a HIGHER loss than removing it from
    the denominator (p_fn=1 -> downweighted away).

    Equivalent statement: the loss must respond to the FN mask. We compare
    full p_fn=0 vs p_fn=1 in negative slots — the FN-aware version must NOT
    just ignore the mask.
    """
    z = _two_view_batch(batch=4, dim=16, seed=3)
    loss_fn = FNWeightedInfoNCELoss(temperature=0.1, max_fn_weight=1.0)
    p_off = torch.zeros(4, 4)
    p_on = torch.ones(4, 4)
    l_off = loss_fn(z, p_off)["loss"].item()
    l_on = loss_fn(z, p_on)["loss"].item()
    assert l_off > l_on, (
        f"FN mask had no effect: p_fn=0 loss={l_off:.4f}, p_fn=1 loss={l_on:.4f}"
    )


# ---------------------------------------------------------------------------
# Saturation at p_fn = 1
# ---------------------------------------------------------------------------


def test_p_fn_one_loss_near_zero():
    """p_fn=1 everywhere removes all negatives -> loss ≈ 0."""
    z = _two_view_batch(batch=4, dim=16, seed=5)
    p_ones = torch.ones(4, 4)
    out = FNWeightedInfoNCELoss(
        temperature=0.1, normalize_embeddings=True, max_fn_weight=1.0
    )(z, p_ones)
    assert out["loss"].item() < 1e-3, (
        f"p_fn=1 should null the denominator, got loss={out['loss'].item():.4f}"
    )


# ---------------------------------------------------------------------------
# Monotonicity in p_fn
# ---------------------------------------------------------------------------


def test_monotonic_decrease_with_p_fn():
    """Increasing p_fn on negatives monotonically decreases the loss."""
    z = _two_view_batch(batch=6, dim=16, seed=7)
    loss_fn = FNWeightedInfoNCELoss(temperature=0.1, max_fn_weight=1.0)
    losses = [
        loss_fn(z, torch.full((6, 6), p))["loss"].item()
        for p in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    for a, b in zip(losses, losses[1:]):
        assert a >= b - 1e-6, f"non-monotonic: {losses}"
    assert losses[0] > losses[-1], f"no overall decrease: {losses}"


# ---------------------------------------------------------------------------
# Gradient flow (R9-adjacent)
# ---------------------------------------------------------------------------


def test_gradient_flows_to_z_and_p_fn():
    """Gradients reach both the embedding tensor and the p_fn input."""
    z = _two_view_batch(batch=4, dim=8, seed=13).clone().requires_grad_(True)
    p_fn = (torch.rand(4, 4) * 0.4).requires_grad_(True)
    out = FNWeightedInfoNCELoss(temperature=0.2)(z, p_fn)
    out["loss"].backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert z.grad.abs().sum() > 0
    assert p_fn.grad is not None and torch.isfinite(p_fn.grad).all()


def test_loss_decreases_with_gradient_step_on_z():
    z = _two_view_batch(batch=8, dim=16, seed=1).clone().requires_grad_(True)
    p_fn = torch.zeros(8, 8)
    loss_fn = FNWeightedInfoNCELoss(temperature=0.5)
    optim = torch.optim.Adam([z], lr=0.1)

    out0 = loss_fn(z, p_fn)
    l0 = out0["loss"].item()
    optim.zero_grad()
    out0["loss"].backward()
    optim.step()
    with torch.no_grad():
        l1 = loss_fn(z, p_fn)["loss"].item()
    assert l1 < l0, f"loss did not decrease: {l0:.4f} -> {l1:.4f}"


# ---------------------------------------------------------------------------
# max_fn_weight clipping behavior
# ---------------------------------------------------------------------------


def test_max_fn_weight_caps_downweighting():
    """With max_fn_weight=0.5 and p_fn=1.0, weight is floored at 0.5 — loss
    must be strictly greater than the uncapped (max_fn_weight=1.0) case."""
    z = _two_view_batch(batch=4, dim=16, seed=21)
    p_ones = torch.ones(4, 4)
    uncapped = FNWeightedInfoNCELoss(temperature=0.1, max_fn_weight=1.0)(z, p_ones)
    capped = FNWeightedInfoNCELoss(temperature=0.1, max_fn_weight=0.5)(z, p_ones)
    assert capped["loss"].item() > uncapped["loss"].item() + 1e-4, (
        f"capping had no effect: capped={capped['loss'].item():.4f}, "
        f"uncapped={uncapped['loss'].item():.4f}"
    )
