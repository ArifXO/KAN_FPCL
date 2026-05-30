"""Tests for Debiased Contrastive Learning (R3, R7, R9)."""

from __future__ import annotations

import pytest
import torch

from src.losses import DebiasedContrastiveLoss, InfoNCELoss


_REQUIRED_KEYS = {
    "loss", "pos_sim_mean", "neg_sim_mean",
    "pos_exp_mean", "neg_exp_mean", "debiased_neg_mean",
    "tau_plus", "temperature",
}


def _two_view_batch(batch: int = 4, dim: int = 8, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v1 = torch.randn(batch, dim, generator=g)
    v2 = v1 + 0.1 * torch.randn(batch, dim, generator=g)
    return torch.cat([v1, v2], dim=0).requires_grad_(True)


# ---------------------------------------------------------------------------
# Constructor validation (R9)
# ---------------------------------------------------------------------------


def test_temperature_validation():
    with pytest.raises(ValueError, match="temperature must be > 0"):
        DebiasedContrastiveLoss(temperature=0.0)
    with pytest.raises(ValueError, match="temperature must be > 0"):
        DebiasedContrastiveLoss(temperature=-0.5)


def test_tau_plus_validation():
    with pytest.raises(ValueError, match="tau_plus"):
        DebiasedContrastiveLoss(tau_plus=-0.01)
    with pytest.raises(ValueError, match="tau_plus"):
        DebiasedContrastiveLoss(tau_plus=1.0)


def test_estimator_clip_min_validation():
    with pytest.raises(ValueError, match="estimator_clip_min"):
        DebiasedContrastiveLoss(estimator_clip_min=0.0)


# ---------------------------------------------------------------------------
# Input validation (R9)
# ---------------------------------------------------------------------------


def test_invalid_batch_shape_raises():
    loss_fn = DebiasedContrastiveLoss()
    with pytest.raises(ValueError, match=r"\[2B, D\]"):
        loss_fn(torch.randn(8))
    with pytest.raises(ValueError, match="even batch"):
        loss_fn(torch.randn(3, 8))


def test_non_finite_z_raises():
    loss_fn = DebiasedContrastiveLoss()
    z = torch.zeros(8, 4)
    z[0, 0] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        loss_fn(z)


def test_forward_does_not_require_labels():
    """Label-free contract: forward() must not depend on labels."""
    loss_fn = DebiasedContrastiveLoss()
    out = loss_fn(_two_view_batch())
    assert "loss" in out


# ---------------------------------------------------------------------------
# Dict / shape contract (R7)
# ---------------------------------------------------------------------------


def test_returns_required_dict_keys():
    loss_fn = DebiasedContrastiveLoss()
    out = loss_fn(_two_view_batch())
    missing = _REQUIRED_KEYS - set(out.keys())
    assert not missing, f"missing keys: {missing}"


def test_loss_is_finite_scalar():
    loss_fn = DebiasedContrastiveLoss(temperature=0.5, tau_plus=0.1)
    out = loss_fn(_two_view_batch())
    loss = out["loss"]
    assert loss.dim() == 0
    assert torch.isfinite(loss).item()
    assert loss.item() > 0


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_gradient_flows_to_input():
    z = _two_view_batch()
    out = DebiasedContrastiveLoss(temperature=0.5)(z)
    out["loss"].backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert z.grad.abs().sum() > 0


def test_gradient_step_decreases_loss():
    z = _two_view_batch(batch=8, dim=16, seed=1)
    loss_fn = DebiasedContrastiveLoss(temperature=0.5, tau_plus=0.1)
    optim = torch.optim.Adam([z], lr=0.1)

    out0 = loss_fn(z)
    loss0 = out0["loss"].item()
    optim.zero_grad()
    out0["loss"].backward()
    optim.step()

    with torch.no_grad():
        loss1 = loss_fn(z)["loss"].item()
    assert loss1 < loss0, f"loss did not decrease: {loss0:.4f} -> {loss1:.4f}"


# ---------------------------------------------------------------------------
# Estimator behavior
# ---------------------------------------------------------------------------


def test_debiased_neg_clamped_never_nan():
    """Adversarial setup: huge tau_plus to force the raw estimator negative.

    Even with tau_plus close to 1, ``debiased_neg`` must stay >= clip and
    ``loss`` must stay finite.
    """
    loss_fn = DebiasedContrastiveLoss(
        temperature=0.5, tau_plus=0.99, estimator_clip_min=1e-8,
    )
    z = _two_view_batch(batch=8, dim=16)
    out = loss_fn(z)
    assert torch.isfinite(out["loss"]).item()
    # float32 round-trip on 1e-8 lands at ~9.9999999e-9; use a tolerant bound.
    assert out["debiased_neg_mean"].item() >= 9e-9


def test_tau_plus_zero_recovers_unbiased_denominator():
    """At tau_plus=0, the estimator is the un-debiased neg-sum: the loss
    differs from InfoNCE only by the diagonal-exclusion convention (which
    matches InfoNCE's masked_fill).

    Both losses use ``temperature`` and exclude the diagonal; the only
    difference at tau_plus=0 is whether the positive pair is in the
    denominator. Debiased CL puts ``pos_exp`` in the denominator (the +
    log term), InfoNCE puts it in via cross-entropy. We just check that
    Debiased CL still ranks the same pair as 'best' — pos_sim_mean still
    exceeds neg_sim_mean — and that the loss is finite.
    """
    z = _two_view_batch(batch=8, dim=16, seed=3)
    out = DebiasedContrastiveLoss(temperature=0.5, tau_plus=0.0)(z)
    assert torch.isfinite(out["loss"]).item()
    assert out["pos_sim_mean"].item() > out["neg_sim_mean"].item()


def test_perfect_positives_give_low_loss():
    """When v1 == v2 exactly, the positive dominates -> loss is small."""
    base = torch.randn(8, 16)
    z = torch.cat([base, base.clone()], dim=0)
    out = DebiasedContrastiveLoss(temperature=0.1, tau_plus=0.05)(z)
    assert out["loss"].item() < 0.5
