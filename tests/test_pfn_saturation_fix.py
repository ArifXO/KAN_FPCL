"""Tests for the Pitfall #5 saturation mitigations.

Covers (per scripts/train spec sections A, B, C, D, E, G):

* G.1  p_fn=0 still equals InfoNCE (allclose 1e-5).
* G.2  max_fn_weight_current=0 (warmup) makes FN loss == InfoNCE.
* G.3  Scorer initial mean p_fn near init_pfn_prior (covered also in
       test_pair_scorer.py for MLPPairScorer; here we add KAN + EdgeAware).
* G.4  p_fn_at_cap_fraction is computed correctly on a synthetic tensor.
* G.5  Mean reg ≈ 0 when mean equals target; positive when far from target.
* G.6  Cap penalty increases when many values cluster near the cap.
* G.7  base_step_metrics row carries raw/clipped + reg keys.
* G.8  Edge-aware all-lambdas-zero still numerically matches Stage 7.
* Schedule helper produces InfoNCE during warmup, then ramps.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

# Ensure scripts/train/ is importable for the schedule helper.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "train"))

from src.losses import (
    EdgeAwareFNWeightedInfoNCELoss,
    FNWeightedInfoNCELoss,
    InfoNCELoss,
)
from src.losses.pfn_diagnostics import pfn_diagnostics, pfn_regularization
from src.models import (
    EdgeAwarePairScorer,
    KANPairScorer,
    MLPPairScorer,
)
from train_common import base_step_metrics, compute_max_fn_weight_current


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _two_view_batch(b=4, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    v1 = torch.randn(b, d, generator=g)
    v2 = v1 + 0.1 * torch.randn(b, d, generator=g)
    return torch.cat([v1, v2], dim=0)


# ---------------------------------------------------------------------------
# G.1 — p_fn=0 backward compat
# ---------------------------------------------------------------------------


def test_pfn_zero_matches_infonce_with_default_reg_off():
    """With pfn_regularization off, p_fn=0 reproduces InfoNCE exactly."""
    z = _two_view_batch(seed=42)
    infonce = InfoNCELoss(temperature=0.1, normalize_embeddings=True)
    fn = FNWeightedInfoNCELoss(
        temperature=0.1, normalize_embeddings=True, max_fn_weight=1.0,
        pfn_regularization=None,
    )
    a = infonce(z)["loss"]
    b = fn(z, torch.zeros(4, 4))["loss"]
    assert torch.allclose(a, b, atol=1e-5)


def test_pfn_zero_loss_key_unchanged_even_with_reg_enabled():
    """`loss` key (contrastive only) must equal InfoNCE even when reg is on.

    The reg lives in `pfn_reg_total`; the train script adds it to the
    backprop objective. The R3 InfoNCE backward-compat invariant applies to
    `loss`, NOT to the train-script's optimizer target.
    """
    z = _two_view_batch(seed=7)
    infonce = InfoNCELoss(temperature=0.1, normalize_embeddings=True)
    fn = FNWeightedInfoNCELoss(
        temperature=0.1, normalize_embeddings=True, max_fn_weight=1.0,
        pfn_regularization={
            "enabled": True, "target_mean": 0.05,
            "lambda_mean": 0.1, "lambda_cap": 0.1,
        },
    )
    a = infonce(z)["loss"]
    b = fn(z, torch.zeros(4, 4))["loss"]
    assert torch.allclose(a, b, atol=1e-5)


# ---------------------------------------------------------------------------
# G.2 — warmup (max_cap=0) == InfoNCE
# ---------------------------------------------------------------------------


def test_warmup_override_zero_matches_infonce():
    """max_fn_weight_override=0 floors the cap → all weights become 1 →
    behavior identical to InfoNCE for ANY p_fn input."""
    z = _two_view_batch(seed=11)
    fn = FNWeightedInfoNCELoss(
        temperature=0.1, normalize_embeddings=True, max_fn_weight=0.5,
    )
    p_random = torch.rand(4, 4) * 0.4
    out = fn(z, p_random, max_fn_weight_override=0.0)
    infonce = InfoNCELoss(temperature=0.1, normalize_embeddings=True)(z)["loss"]
    assert torch.allclose(out["loss"], infonce, atol=1e-5)
    assert out["max_fn_weight_current"].item() == 0.0


def test_override_validation():
    fn = FNWeightedInfoNCELoss()
    z = _two_view_batch()
    p = torch.zeros(4, 4)
    with pytest.raises(ValueError, match="max_fn_weight_override"):
        fn(z, p, max_fn_weight_override=1.5)


# ---------------------------------------------------------------------------
# G.3 — init_pfn_prior on all three scorers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scorer_cls,kwargs", [
    (MLPPairScorer, {"input_dim": 32, "hidden_dim": 16}),
    (KANPairScorer, {"input_dim": 32, "hidden_dim": 5, "num_centers": 6}),
    (EdgeAwarePairScorer, {"input_dim": 32, "use_edge_features": False, "scorer_type": "mlp"}),
])
def test_init_pfn_prior_starts_near_target(scorer_cls, kwargs):
    torch.manual_seed(0)
    s = scorer_cls(init_pfn_prior=0.05, **kwargs)
    z = torch.randn(64, 32)
    p = s(z)
    assert abs(p.mean().item() - 0.05) < 0.05, (
        f"{scorer_cls.__name__} mean={p.mean().item()}"
    )
    # Sanity: also not above the saturation-prone 0.5.
    assert p.mean().item() < 0.2


# ---------------------------------------------------------------------------
# G.4 — at_cap_fraction on a known tensor
# ---------------------------------------------------------------------------


def test_pfn_diagnostics_at_cap_fraction_known():
    # 16-entry tensor: 4 at exactly the cap (0.5), 12 below.
    p = torch.tensor([0.5, 0.5, 0.5, 0.5] + [0.1] * 12).reshape(4, 4)
    diag = pfn_diagnostics(p, p.clamp(0, 0.5), max_fn_weight_current=0.5)
    assert diag["p_fn_at_cap_fraction"].item() == pytest.approx(4 / 16, abs=1e-6)


def test_pfn_diagnostics_at_cap_zero_when_cap_is_zero():
    """With max_cap=0 the cap is inactive (warmup); at_cap_fraction = 0."""
    p = torch.full((4, 4), 0.3)
    diag = pfn_diagnostics(p, p.clamp(0, 0), max_fn_weight_current=0.0)
    assert diag["p_fn_at_cap_fraction"].item() == 0.0


def test_pfn_diagnostics_near_zero_fraction():
    # 4 entries strictly ≤ 0.05 (the near-zero threshold), 12 above.
    p = torch.tensor([0.0, 0.01, 0.04, 0.05, 0.5, 0.5, 0.5, 0.5,
                       0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]).reshape(4, 4)
    diag = pfn_diagnostics(p, p.clamp(0, 0.5), max_fn_weight_current=0.5)
    assert diag["p_fn_near_zero_fraction"].item() == pytest.approx(4 / 16, abs=1e-6)


# ---------------------------------------------------------------------------
# G.5 — mean regularization is zero at target, positive when off
# ---------------------------------------------------------------------------


def test_mean_reg_zero_at_target():
    p = torch.full((4, 4), 0.05)
    reg = pfn_regularization(p, 0.5, {
        "enabled": True, "target_mean": 0.05, "lambda_mean": 1.0,
    })
    assert reg["loss_pfn_mean"].item() == pytest.approx(0.0, abs=1e-8)


def test_mean_reg_positive_when_above_target():
    p = torch.full((4, 4), 0.45)  # well above target
    reg = pfn_regularization(p, 0.5, {
        "enabled": True, "target_mean": 0.05, "lambda_mean": 1.0,
    })
    expected = (0.45 - 0.05) ** 2
    assert reg["loss_pfn_mean"].item() == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# G.6 — cap penalty rises as values crowd the cap
# ---------------------------------------------------------------------------


def test_cap_penalty_increases_near_cap():
    low = torch.full((4, 4), 0.1)
    high = torch.full((4, 4), 0.49)  # just shy of cap=0.5
    cfg = {
        "enabled": True, "lambda_cap": 1.0, "cap_margin": 0.5,
        "target_mean": 0.05, "lambda_mean": 0.0,
    }
    reg_low = pfn_regularization(low, 0.5, cfg)["loss_pfn_cap"].item()
    reg_high = pfn_regularization(high, 0.5, cfg)["loss_pfn_cap"].item()
    assert reg_high > reg_low + 1e-6, f"low={reg_low}, high={reg_high}"


def test_cap_penalty_zero_when_warmup():
    """During warmup (cap=0) the cap penalty is skipped."""
    p = torch.full((4, 4), 0.49)
    reg = pfn_regularization(p, 0.0, {
        "enabled": True, "lambda_cap": 1.0, "cap_margin": 0.5,
    })
    assert reg["loss_pfn_cap"].item() == 0.0


# ---------------------------------------------------------------------------
# G.7 — base_step_metrics carries the diagnostic + reg keys
# ---------------------------------------------------------------------------


def test_base_step_metrics_includes_new_diag_and_reg_keys():
    z = _two_view_batch(seed=3)
    fn = FNWeightedInfoNCELoss(
        max_fn_weight=0.5,
        pfn_regularization={
            "enabled": True, "target_mean": 0.05,
            "lambda_mean": 0.1, "lambda_cap": 0.1,
        },
    )
    p = torch.full((4, 4), 0.3)
    out = fn(z, p, max_fn_weight_override=0.5)
    row = base_step_metrics(step=0, lr=1e-3, out=out, epoch=1)
    for k in (
        "p_fn_raw_mean", "p_fn_raw_std", "p_fn_raw_min", "p_fn_raw_max",
        "p_fn_clipped_mean", "p_fn_clipped_std",
        "p_fn_at_cap_fraction", "p_fn_near_zero_fraction",
        "p_fn_entropy_mean",
        "effective_neg_weight_mean", "effective_neg_weight_min",
        "max_fn_weight_current",
        "loss_pfn_mean", "loss_pfn_cap", "loss_pfn_entropy",
        "pfn_reg_total", "target_pfn_mean",
    ):
        assert k in row, f"missing step-metrics key: {k}"


# ---------------------------------------------------------------------------
# G.8 — edge-aware backward compat (lambdas=0) still equals Stage 7
# ---------------------------------------------------------------------------


def test_edge_aware_lambdas_zero_matches_fn_loss():
    torch.manual_seed(0)
    z = torch.randn(8, 16)
    p_fn = torch.rand(4, 4) * 0.3
    stage7 = FNWeightedInfoNCELoss(temperature=0.1, max_fn_weight=0.5)
    new = EdgeAwareFNWeightedInfoNCELoss(
        temperature=0.1, max_fn_weight=0.5,
        lambda_edge=0.0, lambda_edge_align=0.0,
    )
    out_old = stage7(z, p_fn)
    out_new = new(z, p_fn, None)
    assert torch.allclose(out_old["loss"], out_new["loss"], atol=1e-5)


def test_edge_aware_forwards_override():
    """The edge-aware loss must propagate max_fn_weight_override to the
    inner FN loss; otherwise the warmup schedule would be silently dropped."""
    torch.manual_seed(0)
    z = torch.randn(8, 16)
    p_fn = torch.rand(4, 4) * 0.4
    new = EdgeAwareFNWeightedInfoNCELoss(
        temperature=0.1, max_fn_weight=0.5,
        lambda_edge=0.0, lambda_edge_align=0.0,
    )
    out_warmup = new(z, p_fn, None, max_fn_weight_override=0.0)
    infonce = InfoNCELoss(temperature=0.1, normalize_embeddings=True)(z)["loss"]
    assert torch.allclose(out_warmup["fn_loss"], infonce, atol=1e-5)
    assert out_warmup["max_fn_weight_current"].item() == 0.0


# ---------------------------------------------------------------------------
# Schedule helper
# ---------------------------------------------------------------------------


def test_schedule_disabled_returns_none():
    assert compute_max_fn_weight_current(1, None, 0.5) is None
    assert compute_max_fn_weight_current(1, {"enabled": False}, 0.5) is None


def test_schedule_warmup_ramp_end():
    cfg = {
        "enabled": True,
        "warmup_epochs": 5, "ramp_epochs": 10,
        "max_fn_weight_start": 0.0, "max_fn_weight_end": 0.5,
    }
    # Warmup
    assert compute_max_fn_weight_current(1, cfg, 0.5) == 0.0
    assert compute_max_fn_weight_current(5, cfg, 0.5) == 0.0
    # Mid-ramp (epoch 10 = 5 epochs into the ramp -> half).
    mid = compute_max_fn_weight_current(11, cfg, 0.5)
    assert abs(mid - 0.25) < 1e-6, mid
    # Past ramp
    assert compute_max_fn_weight_current(100, cfg, 0.5) == 0.5


def test_schedule_zero_ramp_is_step():
    cfg = {
        "enabled": True,
        "warmup_epochs": 3, "ramp_epochs": 0,
        "max_fn_weight_start": 0.0, "max_fn_weight_end": 0.5,
    }
    assert compute_max_fn_weight_current(3, cfg, 0.5) == 0.0
    assert compute_max_fn_weight_current(4, cfg, 0.5) == 0.5


# ---------------------------------------------------------------------------
# Loss-internal regularization gradient must flow to scorer parameters when
# pfn_reg_total is added to the optimizer objective.
# ---------------------------------------------------------------------------


def test_pfn_reg_total_grad_to_scorer():
    """Adding pfn_reg_total to the objective produces grads on scorer params."""
    torch.manual_seed(0)
    scorer = MLPPairScorer(input_dim=8, detach_inputs=True, init_pfn_prior=0.4)
    fn = FNWeightedInfoNCELoss(
        max_fn_weight=0.5,
        pfn_regularization={
            "enabled": True, "target_mean": 0.05,
            "lambda_mean": 1.0, "lambda_cap": 1.0,
        },
    )
    z = _two_view_batch(seed=21).requires_grad_(False)
    p_fn = scorer(z[:4])
    out = fn(z, p_fn, max_fn_weight_override=0.5)
    (out["pfn_reg_total"]).backward()
    grad_norm = sum(
        float(p.grad.abs().sum()) for p in scorer.parameters() if p.grad is not None
    )
    assert grad_norm > 0, "pfn_reg_total carried no gradient to the scorer"
