"""Tests for EdgeAwareFNWeightedInfoNCELoss (Stage 7.5 — R3 / H4).

The most important property under test is **backward compatibility**:
when both lambdas are 0 and ``use_edge_features=False``, the new loss is
numerically identical to Stage 7's :class:`FNWeightedInfoNCELoss` on the
same inputs. Drift here would silently corrupt every downstream H2/H3
comparison.
"""

from __future__ import annotations

import pytest
import torch

from src.losses import (
    EdgeAwareFNWeightedInfoNCELoss,
    FNWeightedInfoNCELoss,
)
from src.losses.edge_features import edge_fingerprint
from src.models import EdgeAwarePairScorer


_REQUIRED_KEYS = {
    # Stage 7.5 schema (loss + edge components + back-compat p_fn summaries).
    "loss",
    "fn_loss",
    "edge_contrastive_loss",
    "edge_align_loss",
    "lambda_edge",
    "lambda_edge_align",
    "p_fn_mean",
    "p_fn_max",
    "p_fn_mean_raw",
    "p_fn_max_raw",
    "p_fn_at_cap_fraction",
    "downweighted_fraction",
    "edge_pos_sim_mean",
    "edge_neg_sim_mean",
    "temperature",
    "tau_edge",
    # Pitfall #5 raw/clipped saturation diagnostics + scorer-similarity stats.
    "pos_sim_mean",
    "neg_sim_mean",
    "p_fn_raw_mean",
    "p_fn_raw_std",
    "p_fn_raw_min",
    "p_fn_raw_max",
    "p_fn_clipped_mean",
    "p_fn_clipped_std",
    "p_fn_clipped_min",
    "p_fn_clipped_max",
    "p_fn_near_zero_fraction",
    "p_fn_entropy_mean",
    "effective_neg_weight_mean",
    "effective_neg_weight_min",
    "max_fn_weight_current",
    # Saturation regularization components.
    "loss_pfn_mean",
    "loss_pfn_cap",
    "loss_pfn_entropy",
    "pfn_reg_total",
    "target_pfn_mean",
}


def _synthetic_inputs(b: int = 4, d: int = 8, seed: int = 0):
    torch.manual_seed(seed)
    z = torch.randn(2 * b, d)
    p_fn = torch.rand(b, b) * 0.3       # well-bounded
    edge = torch.randn(2 * b, 256)
    return z, p_fn, edge


# ---------------------------------------------------------------------------
# (h) DICT KEYS (R7)
# ---------------------------------------------------------------------------


def test_dict_has_all_required_keys():
    z, p_fn, edge = _synthetic_inputs()
    loss_fn = EdgeAwareFNWeightedInfoNCELoss(
        lambda_edge=0.0, lambda_edge_align=0.0
    )
    out = loss_fn(z, p_fn, edge)
    assert set(out.keys()) == _REQUIRED_KEYS, (
        f"Missing keys: {_REQUIRED_KEYS - set(out)}; "
        f"unexpected: {set(out) - _REQUIRED_KEYS}"
    )


# ---------------------------------------------------------------------------
# (a) BACKWARD COMPAT: lambdas=0 + no edges => identical to Stage 7
# ---------------------------------------------------------------------------


def test_lambdas_zero_no_edges_matches_stage7():
    z, p_fn, _ = _synthetic_inputs()
    stage7 = FNWeightedInfoNCELoss(temperature=0.1, max_fn_weight=0.5)
    new = EdgeAwareFNWeightedInfoNCELoss(
        temperature=0.1, max_fn_weight=0.5,
        lambda_edge=0.0, lambda_edge_align=0.0,
    )
    out_old = stage7(z, p_fn)
    out_new = new(z, p_fn, None)
    assert torch.allclose(out_old["loss"], out_new["loss"], atol=1e-5), (
        f"Stage7 loss {out_old['loss'].item():.6f} != new {out_new['loss'].item():.6f}"
    )
    assert torch.allclose(out_old["loss"], out_new["fn_loss"], atol=1e-5)


# ---------------------------------------------------------------------------
# (b) lambdas=0 WITH edges supplied => same numerical result
# ---------------------------------------------------------------------------


def test_lambdas_zero_with_edges_matches_stage7():
    z, p_fn, edge = _synthetic_inputs()
    stage7 = FNWeightedInfoNCELoss(temperature=0.1, max_fn_weight=0.5)
    new = EdgeAwareFNWeightedInfoNCELoss(
        temperature=0.1, max_fn_weight=0.5,
        lambda_edge=0.0, lambda_edge_align=0.0,
    )
    out_old = stage7(z, p_fn)
    out_new = new(z, p_fn, edge)
    assert torch.allclose(out_old["loss"], out_new["loss"], atol=1e-5)


# ---------------------------------------------------------------------------
# (c) (d) Missing edge_features when required => ValueError
# ---------------------------------------------------------------------------


def test_missing_edge_features_with_lambda_edge_raises():
    z, p_fn, _ = _synthetic_inputs()
    loss_fn = EdgeAwareFNWeightedInfoNCELoss(lambda_edge=0.5)
    with pytest.raises(ValueError, match="requires edge_features"):
        loss_fn(z, p_fn, None)


def test_missing_edge_features_with_lambda_edge_align_raises():
    z, p_fn, _ = _synthetic_inputs()
    loss_fn = EdgeAwareFNWeightedInfoNCELoss(lambda_edge_align=0.5)
    with pytest.raises(ValueError, match="requires edge_features"):
        loss_fn(z, p_fn, None)


def test_scorer_use_edge_features_true_without_edges_raises():
    scorer = EdgeAwarePairScorer(input_dim=8, use_edge_features=True)
    with pytest.raises(ValueError, match="requires edge_features"):
        scorer(torch.randn(4, 8), None)


# ---------------------------------------------------------------------------
# (e) STABILITY: zero-vector edge features -> finite loss
# ---------------------------------------------------------------------------


def test_zero_edge_features_finite():
    z, p_fn, _ = _synthetic_inputs()
    edge = torch.zeros(2 * 4, 256)
    loss_fn = EdgeAwareFNWeightedInfoNCELoss(
        lambda_edge=0.5, lambda_edge_align=0.5
    )
    out = loss_fn(z, p_fn, edge)
    assert torch.isfinite(out["loss"])
    assert torch.isfinite(out["edge_contrastive_loss"])
    assert torch.isfinite(out["edge_align_loss"])


def test_edge_fingerprint_of_zero_phi_no_nan_in_loss():
    z, p_fn, _ = _synthetic_inputs()
    phi = torch.zeros(2 * 4, 8, 32)
    edge = edge_fingerprint(phi)
    loss_fn = EdgeAwareFNWeightedInfoNCELoss(lambda_edge=0.1, lambda_edge_align=0.1)
    out = loss_fn(z, p_fn, edge)
    assert torch.isfinite(out["loss"])


# ---------------------------------------------------------------------------
# (f) MONOTONICITY: higher edge pos sim => lower edge contrastive loss
# ---------------------------------------------------------------------------


def test_edge_contrastive_lower_when_views_more_aligned():
    """When view2 fingerprints equal view1, contrastive loss must be lower
    than when they are random / orthogonal."""
    b, d = 4, 256
    torch.manual_seed(0)
    f_v1 = torch.nn.functional.normalize(torch.randn(b, d), dim=-1)

    f_aligned = torch.cat([f_v1, f_v1.clone()], dim=0)            # perfect pos
    f_random = torch.cat(
        [f_v1, torch.nn.functional.normalize(torch.randn(b, d), dim=-1)], dim=0
    )

    z = torch.randn(2 * b, 8)
    p_fn = torch.zeros(b, b)
    loss_fn = EdgeAwareFNWeightedInfoNCELoss(lambda_edge=0.0, lambda_edge_align=0.0)

    aligned = loss_fn(z, p_fn, f_aligned)["edge_contrastive_loss"].item()
    random_ = loss_fn(z, p_fn, f_random)["edge_contrastive_loss"].item()
    assert aligned < random_, f"aligned={aligned} not < random={random_}"


# ---------------------------------------------------------------------------
# (g) GRADIENT FLOW: gradients reach z, scorer params, and KAN W via edges
# ---------------------------------------------------------------------------


def test_gradient_flows_to_z_scorer_and_kan_via_edges():
    from src.models.kan import FastKANProjector

    b = 4
    torch.manual_seed(0)
    encoder_feats = torch.randn(2 * b, 32, requires_grad=True)
    projector = FastKANProjector(
        input_dim=32, hidden_dim=16, output_dim=8,
        num_centers=4, normalize=True,
    )
    scorer = EdgeAwarePairScorer(input_dim=8, use_edge_features=True, scorer_type="mlp")
    loss_fn = EdgeAwareFNWeightedInfoNCELoss(
        lambda_edge=0.5, lambda_edge_align=0.5
    )

    z, phi = projector(encoder_feats, return_edges=True)
    edge_feats = edge_fingerprint(phi)
    p_fn = scorer(z[:b], edge_feats[:b])

    out = loss_fn(z, p_fn, edge_feats)
    out["loss"].backward()

    # z gradient (via encoder_feats)
    assert encoder_feats.grad is not None
    assert torch.isfinite(encoder_feats.grad).all()
    assert encoder_feats.grad.abs().sum() > 0

    # Scorer params get grads
    for name, p in scorer.named_parameters():
        assert p.grad is not None, f"no grad for scorer {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad in scorer {name}"

    # The KAN edge path (output_layer.rbf_weight) must receive a gradient via
    # `edge_features` — that is the H4 wiring promise.
    kan_w = projector.output_layer.rbf_weight
    assert kan_w.grad is not None
    assert torch.isfinite(kan_w.grad).all()
    assert kan_w.grad.abs().sum() > 0


def test_gradient_flows_through_edge_to_kan_weights():
    """Train-script-level guard for H4 (mirrors scripts/train/train_edge.py).

    With both aux lambdas at 0 and ``z`` detached into *both* the scorer and
    the loss, the ONLY live path from the loss back to the KAN projector
    weights is ``edge_features -> scorer -> p_fn -> fn_loss``. (``p_fn`` enters
    ``FNWeightedInfoNCELoss`` via ``log(1 - p_fn)`` inside the logsumexp, so it
    is differentiable even when ``z`` is constant.) This isolates the pathway
    that ``train_edge.py`` would sever if it detached ``edge_features`` before
    the scorer — the H4 bug. We assert the gradient is present when the edge
    arg is live and absent when it is detached, so re-introducing the detach
    makes this test fail.
    """
    from src.models.kan import FastKANProjector

    b = 4
    torch.manual_seed(0)
    x = torch.randn(2 * b, 32)

    def kan_grad_norm(detach_edge: bool) -> float:
        projector = FastKANProjector(
            input_dim=32, hidden_dim=16, output_dim=8, num_centers=4, normalize=True
        )
        scorer = EdgeAwarePairScorer(input_dim=8, use_edge_features=True, scorer_type="mlp")
        loss_fn = EdgeAwareFNWeightedInfoNCELoss(lambda_edge=0.0, lambda_edge_align=0.0)

        z, phi = projector(x, return_edges=True)
        edge_feats = edge_fingerprint(phi)
        edge_arg = edge_feats[:b].detach() if detach_edge else edge_feats[:b]
        p_fn = scorer(z[:b].detach(), edge_arg)
        # z detached + edges=None in the loss => the scorer's edge path is the
        # sole route to the projector weights.
        out = loss_fn(z.detach(), p_fn, None)
        out["loss"].backward()
        g = projector.output_layer.rbf_weight.grad
        return float(g.abs().sum()) if g is not None else 0.0

    alive = kan_grad_norm(detach_edge=False)
    severed = kan_grad_norm(detach_edge=True)
    assert alive > 0, (
        "No KAN projector gradient through the scorer -> edge path. H4's "
        "scorer -> edge -> KAN gradient pathway is broken (edge_features likely "
        "detached before the scorer in train_edge.py)."
    )
    assert severed == 0, (
        f"KAN received gradient (={severed}) even with edge_features detached; "
        "the test no longer isolates the scorer -> edge -> KAN path."
    )


# ---------------------------------------------------------------------------
# (i) PARAM PARITY (R1): KAN scorer within ±15 % of MLP scorer
# ---------------------------------------------------------------------------


def test_param_parity_kan_vs_mlp_scorer():
    """R1: KAN scorer must land within ±15 % of the MLP anchor at the
    config the experiment configs actually use (see edge_aware_kan.yaml).
    The KAN-specific knobs (kan_hidden_dim, kan_num_centers) compensate
    for the large pair_feat_dim — see EdgeAwarePairScorer docstring."""
    common = dict(
        input_dim=128, edge_dim=256, hidden_dim=16, num_layers=2,
        use_edge_features=True,
    )
    mlp = EdgeAwarePairScorer(**common, scorer_type="mlp").parameter_count()
    kan = EdgeAwarePairScorer(
        **common,
        scorer_type="kan",
        kan_hidden_dim=4,
        kan_num_centers=3,
        kan_use_base_linear=True,
    ).parameter_count()
    ratio = kan / mlp
    assert 0.85 <= ratio <= 1.15, (
        f"R1 parity violated: kan={kan}, mlp={mlp}, ratio={ratio:.3f}"
    )


# ---------------------------------------------------------------------------
# Misc: invalid edge_features shape, NaN handling
# ---------------------------------------------------------------------------


def test_invalid_edge_features_shape_raises():
    z, p_fn, _ = _synthetic_inputs()
    bad = torch.randn(3, 256)             # 3 != 2B=8
    loss_fn = EdgeAwareFNWeightedInfoNCELoss(lambda_edge=0.1)
    with pytest.raises(ValueError, match=r"edge_features must be shape"):
        loss_fn(z, p_fn, bad)


def test_nan_edge_features_raises():
    z, p_fn, edge = _synthetic_inputs()
    edge[0, 0] = float("nan")
    loss_fn = EdgeAwareFNWeightedInfoNCELoss(lambda_edge=0.1)
    with pytest.raises(ValueError, match="NaN"):
        loss_fn(z, p_fn, edge)


def test_negative_lambda_rejected():
    with pytest.raises(ValueError, match="lambda_edge must be >= 0"):
        EdgeAwareFNWeightedInfoNCELoss(lambda_edge=-0.1)
    with pytest.raises(ValueError, match="lambda_edge_align must be >= 0"):
        EdgeAwareFNWeightedInfoNCELoss(lambda_edge_align=-0.1)
