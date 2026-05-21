"""Tests for InfoNCE loss (R3, R7, R9)."""

import pytest
import torch

from src.losses import InfoNCELoss, build_positive_mask


# ---------------------------------------------------------------------------
# build_positive_mask — basic structure
# ---------------------------------------------------------------------------


def test_positive_mask_shape_and_values():
    mask = build_positive_mask(4)
    assert mask.shape == (8, 8)
    assert mask.dtype == torch.bool
    # (i, i+B) and (i+B, i) only
    for i in range(4):
        assert mask[i, i + 4].item() is True
        assert mask[i + 4, i].item() is True
    # diagonal is NOT positive
    assert not mask.diagonal().any().item()
    # exactly 2B positives total
    assert mask.sum().item() == 8


def test_positive_mask_invalid_batch():
    with pytest.raises(ValueError, match="batch_size must be positive"):
        build_positive_mask(0)


# ---------------------------------------------------------------------------
# InfoNCELoss — contract & numerical behavior
# ---------------------------------------------------------------------------


def _two_view_batch(batch: int = 4, dim: int = 8, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v1 = torch.randn(batch, dim, generator=g)
    v2 = v1 + 0.1 * torch.randn(batch, dim, generator=g)  # close to v1 -> easy positives
    return torch.cat([v1, v2], dim=0).requires_grad_(True)


def test_returns_required_dict_keys():
    loss_fn = InfoNCELoss(temperature=0.1)
    z = _two_view_batch()
    out = loss_fn(z)
    for key in ("loss", "pos_sim_mean", "neg_sim_mean", "temperature"):
        assert key in out, f"missing key {key}"


def test_loss_is_finite_scalar():
    loss_fn = InfoNCELoss(temperature=0.1)
    out = loss_fn(_two_view_batch())
    loss = out["loss"]
    assert loss.dim() == 0
    assert torch.isfinite(loss).item()
    assert loss.item() > 0


def test_temperature_validation():
    with pytest.raises(ValueError, match="temperature must be > 0"):
        InfoNCELoss(temperature=0.0)
    with pytest.raises(ValueError, match="temperature must be > 0"):
        InfoNCELoss(temperature=-0.5)


def test_invalid_batch_shape_raises():
    loss_fn = InfoNCELoss()
    with pytest.raises(ValueError, match=r"\[2B, D\]"):
        loss_fn(torch.randn(8))  # 1D
    with pytest.raises(ValueError, match="even batch dim"):
        loss_fn(torch.randn(3, 8))  # odd batch


def test_diagonal_excluded_from_softmax():
    """Diagonal must contribute 0 probability — self-similarity is masked out.

    Trick: build z where each sample is identical to itself but very different
    from every other sample. If the diagonal is NOT excluded, the softmax
    target (the positive view) is dwarfed by self-similarity and loss explodes.
    With diagonal excluded, the positive can win.
    """
    loss_fn = InfoNCELoss(temperature=0.1, normalize_embeddings=True)
    batch = 4
    # Identical pairs -> positives are perfect cos=1.
    base = torch.randn(batch, 8)
    z = torch.cat([base, base.clone()], dim=0)
    out = loss_fn(z)
    # With identical positives and properly masked diagonal,
    # the loss should be very small (positives perfectly chosen).
    assert out["loss"].item() < 0.1, (
        f"Loss too high ({out['loss'].item():.4f}) — diagonal likely not excluded"
    )


def test_loss_decreases_with_gradient_step():
    """A single Adam step on z should reduce InfoNCE loss."""
    z = _two_view_batch(batch=8, dim=16, seed=1)
    loss_fn = InfoNCELoss(temperature=0.5)
    optim = torch.optim.Adam([z], lr=0.1)

    out0 = loss_fn(z)
    loss0 = out0["loss"].item()
    optim.zero_grad()
    out0["loss"].backward()
    optim.step()

    with torch.no_grad():
        loss1 = loss_fn(z)["loss"].item()
    assert loss1 < loss0, f"loss did not decrease: {loss0:.4f} -> {loss1:.4f}"


def test_gradient_flows_to_input():
    z = _two_view_batch()
    out = InfoNCELoss(temperature=0.2)(z)
    out["loss"].backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert z.grad.abs().sum() > 0, "no gradient reached the input"


def test_r3_positive_only_loss_near_zero():
    """R3: when all positive pairs are perfect, loss → 0 (within log-softmax floor)."""
    base = torch.randn(8, 16)
    z = torch.cat([base, base.clone()], dim=0)  # v1 == v2
    out = InfoNCELoss(temperature=0.05)(z)
    assert out["loss"].item() < 0.05


def test_r3_negative_only_loss_finite_and_large():
    """R3: when positives are dissimilar (orthogonal to their pair), loss is finite and > log(2B-1)/2.

    Construct z so that each v1_i is orthogonal to v2_i but parallel to every
    other v2_j — i.e., the "positive" target is the WORST possible match.
    """
    n = 4
    # v1 = e_i (standard basis), v2 = e_{(i+1) % n} so positive aligns to NOT-i.
    v1 = torch.eye(n, n)
    v2 = torch.roll(v1, shifts=1, dims=0)
    z = torch.cat([v1, v2], dim=0)
    out = InfoNCELoss(temperature=0.1)(z)
    assert torch.isfinite(out["loss"]).item()
    # Loss must be substantially > 0 in this adversarial setup.
    assert out["loss"].item() > 1.0


def test_r3_fn_case_label_flip_increases_loss():
    """R3 (FN case): for vanilla InfoNCE, breaking the positive pairing must
    strictly increase the loss compared to the unbroken case.

    We simulate "marking a true positive as negative" by swapping a positive
    pair so it no longer aligns with its target index — the model now treats
    a real positive as if it were a negative.
    """
    base = torch.randn(4, 8)
    z_good = torch.cat([base, base + 0.05 * torch.randn(4, 8)], dim=0)
    # Permute the second-view block so positive-pair alignment is broken
    perm = torch.tensor([1, 0, 3, 2])
    z_bad = torch.cat([base, (base + 0.05 * torch.randn(4, 8))[perm]], dim=0)
    loss_fn = InfoNCELoss(temperature=0.1)
    good = loss_fn(z_good)["loss"].item()
    bad = loss_fn(z_bad)["loss"].item()
    assert bad > good, f"FN-style perturbation did not increase loss: {good:.4f} -> {bad:.4f}"


def test_pos_sim_greater_than_neg_sim_when_views_close():
    """Diagnostic sanity: with close v1/v2, pos sim should beat neg sim."""
    out = InfoNCELoss(temperature=0.1)(_two_view_batch(batch=8, dim=16))
    assert out["pos_sim_mean"].item() > out["neg_sim_mean"].item()
