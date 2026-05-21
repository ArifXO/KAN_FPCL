"""Embedding geometry metrics for normalized representations (Stage 8)."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def _check_2d_float(name: str, z: Tensor, min_rows: int = 1) -> Tensor:
    if not isinstance(z, Tensor):
        raise ValueError(f"{name} must be a torch.Tensor, got {type(z).__name__}.")
    if z.ndim != 2:
        raise ValueError(f"{name} must have shape [N, D], got {tuple(z.shape)}.")
    if z.shape[0] < min_rows:
        raise ValueError(
            f"{name} must have at least {min_rows} rows, got {z.shape[0]}."
        )
    if z.shape[1] <= 0:
        raise ValueError(f"{name} must have a positive feature dimension.")
    if not z.dtype.is_floating_point:
        raise ValueError(f"{name} must be floating point, got {z.dtype}.")
    if not torch.isfinite(z).all():
        raise ValueError(f"{name} contains NaN or Inf values.")
    return z


def alignment(z_a: Tensor, z_b: Tensor) -> Tensor:
    """Mean squared distance between positive-pair embeddings."""
    z_a = _check_2d_float("z_a", z_a)
    z_b = _check_2d_float("z_b", z_b)
    if z_a.shape != z_b.shape:
        raise ValueError(
            f"z_a and z_b must have the same shape, got "
            f"{tuple(z_a.shape)} vs {tuple(z_b.shape)}."
        )
    return ((z_a - z_b) ** 2).sum(dim=-1).mean()


def uniformity(z: Tensor, t: float = 2.0) -> Tensor:
    """Wang-Isola uniformity loss using ``torch.pdist`` for pair distances."""
    z = _check_2d_float("z", z, min_rows=2)
    if isinstance(t, bool) or not isinstance(t, (float, int)):
        raise ValueError(f"t must be a positive finite number, got {t!r}.")
    t = float(t)
    if not math.isfinite(t) or t <= 0:
        raise ValueError(f"t must be a positive finite number, got {t!r}.")

    pairwise = torch.pdist(z, p=2)
    mean_exp = pairwise.pow(2).mul(-t).exp().mean()
    return mean_exp.clamp_min(torch.finfo(z.dtype).tiny).log()


def effective_rank(z: Tensor, eps: float = 1e-12) -> Tensor:
    """Effective rank from singular-value entropy."""
    z = _check_2d_float("z", z)
    if isinstance(eps, bool) or not isinstance(eps, (float, int)):
        raise ValueError(f"eps must be a positive finite number, got {eps!r}.")
    eps = float(eps)
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError(f"eps must be a positive finite number, got {eps!r}.")

    singular_values = torch.linalg.svdvals(z)
    total = singular_values.sum()
    if float(total.detach()) <= eps:
        raise ValueError("effective_rank is undefined for all-zero embeddings.")
    probs = singular_values / total.clamp_min(eps)
    probs = probs[probs > eps]
    entropy = -(probs * probs.log()).sum()
    return entropy.exp()


def per_dim_std(z: Tensor) -> Tensor:
    """Per-feature standard deviation of embeddings."""
    z = _check_2d_float("z", z, min_rows=2)
    return z.std(dim=0)


def off_diagonal_covariance_norm(z: Tensor) -> Tensor:
    """Frobenius norm of off-diagonal covariance entries divided by D."""
    z = _check_2d_float("z", z, min_rows=2)
    centered = z - z.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / (z.shape[0] - 1)
    off_diag = cov - torch.diag(torch.diagonal(cov))
    return off_diag.pow(2).sum().sqrt() / z.shape[1]


__all__ = [
    "alignment",
    "uniformity",
    "effective_rank",
    "per_dim_std",
    "off_diagonal_covariance_norm",
]
