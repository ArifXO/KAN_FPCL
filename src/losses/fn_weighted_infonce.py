"""False-negative weighted InfoNCE loss (Stage 6 — H2).

The vanilla InfoNCE in :mod:`src.losses.infonce` treats every off-diagonal,
non-positive pair as a true negative. In multi-label CXR data this is wrong:
two crops from different patients may share findings (e.g. cardiomegaly), so
pushing their embeddings apart corrupts the manifold.

This loss accepts a per-pair false-negative probability ``p_fn`` from a pair
scorer (e.g. :class:`src.models.pair_scorer.MLPPairScorer`) and downweights
likely-FN pairs inside the InfoNCE denominator:

    L_i = -log( exp(s_i,pos / τ)  /  ( exp(s_i,pos / τ)
                                       + Σ_{j∈neg} (1 - p_fn_ij) · exp(s_i,j / τ) ) )

Implemented numerically via ``logits + log(weight)`` inside ``logsumexp`` (R9).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .masks import build_positive_mask


class FNWeightedInfoNCELoss(nn.Module):
    """SimCLR-style InfoNCE with per-pair false-negative downweighting.

    Args:
        temperature: Softmax temperature τ > 0 (SimCLR default 0.1).
        normalize_embeddings: L2-normalize ``z`` along the embedding axis.
        max_fn_weight: Upper clip applied to ``p_fn`` before computing the
            denominator weight. ``1.0`` (default) allows full removal of a
            negative; ``< 1.0`` caps the maximum downweighting (a floor of
            ``1 - max_fn_weight`` on negative weights) so suspected FNs
            still contribute some signal.

    Forward inputs:
        z: ``[2B, D]`` — concatenated views ``[v1; v2]`` (positive of ``i`` is
            ``i + B``).
        p_fn: ``[B, B]`` — false-negative probabilities among view1 samples,
            entries in ``[0, 1]``. Tiled to ``[2B, 2B]`` via ``repeat(2, 2)``
            so the same FN belief applies to all four (vi, vj) view pairings.

    Returns (R7):
        dict with keys ``loss``, ``pos_sim_mean``, ``neg_sim_mean``,
        ``p_fn_mean``, ``p_fn_max``, ``downweighted_fraction``, ``temperature``.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        normalize_embeddings: bool = True,
        max_fn_weight: float = 1.0,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError(
                f"FN-weighted InfoNCE temperature must be > 0, got {temperature}."
            )
        if not (0.0 <= max_fn_weight <= 1.0):
            raise ValueError(
                f"max_fn_weight must be in [0, 1], got {max_fn_weight}."
            )
        self.temperature = float(temperature)
        self.normalize_embeddings = bool(normalize_embeddings)
        self.max_fn_weight = float(max_fn_weight)

    def forward(
        self, z: torch.Tensor, p_fn: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if z.dim() != 2:
            raise ValueError(
                f"FN-weighted InfoNCE expects z of shape [2B, D], got {tuple(z.shape)}"
            )
        n = z.shape[0]
        if n % 2 != 0 or n < 4:
            raise ValueError(
                f"FN-weighted InfoNCE expects 2B with B>=2, got 2B={n}"
            )
        batch = n // 2

        if p_fn.dim() != 2 or p_fn.shape != (batch, batch):
            raise ValueError(
                f"p_fn must be shape [B, B] with B={batch}, got {tuple(p_fn.shape)}"
            )
        if torch.isnan(p_fn).any():
            raise ValueError(
                "p_fn contains NaN values; check pair-scorer outputs."
            )
        if (p_fn < 0).any() or (p_fn > 1).any():
            raise ValueError(
                f"p_fn must lie in [0, 1], got min={p_fn.min().item():.4f}, "
                f"max={p_fn.max().item():.4f}. Ensure the scorer applies sigmoid."
            )

        if self.normalize_embeddings:
            z = F.normalize(z, dim=-1, eps=1e-12)

        sim_raw = z @ z.t()
        logits = sim_raw / self.temperature

        diag_mask = torch.eye(n, dtype=torch.bool, device=z.device)
        pos_mask = build_positive_mask(batch).to(z.device)
        neg_mask = ~pos_mask & ~diag_mask

        # Tile p_fn to [2B, 2B] so the same FN belief applies to (v1,v1),
        # (v1,v2), (v2,v1), (v2,v2) views of the same pair of underlying images.
        p_fn_clamped = p_fn.clamp(min=0.0, max=self.max_fn_weight)
        p_fn_full = p_fn_clamped.repeat(2, 2)

        # NUMERICAL: clamp (1 - p_fn) before .log() to avoid -inf when p_fn→1.
        # torch.where evaluates BOTH branches eagerly (PyTorch semantic), so
        # the negative branch must itself be finite even at positive slots.
        weight = (1.0 - p_fn_full).clamp_min(1e-10)
        weight = torch.where(pos_mask, torch.ones_like(weight), weight)
        log_weight = weight.log()
        log_weight = log_weight.masked_fill(diag_mask, float("-inf"))

        log_denom = torch.logsumexp(logits + log_weight, dim=-1)
        targets = torch.cat(
            [torch.arange(batch, n, device=z.device),
             torch.arange(0, batch, device=z.device)]
        )
        log_pos = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = (log_denom - log_pos).mean()

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"FN-weighted InfoNCE produced non-finite loss={loss.item()}. "
                f"temperature={self.temperature}, batch={batch}, "
                f"max_fn_weight={self.max_fn_weight}, "
                f"p_fn min={p_fn.min().item():.4f} max={p_fn.max().item():.4f}, "
                f"sim_raw min={sim_raw.min().item():.4f} max={sim_raw.max().item():.4f}"
            )

        neg_p = p_fn_full[neg_mask]
        return {
            "loss": loss,
            "pos_sim_mean": sim_raw[pos_mask].mean().detach(),
            "neg_sim_mean": sim_raw[neg_mask].mean().detach(),
            "p_fn_mean": neg_p.mean().detach(),
            "p_fn_max": neg_p.max().detach(),
            "downweighted_fraction": (neg_p > 0.5).float().mean().detach(),
            "temperature": torch.tensor(self.temperature, device=z.device),
        }
