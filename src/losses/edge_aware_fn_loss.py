"""Edge-aware FN-weighted InfoNCE (Stage 7.5 — H4 mitigation).

Strict generalization of :class:`FNWeightedInfoNCELoss`:

    L = fn_loss
        + λ_edge       · edge_contrastive_loss
        + λ_edge_align · edge_align_loss

* ``fn_loss`` — :class:`FNWeightedInfoNCELoss(z, p_fn)` (composed, not
  duplicated). Identical numerics to Stage 6/7.
* ``edge_contrastive_loss`` — SimCLR-style InfoNCE on edge fingerprints
  with separate temperature ``tau_edge``. Diagonal masked.
* ``edge_align_loss`` — ``mean_i(1 - cos(F_i^view1, F_i^view2))``. Encourages
  the same image's two augmented views to land on consistent KAN edges, so
  the edge channel is invariant to augmentation noise.

Edge-feature shape convention: ``edge_features[2B, 256]`` — concatenated
view1+view2 fingerprints (mirrors ``z``'s ``[2B, D]`` layout). When all
lambdas are zero, behavior is numerically identical to Stage 7's loss
(R3 backward-compat test).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fn_weighted_infonce import FNWeightedInfoNCELoss
from .masks import build_positive_mask


class EdgeAwareFNWeightedInfoNCELoss(nn.Module):
    """FN-weighted InfoNCE plus optional edge-contrastive and edge-align terms.

    Args:
        temperature: τ for the FN-weighted InfoNCE on ``z`` (matches Stage 6/7).
        normalize_embeddings: L2-normalize ``z`` (matches Stage 6/7).
        max_fn_weight: Upper clip on ``p_fn`` inside the FN-weighted loss.
        lambda_edge: Weight on the edge-contrastive auxiliary (0 ⇒ disabled).
        tau_edge: Temperature for the edge-contrastive InfoNCE.
        lambda_edge_align: Weight on the edge-alignment auxiliary (0 ⇒ disabled).

    Forward inputs:
        z: ``[2B, D]`` — concatenated views (same as Stage 6/7).
        p_fn: ``[B, B]`` — per-pair FN beliefs from a pair scorer.
        edge_features: ``[2B, 256]`` or ``None`` — concatenated fingerprints
            for both views. ``None`` is allowed only when both lambdas are 0
            *and* the scorer producing ``p_fn`` did not consume edges itself
            (in which case the auxiliary terms simply report 0).

    Returns (R7) a 16-key dict.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        normalize_embeddings: bool = True,
        max_fn_weight: float = 1.0,
        lambda_edge: float = 0.0,
        tau_edge: float = 0.1,
        lambda_edge_align: float = 0.0,
        pfn_regularization: Optional[dict] = None,
    ):
        super().__init__()
        if tau_edge <= 0:
            raise ValueError(f"tau_edge must be > 0, got {tau_edge}.")
        if lambda_edge < 0:
            raise ValueError(f"lambda_edge must be >= 0, got {lambda_edge}.")
        if lambda_edge_align < 0:
            raise ValueError(f"lambda_edge_align must be >= 0, got {lambda_edge_align}.")

        self.fn_loss = FNWeightedInfoNCELoss(
            temperature=temperature,
            normalize_embeddings=normalize_embeddings,
            max_fn_weight=max_fn_weight,
            pfn_regularization=pfn_regularization,
        )
        self.lambda_edge = float(lambda_edge)
        self.tau_edge = float(tau_edge)
        self.lambda_edge_align = float(lambda_edge_align)

    # ---- helpers ----------------------------------------------------------

    def _edge_contrastive(self, fingerprint: torch.Tensor) -> dict[str, torch.Tensor]:
        """SimCLR InfoNCE on ``[2B, 256]`` L2-normalized fingerprints."""
        n = fingerprint.shape[0]
        batch = n // 2
        f = F.normalize(fingerprint, dim=-1, eps=1e-12)
        sim = f @ f.t()
        logits = sim / self.tau_edge
        diag = torch.eye(n, dtype=torch.bool, device=f.device)
        logits = logits.masked_fill(diag, float("-inf"))
        targets = torch.cat(
            [torch.arange(batch, n, device=f.device),
             torch.arange(0, batch, device=f.device)]
        )
        loss = F.cross_entropy(logits, targets)

        pos_mask = build_positive_mask(batch).to(f.device)
        neg_mask = ~pos_mask & ~diag
        return {
            "loss": loss,
            "pos_sim_mean": sim[pos_mask].mean().detach(),
            "neg_sim_mean": sim[neg_mask].mean().detach(),
        }

    def _edge_align(self, fingerprint: torch.Tensor) -> torch.Tensor:
        """Mean (1 - cos) between view1 and view2 fingerprints of the same image."""
        n = fingerprint.shape[0]
        batch = n // 2
        f = F.normalize(fingerprint, dim=-1, eps=1e-12)
        cos = (f[:batch] * f[batch:]).sum(dim=-1)
        return (1.0 - cos).mean()

    # ---- forward ----------------------------------------------------------

    def forward(
        self,
        z: torch.Tensor,
        p_fn: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
        max_fn_weight_override: Optional[float] = None,
    ) -> dict[str, torch.Tensor]:
        needs_edges = self.lambda_edge > 0 or self.lambda_edge_align > 0
        if needs_edges and edge_features is None:
            raise ValueError(
                "EdgeAwareFNWeightedInfoNCELoss requires edge_features when "
                f"lambda_edge={self.lambda_edge} or lambda_edge_align="
                f"{self.lambda_edge_align} is positive."
            )
        if edge_features is not None:
            if edge_features.dim() != 2 or edge_features.shape[0] != z.shape[0]:
                raise ValueError(
                    f"edge_features must be shape [2B={z.shape[0]}, D_F], "
                    f"got {tuple(edge_features.shape)}."
                )
            if torch.isnan(edge_features).any() or not torch.isfinite(edge_features).all():
                raise ValueError("edge_features contains NaN or inf.")

        fn_out = self.fn_loss(z, p_fn, max_fn_weight_override=max_fn_weight_override)
        fn_scalar = fn_out["loss"]
        device = fn_scalar.device

        if edge_features is not None:
            edge_c = self._edge_contrastive(edge_features)
            edge_contrastive_loss = edge_c["loss"]
            edge_pos_sim_mean = edge_c["pos_sim_mean"]
            edge_neg_sim_mean = edge_c["neg_sim_mean"]
            edge_align_loss = self._edge_align(edge_features)
        else:
            zero = torch.zeros((), device=device)
            edge_contrastive_loss = zero
            edge_align_loss = zero
            edge_pos_sim_mean = zero.detach()
            edge_neg_sim_mean = zero.detach()

        total = (
            fn_scalar
            + self.lambda_edge * edge_contrastive_loss
            + self.lambda_edge_align * edge_align_loss
        )
        if not torch.isfinite(total):
            raise RuntimeError(
                f"EdgeAwareFNWeightedInfoNCELoss produced non-finite total={total.item()}. "
                f"fn={fn_scalar.item():.4f}, edge_c={float(edge_contrastive_loss):.4f}, "
                f"edge_a={float(edge_align_loss):.4f}, "
                f"lambda_edge={self.lambda_edge}, lambda_edge_align={self.lambda_edge_align}."
            )

        # Pass through every fn_out diagnostic (including the new raw/clipped
        # p_fn stats + pfn_reg_total). pfn_reg_total carries live grad — the
        # train script adds it to the optimizer objective outside `loss`.
        result = {
            "loss": total,
            "fn_loss": fn_scalar.detach(),
            "edge_contrastive_loss": edge_contrastive_loss.detach(),
            "edge_align_loss": edge_align_loss.detach(),
            "lambda_edge": torch.tensor(self.lambda_edge, device=device),
            "lambda_edge_align": torch.tensor(self.lambda_edge_align, device=device),
            "edge_pos_sim_mean": edge_pos_sim_mean,
            "edge_neg_sim_mean": edge_neg_sim_mean,
            "tau_edge": torch.tensor(self.tau_edge, device=device),
        }
        # Skip the inner-loss scalar `loss` (we already added `fn_loss`).
        for k, v in fn_out.items():
            if k == "loss":
                continue
            result[k] = v
        return result
