"""Debiased Contrastive Learning loss (Chuang et al. 2020) — R3, R7, R9.

Vanilla InfoNCE treats every non-positive sample as a true negative, but
multi-label CXR data routinely contains hidden positives among negatives.
The debiased estimator approximates the *true* negative expectation:

    E[neg] = (1 / (1 - τ⁺)) · ( E_uniform[neg] - τ⁺ · E[pos] )

with ``τ⁺`` the assumed positive class prior. ``debiased_neg`` is clamped to
``estimator_clip_min`` to prevent the bias correction from going negative
(which produces an undefined log).

Label-free: this loss does NOT take labels and is appropriate as a
self-supervised baseline alongside InfoNCE. It must NOT be routed through
the FN scorer / KAN scorer / edge-aware paths (those are studied separately).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .masks import build_positive_mask


class DebiasedContrastiveLoss(nn.Module):
    """Debiased Contrastive Learning over ``z`` of shape ``[2B, D]``.

    Args:
        temperature: Softmax temperature τ > 0 (default 0.5, paper).
        tau_plus: Assumed positive class prior in ``[0, 1)`` (default 0.1).
        normalize_embeddings: L2-normalize ``z`` along the embedding axis.
        estimator_clip_min: Lower bound on ``debiased_neg`` (default 1e-8).

    Forward inputs:
        z: ``[2B, D]`` — concatenated views ``[v1; v2]``.

    Returns (R7) a dict with named scalars; ``loss`` is the only key with
    grad — everything else is detached for logging.
    """

    def __init__(
        self,
        temperature: float = 0.5,
        tau_plus: float = 0.1,
        normalize_embeddings: bool = True,
        estimator_clip_min: float = 1e-8,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(
                f"Debiased CL temperature must be > 0, got {temperature}."
            )
        if not (0.0 <= tau_plus < 1.0):
            raise ValueError(
                f"tau_plus must be in [0, 1), got {tau_plus}. The estimator "
                "divides by (1 - tau_plus) and is undefined at tau_plus=1."
            )
        if estimator_clip_min <= 0:
            raise ValueError(
                f"estimator_clip_min must be > 0, got {estimator_clip_min}."
            )
        self.temperature = float(temperature)
        self.tau_plus = float(tau_plus)
        self.normalize_embeddings = bool(normalize_embeddings)
        self.estimator_clip_min = float(estimator_clip_min)

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        if z.dim() != 2:
            raise ValueError(
                f"Debiased CL expects z of shape [2B, D], got {tuple(z.shape)}."
            )
        n = z.shape[0]
        if n % 2 != 0 or n < 4:
            raise ValueError(
                f"Debiased CL expects an even batch dim 2B with B>=2, got 2B={n}."
            )
        if not torch.isfinite(z).all():
            raise ValueError(
                "z contains non-finite values; check the projector output."
            )
        batch = n // 2

        if self.normalize_embeddings:
            z = F.normalize(z, dim=-1, eps=1e-12)

        sim_raw = z @ z.t()  # [2B, 2B] cosine sims (already L2 if normalized).
        diag_mask = torch.eye(n, dtype=torch.bool, device=z.device)
        pos_mask = build_positive_mask(batch).to(z.device)
        neg_mask = ~pos_mask & ~diag_mask

        logits = sim_raw / self.temperature
        # Exclude self-comparison from both pos and neg pools.
        exp_all = torch.exp(logits.masked_fill(diag_mask, float("-inf")))

        # Positive of i is i+B (and i+B -> i). Single positive per anchor.
        targets = torch.cat(
            [
                torch.arange(batch, n, device=z.device),
                torch.arange(0, batch, device=z.device),
            ]
        )
        pos_exp = exp_all.gather(1, targets.unsqueeze(1)).squeeze(1)  # [2B]

        # Sum of exp over negatives only (diagonal and positive excluded).
        neg_exp_sum = (exp_all * neg_mask.float()).sum(dim=-1)  # [2B]
        n_neg = float(neg_mask.sum(dim=-1)[0].item())  # constant per anchor.

        # Debiased negative estimator. ``pos_exp`` is the empirical positive
        # expectation we subtract from the uniform-sample negative pool.
        debiased_neg = (neg_exp_sum - self.tau_plus * n_neg * pos_exp) / (
            1.0 - self.tau_plus
        )
        debiased_neg = debiased_neg.clamp_min(self.estimator_clip_min)

        loss = -(torch.log(pos_exp) - torch.log(pos_exp + debiased_neg)).mean()

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Debiased CL produced non-finite loss={loss.item()}. "
                f"temperature={self.temperature}, tau_plus={self.tau_plus}, "
                f"pos_exp min={pos_exp.min().item():.4e}, "
                f"debiased_neg min={debiased_neg.min().item():.4e}."
            )

        return {
            "loss": loss,
            "pos_sim_mean": sim_raw[pos_mask].mean().detach(),
            "neg_sim_mean": sim_raw[neg_mask].mean().detach(),
            "pos_exp_mean": pos_exp.mean().detach(),
            "neg_exp_mean": neg_exp_sum.mean().detach(),
            "debiased_neg_mean": debiased_neg.mean().detach(),
            "tau_plus": torch.tensor(self.tau_plus, device=z.device),
            "temperature": torch.tensor(self.temperature, device=z.device),
        }
