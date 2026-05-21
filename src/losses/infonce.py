"""SimCLR InfoNCE loss with diagonal exclusion (R3, R7, R9)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .masks import build_positive_mask


class InfoNCELoss(nn.Module):
    """SimCLR-style InfoNCE on ``z`` of shape ``[2B, D]``.

    The input is the concatenation of two augmented views, so the positive
    of index ``i`` is ``i + B``. The diagonal is excluded by setting
    similarities to ``-inf`` before softmax.

    Returns a dict (R7) with the scalar loss and named diagnostics for
    logging — never a raw scalar.
    """

    def __init__(self, temperature: float = 0.1, normalize_embeddings: bool = True):
        super().__init__()
        if temperature <= 0:
            raise ValueError(
                f"InfoNCE temperature must be > 0, got {temperature}. "
                "Pick a value in (0, 1] (SimCLR default: 0.1)."
            )
        self.temperature = float(temperature)
        self.normalize_embeddings = bool(normalize_embeddings)

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        if z.dim() != 2:
            raise ValueError(f"InfoNCE expects z of shape [2B, D], got {tuple(z.shape)}")
        n = z.shape[0]
        if n % 2 != 0 or n < 4:
            raise ValueError(
                f"InfoNCE expects an even batch dim 2B with B>=2, got 2B={n}"
            )
        batch = n // 2

        if self.normalize_embeddings:
            z = F.normalize(z, dim=-1, eps=1e-12)

        # Raw cosine sims (diagnostics)
        sim_raw = z @ z.t()

        # Scaled logits with diagonal excluded
        logits = sim_raw / self.temperature
        diag_mask = torch.eye(n, dtype=torch.bool, device=z.device)
        logits = logits.masked_fill(diag_mask, float("-inf"))

        # Positive targets: i -> i + B, i + B -> i
        targets = torch.cat(
            [torch.arange(batch, n, device=z.device), torch.arange(0, batch, device=z.device)]
        )
        loss = F.cross_entropy(logits, targets)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"InfoNCE produced non-finite loss={loss.item()}. "
                f"temperature={self.temperature}, batch={batch}, "
                f"sim_raw min={sim_raw.min().item():.4f} max={sim_raw.max().item():.4f}"
            )

        pos_mask = build_positive_mask(batch).to(z.device)
        neg_mask = ~pos_mask & ~diag_mask

        return {
            "loss": loss,
            "pos_sim_mean": sim_raw[pos_mask].mean().detach(),
            "neg_sim_mean": sim_raw[neg_mask].mean().detach(),
            "temperature": torch.tensor(self.temperature, device=z.device),
        }
