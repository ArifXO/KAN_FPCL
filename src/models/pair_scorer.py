"""MLP pair scorer for false-negative detection (Stage 6 — H2).

Given a single view of projected embeddings ``z[B, D]``, the scorer outputs
``p_fn[B, B] ∈ [0, 1]`` — the per-pair probability that two samples in the
batch are false negatives (i.e. share a latent label even though they are
different images). The score is fed to
:class:`src.losses.fn_weighted_infonce.FNWeightedInfoNCELoss`.

The MLP operates on pair features ``[z_i ‖ z_j]`` for all ``B²`` pairs;
output is squashed with ``sigmoid`` to honor the loss's ``p_fn ∈ [0, 1]``
contract (R9 guard in the loss will reject violations).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPPairScorer(nn.Module):
    """Two-or-more-layer MLP producing pairwise false-negative probabilities.

    Args:
        input_dim: Embedding dimension ``D`` of the projector output.
        hidden_dim: Width of each hidden layer.
        num_layers: Number of hidden ``Linear + ReLU`` blocks before the
            final scalar projection (default 2 — keeps the parameter
            footprint small for R1 parity vs the KAN scorer in Stage 7).

    Forward:
        z_view1: ``[B, D]`` — single-view embeddings.
        Returns ``p_fn[B, B]`` with values in ``[0, 1]``.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 2):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}.")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}.")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)

        layers: list[nn.Module] = []
        in_dim = 2 * input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, z_view1: torch.Tensor) -> torch.Tensor:
        if z_view1.dim() != 2:
            raise ValueError(
                f"MLPPairScorer expects z of shape [B, D], got {tuple(z_view1.shape)}."
            )
        b, d = z_view1.shape
        if d != self.input_dim:
            raise ValueError(
                f"MLPPairScorer was built for input_dim={self.input_dim} but got D={d}."
            )

        z_i = z_view1.unsqueeze(1).expand(b, b, d)
        z_j = z_view1.unsqueeze(0).expand(b, b, d)
        pair_feat = torch.cat([z_i, z_j], dim=-1)
        logits = self.mlp(pair_feat).squeeze(-1)
        return torch.sigmoid(logits)

    def parameter_count(self) -> int:
        """Return total number of trainable parameters in this scorer."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
