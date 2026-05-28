"""MLP and KAN pair scorers for false-negative detection (Stages 6/7 — H2/H3).

Given a single view of projected embeddings ``z[B, D]``, each scorer outputs
``p_fn[B, B] ∈ [0, 1]`` — the per-pair probability that two samples in the
batch are false negatives (i.e. share a latent label even though they are
different images). The score is fed to
:class:`src.losses.fn_weighted_infonce.FNWeightedInfoNCELoss`.

Both scorers operate on pair features ``[z_i ‖ z_j]`` for all ``B²`` pairs;
output is squashed with ``sigmoid`` to honor the loss's ``p_fn ∈ [0, 1]``
contract (R9 guard in the loss will reject violations).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .kan import FastKANLayer


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
        logits = 0.5 * (logits + logits.T)  # enforce symmetry: p_fn[i,j] == p_fn[j,i]
        return torch.sigmoid(logits)

    def parameter_count(self) -> int:
        """Return total number of trainable parameters in this scorer."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class KANPairScorer(nn.Module):
    """FastKAN-based pair scorer for H3 comparison against MLPPairScorer.

    R1 parameter parity (input_dim=128 defaults):
        MLPPairScorer(128, hidden_dim=32, num_layers=2):
            Linear(256->32): 8224  +  Linear(32->32): 1056  +  Linear(32->1): 33
            = 9,313 params  (MLP anchor)
        KANPairScorer(128, hidden_dim=5, num_centers=6, use_base_linear=True):
            FastKANLayer(256->5, c=6): rbf[5*256*6]=7680 + bw=1 + base_linear[256*5+5]=1285
            = 8,966  +  Linear(5->1): 6  =  8,972 params
            Parity: |8972 - 9313| / 9313 = 3.7%  << 15%  (R1 satisfied)
        NOTE: num_centers=8 would give 11,532 params (24% over) — violates R1.
              Default is num_centers=6 to satisfy parity.

    Args:
        input_dim: Embedding dimension D of the projector output.
        hidden_dim: FastKANLayer output dim. Default 5.
        num_centers: RBF centers per edge. Default 6 (satisfies R1 parity;
            num_centers=8 would violate ±15% budget).
        use_base_linear: FastKAN base linear path. Default True.

    Forward:
        z_view1: [B, D] — single-view embeddings.
        Returns p_fn[B, B] in [0, 1].
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 5,
        num_centers: int = 6,
        use_base_linear: bool = True,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}.")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        if num_centers <= 0:
            raise ValueError(f"num_centers must be positive, got {num_centers}.")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)

        self.kan_layer = FastKANLayer(
            input_dim=2 * self.input_dim,
            output_dim=hidden_dim,
            num_centers=num_centers,
            use_base_linear=use_base_linear,
        )
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, z_view1: torch.Tensor) -> torch.Tensor:
        if z_view1.dim() != 2:
            raise ValueError(
                f"KANPairScorer expects z of shape [B, D], got {tuple(z_view1.shape)}."
            )
        b, d = z_view1.shape
        if d != self.input_dim:
            raise ValueError(
                f"KANPairScorer was built for input_dim={self.input_dim} but got D={d}."
            )

        z_i = z_view1.unsqueeze(1).expand(b, b, d)
        z_j = z_view1.unsqueeze(0).expand(b, b, d)
        pair_feat = torch.cat([z_i, z_j], dim=-1).reshape(b * b, 2 * d)
        hidden = self.kan_layer(pair_feat)
        logits = self.out(hidden).reshape(b, b)
        logits = 0.5 * (logits + logits.T)  # enforce symmetry: p_fn[i,j] == p_fn[j,i]
        return torch.sigmoid(logits)

    def parameter_count(self) -> int:
        """Return total number of trainable parameters in this scorer."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
