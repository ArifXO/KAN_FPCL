"""2-layer MLP projection head for contrastive learning (Stage 2)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPHead(nn.Module):
    """Two-layer MLP projector: in_dim -> hidden_dim -> output_dim.

    Args:
        in_dim: Input feature dimension (typically encoder output, e.g. 512).
        hidden_dim: Hidden-layer width.
        output_dim: Projected embedding dimension.
        use_batch_norm: Apply BatchNorm1d after the hidden linear.
        l2_normalize: L2-normalize the final embedding (stability clamp 1e-12).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        output_dim: int,
        use_batch_norm: bool = True,
        l2_normalize: bool = True,
    ):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim)]
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.l2_normalize = l2_normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        if self.l2_normalize:
            norm = z.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            z = z / norm
        return z

    def parameter_count(self) -> int:
        """Return total number of trainable parameters in this head."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
