"""Edge-aware FN pair scorer (Stage 7.5 — H3/H4).

Generalizes the Stage 6 MLPPairScorer. Pair features are handcrafted
geometric statistics: ``[cos(z_i,z_k), L2(z_i,z_k), |z_i-z_k|, z_i*z_k]``,
with optional edge-fingerprint terms ``[e_ik, δ_ik]`` when
``use_edge_features=True`` (H4 hypothesis). Head is MLP or FastKAN.
KAN-specific knobs (kan_hidden_dim, kan_num_centers, kan_use_base_linear)
tune for ±15 % R1 parity vs the MLP anchor.

CONTRACT: no labels enter this module. The label-aware oracle variant is
reserved for Stage 10 ablations.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kan import FastKANLayer

_ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}


def _build_mlp(in_dim: int, hidden_dim: int, num_layers: int, act: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(num_layers):
        layers.append(nn.Linear(d, hidden_dim))
        layers.append(_ACTIVATIONS[act]())
        d = hidden_dim
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)


class EdgeAwarePairScorer(nn.Module):
    """Pair scorer with handcrafted geometric features + optional KAN-edge channel.

    Args:
        input_dim: Embedding dim ``D`` of the projector output (z).
        edge_dim: Width of the edge fingerprint (must be 256 — matches
            :func:`src.losses.edge_features.edge_fingerprint`). Stored for
            shape validation only; not used when ``use_edge_features=False``.
        hidden_dim: Hidden width of the scorer head.
        num_layers: Number of hidden layers (MLP) / single FastKAN layer
            replaces them when ``scorer_type='kan'``.
        activation: Activation name for the MLP head (``relu|gelu|silu``).
        use_edge_features: When True, two extra scalar pair features
            (``e_ik``, ``δ_ik``) are appended and ``forward`` requires
            ``edge_features`` to be supplied.
        scorer_type: ``'mlp'`` (Linear+activation stack) or ``'kan'``
            (single FastKANLayer with ``base_linear=True``).
        kan_num_centers: KAN RBF center count when ``scorer_type='kan'``.
            Defaults to 4 to keep the param count near the MLP anchor.

    Forward:
        z_view1: ``[B, D]`` — single-view embeddings.
        edge_features: ``[B, edge_dim]`` view-1 fingerprints, required iff
            ``use_edge_features=True``.
        Returns ``p_fn[B, B] ∈ [0, 1]``.
    """

    def __init__(
        self,
        input_dim: int,
        edge_dim: int = 256,
        hidden_dim: int = 16,
        num_layers: int = 2,
        activation: str = "relu",
        use_edge_features: bool = False,
        scorer_type: str = "mlp",
        kan_num_centers: int = 4,
        kan_hidden_dim: Optional[int] = None,
        kan_use_base_linear: bool = True,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}.")
        if edge_dim <= 0:
            raise ValueError(f"edge_dim must be positive, got {edge_dim}.")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}.")
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {sorted(_ACTIVATIONS)}, got {activation!r}."
            )
        if scorer_type not in ("mlp", "kan"):
            raise ValueError(
                f"scorer_type must be 'mlp' or 'kan', got {scorer_type!r}."
            )

        self.input_dim = int(input_dim)
        self.edge_dim = int(edge_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.use_edge_features = bool(use_edge_features)
        self.scorer_type = scorer_type

        # Feature layout: [cos, L2, |z_i-z_k|(D), z_i*z_k(D)] (+ [e_ik, δ_ik] if edge).
        pair_feat_dim = 2 + 2 * self.input_dim + (2 if self.use_edge_features else 0)
        self.pair_feat_dim = pair_feat_dim

        if scorer_type == "mlp":
            self._head_mlp: Optional[nn.Module] = _build_mlp(
                pair_feat_dim, hidden_dim, num_layers, activation
            )
            self._head_kan: Optional[FastKANLayer] = None
            self._head_kan_out: Optional[nn.Linear] = None
        else:
            # Single FastKAN layer pair_feat_dim -> kan_hidden_dim, then linear to 1.
            # Knobs (kan_hidden_dim, kan_num_centers, kan_use_base_linear) exist
            # because pair_feat_dim is large (~260), so the KAN layer's
            # rbf_weight tensor (h*pair_feat_dim*centers) and optional base
            # Linear (pair_feat_dim*h+h) easily blow past the MLP anchor's
            # parameter count. Tune them for ±15 % parity (R1).
            # FastKANLayer requires 2-D input, so forward() flattens
            # [B,B,F] -> [B*B,F] before calling it.
            kan_h = int(kan_hidden_dim) if kan_hidden_dim is not None else hidden_dim
            self._head_mlp = None
            self._head_kan = FastKANLayer(
                input_dim=pair_feat_dim,
                output_dim=kan_h,
                num_centers=kan_num_centers,
                use_base_linear=kan_use_base_linear,
                base_activation="silu",
            )
            self._head_kan_out = nn.Linear(kan_h, 1)
            self._kan_h = kan_h

    def _pair_features(
        self, z: torch.Tensor, edge_features: Optional[torch.Tensor]
    ) -> torch.Tensor:
        b, d = z.shape
        z_norm = F.normalize(z, dim=-1, eps=1e-12)
        cos = (z_norm @ z_norm.t()).unsqueeze(-1)            # [B,B,1]
        z_i = z.unsqueeze(1).expand(b, b, d)                 # [B,B,D]
        z_j = z.unsqueeze(0).expand(b, b, d)                 # [B,B,D]
        diff_abs = (z_i - z_j).abs()                          # [B,B,D]
        prod = z_i * z_j                                      # [B,B,D]
        l2 = (z_i - z_j).pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()  # [B,B,1]
        feats = [cos, l2, diff_abs, prod]

        if self.use_edge_features:
            f_norm = F.normalize(edge_features, dim=-1, eps=1e-12)
            e = (f_norm @ f_norm.t()).unsqueeze(-1)           # [B,B,1]
            delta = e - cos
            feats.extend([e, delta])

        return torch.cat(feats, dim=-1)                       # [B,B, pair_feat_dim]

    def forward(
        self,
        z_view1: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if z_view1.dim() != 2:
            raise ValueError(
                f"EdgeAwarePairScorer expects z of shape [B, D], got {tuple(z_view1.shape)}."
            )
        b, d = z_view1.shape
        if d != self.input_dim:
            raise ValueError(
                f"EdgeAwarePairScorer was built for input_dim={self.input_dim} but got D={d}."
            )

        if self.use_edge_features:
            if edge_features is None:
                raise ValueError(
                    "EdgeAwarePairScorer(use_edge_features=True) requires "
                    "edge_features=[B, edge_dim] at forward time, got None."
                )
            if edge_features.dim() != 2 or edge_features.shape != (b, self.edge_dim):
                raise ValueError(
                    f"edge_features must have shape [B={b}, edge_dim={self.edge_dim}], "
                    f"got {tuple(edge_features.shape)}."
                )

        feats = self._pair_features(z_view1, edge_features)   # [B,B, pair_feat_dim]
        if self.scorer_type == "mlp":
            logits = self._head_mlp(feats).squeeze(-1)         # [B,B]
        else:
            flat = feats.reshape(b * b, self.pair_feat_dim)
            hidden = self._head_kan(flat)                       # [B*B, kan_h]
            logits = self._head_kan_out(hidden).reshape(b, b)   # [B,B]
        return torch.sigmoid(logits)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
