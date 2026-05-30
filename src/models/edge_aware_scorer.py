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

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kan import FastKANLayer

_ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}


def _logit(p: float) -> float:
    if not (0.0 < p < 1.0):
        raise ValueError(f"init_pfn_prior must be in (0, 1), got {p}.")
    return math.log(p / (1.0 - p))


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
    """Pair scorer with handcrafted geometric features + optional edge channel.

    Args: input_dim (D of z); edge_dim (256, matches edge_fingerprint);
    hidden_dim, num_layers, activation (MLP head); use_edge_features (adds
    [e_ik, δ_ik] features and requires edge_features at forward); scorer_type
    ('mlp'|'kan'); kan_num_centers, kan_hidden_dim, kan_use_base_linear (KAN
    parity knobs); init_pfn_prior (final-bias init); detach_inputs (severs
    H4 path if True — default False, see __init__ comment).

    Forward(z_view1[B,D], edge_features[B,edge_dim]) -> p_fn[B,B] in [0,1].
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
        init_pfn_prior: float | None = 0.05,
        detach_inputs: bool = False,
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
        # DEVIATION (CLAUDE.md §5): default False, not True per the
        # saturation spec — True would sever the H4 scorer->edge->KAN path
        # validated by test_gradient_flows_through_edge_to_kan_weights.
        self.detach_inputs = bool(detach_inputs)

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
            # KAN knobs tune for R1 parity vs the MLP anchor; forward()
            # flattens [B,B,F] -> [B*B,F] because FastKANLayer is 2-D only.
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

        if init_pfn_prior is not None:
            with torch.no_grad():
                if scorer_type == "mlp":
                    final_linear = self._head_mlp[-1]
                else:
                    final_linear = self._head_kan_out
                final_linear.bias.fill_(_logit(float(init_pfn_prior)))

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

        if self.detach_inputs:
            z_view1 = z_view1.detach()
            if edge_features is not None:
                edge_features = edge_features.detach()

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
