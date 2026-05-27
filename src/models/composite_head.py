"""Composite projection heads: projector + optional residual warp."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .kan import FastKANProjector


class ProjectorWithWarp(nn.Module):
    """Chain a projector (512->D) with an optional residual warp (D->D).

    When ``warp`` is None, this is equivalent to the projector alone.
    When ``warp`` is provided, output = ``warp(projector(h))`` and the output
    width is the warp's (== projector output dim).

    ``return_edges=True`` yields the last-layer KAN edge tensor. The warp is the
    last transformation, so when a warp is present its edges are returned; with
    no warp, edges come from the projector only if it is a ``FastKANProjector``
    (an ``MLPHead`` projector has no edges and yields ``phi=None``).
    """

    def __init__(self, projector: nn.Module, warp: Optional[nn.Module] = None):
        super().__init__()
        self.projector = projector
        self.warp = warp

    def forward(self, h: torch.Tensor, return_edges: bool = False):
        if not return_edges:
            z = self.projector(h)
            if self.warp is not None:
                z = self.warp(z)
            return z

        if isinstance(self.projector, FastKANProjector):
            z, phi = self.projector(h, return_edges=True)
        else:
            z, phi = self.projector(h), None

        if self.warp is not None:
            z, phi = self.warp(z, return_edges=True)

        return z, phi

    @property
    def alpha(self):
        """Expose the warp's scalar alpha for logging, or None."""
        if self.warp is not None and hasattr(self.warp, "alpha"):
            return self.warp.alpha
        return None

    def parameter_count(self) -> int:
        """Return the number of trainable parameters in this head."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
