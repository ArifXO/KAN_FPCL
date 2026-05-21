"""Residual FastKAN warp for normalized embedding refinement."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fastkan import FastKANProjector


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return value


def _finite_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be a finite number, got {value!r}.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return value


def _bool(name: str, value: bool) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool, got {type(value).__name__}.")
    return value


class ResidualFastKANWarp(nn.Module):
    """Apply ``normalize(z + alpha * KAN(LayerNorm(z)))``.

    The residual branch is same-dimensional, so the module can refine an
    existing embedding while preserving its output width.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_centers: int = 8,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        use_base_linear: bool = True,
        base_activation: str = "silu",
        rbf_init_scale: float = 0.1,
        alpha_init: float = 0.0,
        learnable_alpha: bool = True,
        clamp_alpha: bool = True,
        clamp_max: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = _positive_int("input_dim", input_dim)
        self.hidden_dim = _positive_int("hidden_dim", hidden_dim)
        alpha_init = _finite_float("alpha_init", alpha_init)
        clamp_max = _finite_float("clamp_max", clamp_max)
        self.learnable_alpha = _bool("learnable_alpha", learnable_alpha)
        self.clamp_alpha = _bool("clamp_alpha", clamp_alpha)
        if clamp_max <= 0:
            raise ValueError(f"clamp_max must be positive, got {clamp_max}.")

        self.clamp_max = clamp_max
        self.norm = nn.LayerNorm(self.input_dim)
        self.kan = FastKANProjector(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.input_dim,
            num_centers=num_centers,
            grid_min=grid_min,
            grid_max=grid_max,
            use_base_linear=use_base_linear,
            base_activation=base_activation,
            rbf_init_scale=rbf_init_scale,
            normalize=False,
        )

        alpha_tensor = torch.tensor(alpha_init, dtype=torch.float32)
        if self.learnable_alpha:
            self.alpha_raw = nn.Parameter(alpha_tensor)
        else:
            self.register_buffer("alpha_raw", alpha_tensor)

    @property
    def alpha(self) -> torch.Tensor:
        """Return the effective scalar alpha, clamped for logging if enabled."""
        if self.clamp_alpha:
            return self.alpha_raw.clamp(0.0, self.clamp_max)
        return self.alpha_raw

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Warp a ``[batch, input_dim]`` embedding and L2-normalize the result."""
        if z.ndim != 2 or z.shape[-1] != self.input_dim:
            raise ValueError(
                "ResidualFastKANWarp expected input shape "
                f"[batch, {self.input_dim}], got {tuple(z.shape)}."
            )
        if not self.learnable_alpha and float(self.alpha.detach()) == 0.0:
            return F.normalize(z, dim=-1, eps=1e-12)

        delta = self.kan(self.norm(z))
        warped = z + self.alpha * delta
        return F.normalize(warped, dim=-1, eps=1e-12)

    def parameter_count(self) -> int:
        """Return the number of trainable parameters in this warp."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
