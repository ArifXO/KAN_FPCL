"""FastKAN projection layers for contrastive heads."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


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


def _base_activation(name: str) -> nn.Module:
    if not isinstance(name, str):
        raise ValueError(f"base_activation must be a string, got {type(name).__name__}.")
    activations = {
        "silu": nn.SiLU(),
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "identity": nn.Identity(),
        "none": nn.Identity(),
    }
    try:
        return activations[name.lower()]
    except KeyError as exc:
        choices = ", ".join(sorted(activations))
        raise ValueError(
            f"base_activation must be one of [{choices}], got {name!r}."
        ) from exc


class FastKANLayer(nn.Module):
    """RBF FastKAN layer with optional base linear path.

    The edge tensor returned by ``return_edges=True`` is computed before the
    input-edge sum, so it has shape ``[batch, output_dim, input_dim]``.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_centers: int = 8,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        use_base_linear: bool = True,
        base_activation: str = "silu",
        rbf_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = _positive_int("input_dim", input_dim)
        self.output_dim = _positive_int("output_dim", output_dim)
        self.num_centers = _positive_int("num_centers", num_centers)
        grid_min = _finite_float("grid_min", grid_min)
        grid_max = _finite_float("grid_max", grid_max)
        rbf_init_scale = _finite_float("rbf_init_scale", rbf_init_scale)
        if grid_min >= grid_max:
            raise ValueError(
                f"grid_min must be less than grid_max, got {grid_min} >= {grid_max}."
            )
        if rbf_init_scale <= 0:
            raise ValueError(
                f"rbf_init_scale must be positive, got {rbf_init_scale}."
            )

        centers = torch.linspace(grid_min, grid_max, self.num_centers)
        self.register_buffer("centers", centers)
        grid_width = grid_max - grid_min
        initial_bw = grid_width / max(self.num_centers - 1, 1)
        self.bandwidth = nn.Parameter(torch.tensor(initial_bw, dtype=torch.float32))

        self.rbf_weight = nn.Parameter(
            torch.empty(self.output_dim, self.input_dim, self.num_centers)
        )
        nn.init.normal_(self.rbf_weight, mean=0.0, std=rbf_init_scale)

        self.use_base_linear = bool(use_base_linear)
        self.base = (
            nn.Linear(self.input_dim, self.output_dim)
            if self.use_base_linear
            else None
        )
        self.base_activation = _base_activation(base_activation)

    def forward(
        self, x: torch.Tensor, return_edges: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply the FastKAN layer to a ``[batch, input_dim]`` tensor."""
        if x.ndim != 2 or x.shape[-1] != self.input_dim:
            raise ValueError(
                "FastKANLayer expected input shape "
                f"[batch, {self.input_dim}], got {tuple(x.shape)}."
            )

        bandwidth = self.bandwidth.clamp_min(1e-6)
        scaled = (x.unsqueeze(-1) - self.centers) / bandwidth
        rbf = torch.exp(-(scaled.square()))
        phi = torch.einsum("bic,oic->boi", rbf, self.rbf_weight)
        output = phi.sum(dim=-1)
        if self.base is not None:
            output = output + self.base(self.base_activation(x))
        if return_edges:
            return output, phi
        return output

    def parameter_count(self) -> int:
        """Return the number of trainable parameters in this layer."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FastKANProjector(nn.Module):
    """Two-layer FastKAN projection head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_centers: int = 8,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        use_base_linear: bool = True,
        base_activation: str = "silu",
        rbf_init_scale: float = 0.1,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = _positive_int("input_dim", input_dim)
        self.hidden_dim = _positive_int("hidden_dim", hidden_dim)
        self.output_dim = _positive_int("output_dim", output_dim)
        self.hidden_layer = FastKANLayer(
            self.input_dim,
            self.hidden_dim,
            num_centers,
            grid_min,
            grid_max,
            use_base_linear,
            base_activation,
            rbf_init_scale,
        )
        self.output_layer = FastKANLayer(
            self.hidden_dim,
            self.output_dim,
            num_centers,
            grid_min,
            grid_max,
            use_base_linear,
            base_activation,
            rbf_init_scale,
        )
        self.normalize = bool(normalize)

    @property
    def num_edges(self) -> int:
        """Return the number of input-output edges in the last KAN layer."""
        return self.output_layer.output_dim * self.output_layer.input_dim

    def forward(
        self, h: torch.Tensor, return_edges: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Project encoder features and optionally return last-layer edges."""
        hidden = self.hidden_layer(h)
        output = self.output_layer(hidden, return_edges=return_edges)
        if return_edges:
            z, phi = output
        else:
            z = output
            phi = None
        if self.normalize:
            norm = z.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            z = z / norm
        if return_edges:
            return z, phi
        return z

    def parameter_count(self) -> int:
        """Return the number of trainable parameters in this projector."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
