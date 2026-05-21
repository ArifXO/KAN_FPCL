"""Parameter-count utilities (R1, R8)."""

from __future__ import annotations

import torch.nn as nn


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    """Return total number of parameters in ``module``."""
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def format_param_report(named_modules: dict[str, nn.Module]) -> str:
    """Render a multi-line parameter report for ``param_count.txt`` (R8).

    Example:
        encoder: 11,168,832
        head:    156,800
        total:   11,325,632
    """
    lines = []
    total = 0
    for name, mod in named_modules.items():
        n = count_parameters(mod)
        total += n
        lines.append(f"{name}: {n:,}")
    lines.append(f"total: {total:,}")
    return "\n".join(lines)
