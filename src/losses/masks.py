"""Mask builders for contrastive losses (R3)."""

from __future__ import annotations

import torch


def build_positive_mask(batch_size: int) -> torch.Tensor:
    """Build the SimCLR positive-pair mask of shape ``[2B, 2B]``.

    Given a batch of ``B`` images with two augmented views concatenated as
    ``[v1_0, ..., v1_{B-1}, v2_0, ..., v2_{B-1}]``, the only positive for
    sample ``i`` (``i < B``) is index ``i + B`` (and vice versa). The
    diagonal (self-similarity) is **not** marked positive.

    Args:
        batch_size: Number of original images ``B`` (so the mask is 2B x 2B).

    Returns:
        bool tensor of shape ``[2*batch_size, 2*batch_size]`` with True at
        ``(i, i+B)`` and ``(i+B, i)``, False everywhere else (including
        the diagonal).

    Raises:
        ValueError: If ``batch_size <= 0``.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    n = 2 * batch_size
    mask = torch.zeros(n, n, dtype=torch.bool)
    idx = torch.arange(batch_size)
    mask[idx, idx + batch_size] = True
    mask[idx + batch_size, idx] = True
    return mask
