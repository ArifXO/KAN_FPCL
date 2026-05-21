"""Edge fingerprints from FastKAN edge tensors (Stage 7.5 — H4).

The FastKAN edge tensor ``phi`` of shape ``[B, O, I]`` (output_dim x input_dim
fan-out per sample) carries spline-activated edge magnitudes that capture
sample-specific basis usage. We compress it to a fixed-length fingerprint
``F[B, 256]`` so the pair scorer can consult two extra disagreement signals
between samples (see ``EdgeAwarePairScorer``) without paying the full
``O*I`` memory cost (R-pitfall #4: edge fingerprint compression).

Two pure tensor functions are exported. No nn.Module state, no learnable
parameters here — projection is a deterministic average-pool 1d, so the
fingerprint is reproducible from ``phi`` alone (R8 reproducibility).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_FINGERPRINT_DIM = 256


def edge_fingerprint(phi: torch.Tensor) -> torch.Tensor:
    """Compress KAN edges ``phi[B, O, I]`` to L2-normalized ``F[B, 256]``.

    Steps:
      1. Flatten last two dims -> ``[B, O*I]``.
      2. Adaptive-average-pool to ``min(O*I, 256)`` channels (a deterministic
         fixed linear projection — no parameters, R8-friendly).
      3. Right-pad with zeros to width 256 when ``O*I < 256``.
      4. L2-normalize with eps for the zero-edge case.

    Args:
        phi: ``[B, O, I]`` edge tensor (returned by ``FastKANLayer(..., return_edges=True)``).

    Returns:
        ``F[B, 256]`` fingerprint with ``||F_i||_2 ≤ 1``.

    Raises:
        ValueError: if ``phi.ndim != 3``.
    """
    if phi.dim() != 3:
        raise ValueError(
            f"edge_fingerprint expects phi of shape [B, O, I], got {tuple(phi.shape)}."
        )
    b, o, i = phi.shape
    flat = phi.reshape(b, o * i)

    target = min(o * i, _FINGERPRINT_DIM)
    # adaptive_avg_pool1d treats the middle dim as channels; we want length-axis
    # pooling so swap [B, L] -> [B, 1, L] -> pool -> [B, 1, target] -> squeeze.
    pooled = F.adaptive_avg_pool1d(flat.unsqueeze(1), output_size=target).squeeze(1)

    if target < _FINGERPRINT_DIM:
        pad = pooled.new_zeros(b, _FINGERPRINT_DIM - target)
        pooled = torch.cat([pooled, pad], dim=-1)

    # F.normalize handles the all-zero case via its eps and yields zero vectors,
    # so cosine sims with zero fingerprints are 0 rather than NaN.
    return F.normalize(pooled, dim=-1, eps=1e-12)


def edge_pair_similarity(fingerprint: torch.Tensor) -> torch.Tensor:
    """Cosine-similarity matrix of edge fingerprints.

    Args:
        fingerprint: ``[B, 256]`` already L2-normalized (output of
            :func:`edge_fingerprint`).

    Returns:
        ``[B, B]`` cosine sim matrix. Symmetric; diagonal = 1.0 for non-zero
        rows and 0.0 for zero rows.

    Raises:
        ValueError: if input is not 2-D.
    """
    if fingerprint.dim() != 2:
        raise ValueError(
            f"edge_pair_similarity expects [B, D], got {tuple(fingerprint.shape)}."
        )
    # Re-normalize defensively — caller-supplied fingerprints may have drifted.
    f = F.normalize(fingerprint, dim=-1, eps=1e-12)
    return f @ f.t()
