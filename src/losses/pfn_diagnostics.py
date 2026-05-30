"""p_fn scorer diagnostics + saturation regularization (Pitfall #5).

Extracted from :mod:`src.losses.fn_weighted_infonce` so the FN-weighted loss
itself stays under R10's 200-line budget. Two responsibilities:

1. :func:`pfn_diagnostics` — pure-tensor descriptive statistics over the
   scorer's raw output ``p_fn[B, B] ∈ [0, 1]`` and its clipped form (the
   per-pair weight floor inside the FN denominator). Used by the loss to
   build its R7 dict and by analysis scripts (``analyze_pfn_saturation``).

2. :func:`pfn_regularization` — two penalties that fight the trivial
   collapse described in CLAUDE.md Pitfall #5:

   * **mean prior**  ``(mean(p_fn_raw) - target)²`` pulls the scorer toward
     a small prior probability (default 0.05) so it cannot drift the loss
     toward zero by globally suppressing negatives.
   * **cap penalty**  ``mean(relu(p_fn_raw / max_cap - cap_margin)²)``
     punishes values that crowd the clip cap.

DEVIATION FROM SPEC (intentional, documented per CLAUDE.md §5):
  The cap penalty operates on the *raw* (sigmoid-bounded) ``p_fn_raw``, not
  on the cap-clipped tensor the spec wrote. ``p_fn.clamp(0, max_cap)`` has
  zero gradient above ``max_cap``, so penalizing the clipped tensor would
  carry no gradient back to the scorer in the exact state we need to undo —
  the already-saturated case. Raw inputs are still in ``[0, 1]`` thanks to
  the scorer's sigmoid, so the ratio is bounded.

No labels enter either function (H2 contract).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def pfn_diagnostics(
    p_fn_raw: torch.Tensor,
    p_fn_clipped: torch.Tensor,
    max_fn_weight_current: float,
    near_zero_threshold: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Descriptive stats for the raw scorer output and its clipped form.

    Returns a dict of *detached* zero-dim tensors safe to log row-by-row.
    Bernoulli entropy is computed per entry (treating each ``p_ij`` as the
    parameter of an independent Bernoulli), clamped for log stability.
    """
    raw_mean = p_fn_raw.mean()
    raw_std = p_fn_raw.std(unbiased=False)
    raw_min = p_fn_raw.min()
    raw_max = p_fn_raw.max()

    clip_mean = p_fn_clipped.mean()
    clip_std = p_fn_clipped.std(unbiased=False)
    clip_min = p_fn_clipped.min()
    clip_max = p_fn_clipped.max()

    # Bernoulli entropy per entry — clamp for finite log; mean over [B,B].
    p_safe = p_fn_raw.clamp(min=1e-12, max=1.0 - 1e-12)
    entropy = -(p_safe * p_safe.log() + (1.0 - p_safe) * (1.0 - p_safe).log())
    entropy_mean = entropy.mean()

    if max_fn_weight_current > 0:
        at_cap = (p_fn_raw >= max_fn_weight_current).float().mean()
    else:
        at_cap = torch.zeros((), device=p_fn_raw.device, dtype=p_fn_raw.dtype)
    near_zero = (p_fn_raw <= near_zero_threshold).float().mean()

    return {
        "p_fn_raw_mean": raw_mean.detach(),
        "p_fn_raw_std": raw_std.detach(),
        "p_fn_raw_min": raw_min.detach(),
        "p_fn_raw_max": raw_max.detach(),
        "p_fn_clipped_mean": clip_mean.detach(),
        "p_fn_clipped_std": clip_std.detach(),
        "p_fn_clipped_min": clip_min.detach(),
        "p_fn_clipped_max": clip_max.detach(),
        "p_fn_at_cap_fraction": at_cap.detach(),
        "p_fn_near_zero_fraction": near_zero.detach(),
        "p_fn_entropy_mean": entropy_mean.detach(),
    }


def pfn_regularization(
    p_fn_raw: torch.Tensor,
    max_fn_weight_current: float,
    cfg: Optional[dict] = None,
) -> dict[str, torch.Tensor]:
    """Saturation-fighting penalties on the raw scorer output.

    ``cfg`` keys (all optional, sensible defaults below):
      enabled (bool, default False)
      target_mean (float, default 0.05)
      lambda_mean (float, default 0.0)
      lambda_cap (float, default 0.0)
      lambda_entropy (float, default 0.0)
      cap_margin (float, default 0.98)

    Returns ``loss_pfn_mean``, ``loss_pfn_cap``, ``loss_pfn_entropy``,
    ``pfn_reg_total`` (all live tensors — caller adds to its objective and
    backprops), plus ``target_pfn_mean`` for logging.

    With ``enabled=False`` (or all lambdas zero), every term is a zero
    tensor — backward-compat with R3 ``p_fn=0 == InfoNCE`` is preserved
    because the caller adds ``pfn_reg_total`` *outside* the contrastive
    ``loss`` key.
    """
    cfg = dict(cfg or {})
    enabled = bool(cfg.get("enabled", False))
    target_mean = float(cfg.get("target_mean", 0.05))
    lambda_mean = float(cfg.get("lambda_mean", 0.0))
    lambda_cap = float(cfg.get("lambda_cap", 0.0))
    lambda_entropy = float(cfg.get("lambda_entropy", 0.0))
    cap_margin = float(cfg.get("cap_margin", 0.98))

    device = p_fn_raw.device
    dtype = p_fn_raw.dtype
    zero = torch.zeros((), device=device, dtype=dtype)

    if enabled and lambda_mean > 0:
        loss_pfn_mean = (p_fn_raw.mean() - target_mean) ** 2
    else:
        loss_pfn_mean = zero

    # Cap penalty: operate on the raw (sigmoid-bounded) value — see module
    # docstring for the deviation rationale. With max_fn_weight_current=0
    # (e.g. warmup), the cap is inactive and we skip the term.
    if enabled and lambda_cap > 0 and max_fn_weight_current > 0:
        denom = max(max_fn_weight_current, 1e-3)
        ratio = p_fn_raw / denom
        loss_pfn_cap = F.relu(ratio - cap_margin).pow(2).mean()
    else:
        loss_pfn_cap = zero

    # Entropy penalty (logged by default; opt-in via lambda_entropy > 0).
    # We penalize LOW entropy (collapse to constant) -> minimize -H.
    if enabled and lambda_entropy > 0:
        p_safe = p_fn_raw.clamp(min=1e-12, max=1.0 - 1e-12)
        entropy = -(p_safe * p_safe.log() + (1.0 - p_safe) * (1.0 - p_safe).log())
        loss_pfn_entropy = -entropy.mean()
    else:
        loss_pfn_entropy = zero

    pfn_reg_total = (
        lambda_mean * loss_pfn_mean
        + lambda_cap * loss_pfn_cap
        + lambda_entropy * loss_pfn_entropy
    )

    return {
        "loss_pfn_mean": loss_pfn_mean,
        "loss_pfn_cap": loss_pfn_cap,
        "loss_pfn_entropy": loss_pfn_entropy,
        "pfn_reg_total": pfn_reg_total,
        "target_pfn_mean": torch.tensor(target_mean, device=device, dtype=dtype),
    }
