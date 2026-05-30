"""Multilabel positive-mask builders for SupCon Oracle (R3, R10).

The two-view contrastive layout is ``z = [v1; v2]`` of shape ``[2B, D]``;
labels are ``[B, C]`` multi-hot and must be expanded to ``[2B, C]`` in the
same ordering. These helpers produce ``[2B, 2B]`` boolean / float masks
keyed off label overlap or Jaccard similarity.

Oracle note: these masks consume ChestMNIST training labels and are only
used by :class:`src.losses.MultilabelSupConOracleLoss`. Do NOT route them
through the FN scorer, KAN scorer, or edge-aware KAN-FPCL paths — those
are label-free by design.
"""

from __future__ import annotations

import torch


def expand_labels_for_two_views(labels: torch.Tensor) -> torch.Tensor:
    """``[B, C]`` -> ``[2B, C]`` by repeating each row twice in v1/v2 order.

    Matches the ``z = [v1; v2]`` ordering used everywhere else in the codebase:
    rows ``0..B-1`` are the first view and rows ``B..2B-1`` are the second view
    of the SAME originals.
    """
    if labels.dim() != 2:
        raise ValueError(
            f"labels must be [B, C], got shape {tuple(labels.shape)}."
        )
    return torch.cat([labels, labels], dim=0)


def multilabel_overlap_matrix(labels_2v: torch.Tensor) -> torch.Tensor:
    """``[2B, 2B]`` integer intersection counts ``|L_i ∩ L_j|``."""
    if labels_2v.dim() != 2:
        raise ValueError(
            f"labels_2v must be [2B, C], got shape {tuple(labels_2v.shape)}."
        )
    return labels_2v.float() @ labels_2v.float().t()


def multilabel_jaccard_matrix(labels_2v: torch.Tensor) -> torch.Tensor:
    """``[2B, 2B]`` Jaccard similarity ``|L_i ∩ L_j| / |L_i ∪ L_j|``.

    Empty-vs-empty (no labels on either side) yields 0 so a row of all-zero
    labels never produces NaN — those anchors fall through to the
    self-view-only positive in the SupCon Oracle loss.
    """
    if labels_2v.dim() != 2:
        raise ValueError(
            f"labels_2v must be [2B, C], got shape {tuple(labels_2v.shape)}."
        )
    f = labels_2v.float()
    inter = f @ f.t()
    sums = f.sum(dim=-1, keepdim=True)  # [2B, 1]
    union = sums + sums.t() - inter
    # NUMERICAL: 0/0 -> 0, never NaN. clamp_min before division.
    return inter / union.clamp_min(1e-12)


def build_self_view_positive_mask(B: int) -> torch.Tensor:
    """``[2B, 2B]`` bool mask: True at ``(i, i+B)`` and ``(i+B, i)`` only."""
    if B <= 0:
        raise ValueError(f"B must be positive, got {B}.")
    n = 2 * B
    mask = torch.zeros(n, n, dtype=torch.bool)
    idx = torch.arange(B)
    mask[idx, idx + B] = True
    mask[idx + B, idx] = True
    return mask


def build_supcon_oracle_positive_mask(
    labels: torch.Tensor,
    positive_mode: str = "any_overlap",
    min_jaccard: float = 0.0,
    include_self_view_positive: bool = True,
) -> torch.Tensor:
    """``[2B, 2B]`` bool positive mask for SupCon Oracle.

    Args:
        labels: ``[B, C]`` multi-hot labels.
        positive_mode: ``"any_overlap"`` (|L_i ∩ L_j| > 0) or ``"jaccard"``
            (Jaccard(L_i, L_j) > ``min_jaccard``).
        min_jaccard: Threshold for ``"jaccard"`` mode (ignored otherwise).
        include_self_view_positive: Always mark the augmented view of the
            same image as positive, even if its (zero) labels disagree.

    Returns:
        Bool ``[2B, 2B]`` mask with the diagonal always False.
    """
    if positive_mode not in ("any_overlap", "jaccard"):
        raise ValueError(
            f"positive_mode must be 'any_overlap' or 'jaccard', got "
            f"{positive_mode!r}."
        )
    if not (0.0 <= float(min_jaccard) < 1.0):
        raise ValueError(
            f"min_jaccard must be in [0, 1), got {min_jaccard}."
        )
    B = labels.shape[0]
    n = 2 * B
    labels_2v = expand_labels_for_two_views(labels)

    if positive_mode == "any_overlap":
        mask = multilabel_overlap_matrix(labels_2v) > 0
    else:
        mask = multilabel_jaccard_matrix(labels_2v) > float(min_jaccard)

    if include_self_view_positive:
        sv = build_self_view_positive_mask(B).to(mask.device)
        mask = mask | sv

    # Diagonal is never a positive (self-comparison).
    diag = torch.eye(n, dtype=torch.bool, device=mask.device)
    mask = mask & ~diag
    return mask
