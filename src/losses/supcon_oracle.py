"""Multi-label Supervised Contrastive (SupCon) Oracle for ChestMNIST.

**Oracle / upper-bound baseline** — consumes ChestMNIST training labels
during contrastive pre-training, so it is NOT self-supervised. Provides
the label-aware ceiling for InfoNCE / Debiased CL / FN-weighted / edge-aware
KAN-FPCL comparisons.

Generalises SupCon (Khosla et al. 2020) to multi-label data: positives are
pairs sharing ≥1 label (``any_overlap``) or with Jaccard > ``min_jaccard``
(``jaccard``, also used as detached positive weight). The two augmented
views of the same image are positives at weight 1.0 when
``include_self_view_positive=True``. ``no_positive_policy`` controls
zero-positive anchors. R7 dict return, never silently NaN.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .multilabel_masks import (
    build_self_view_positive_mask,
    build_supcon_oracle_positive_mask,
    expand_labels_for_two_views,
    multilabel_jaccard_matrix,
)

_VALID_MODES = ("any_overlap", "jaccard")
_VALID_NO_POS = ("self_view_only", "drop_anchor")


class MultilabelSupConOracleLoss(nn.Module):
    """Multi-label SupCon Oracle loss. Forward: ``z`` ``[2B, D]``, ``labels``
    ``[B, C]`` (un-duplicated multi-hot). See module docstring for args."""

    def __init__(
        self,
        temperature: float = 0.1,
        normalize_embeddings: bool = True,
        positive_mode: str = "any_overlap",
        min_jaccard: float = 0.0,
        include_self_view_positive: bool = True,
        no_positive_policy: str = "self_view_only",
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(
                f"SupCon Oracle temperature must be > 0, got {temperature}."
            )
        if positive_mode not in _VALID_MODES:
            raise ValueError(
                f"positive_mode must be one of {_VALID_MODES}, got "
                f"{positive_mode!r}."
            )
        if not (0.0 <= float(min_jaccard) < 1.0):
            raise ValueError(
                f"min_jaccard must be in [0, 1), got {min_jaccard}."
            )
        if no_positive_policy not in _VALID_NO_POS:
            raise ValueError(
                f"no_positive_policy must be one of {_VALID_NO_POS}, got "
                f"{no_positive_policy!r}."
            )
        self.temperature = float(temperature)
        self.normalize_embeddings = bool(normalize_embeddings)
        self.positive_mode = positive_mode
        self.min_jaccard = float(min_jaccard)
        self.include_self_view_positive = bool(include_self_view_positive)
        self.no_positive_policy = no_positive_policy

    def forward(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if z.dim() != 2:
            raise ValueError(
                f"SupCon Oracle expects z of shape [2B, D], got {tuple(z.shape)}."
            )
        n = z.shape[0]
        if n % 2 != 0 or n < 4:
            raise ValueError(
                f"SupCon Oracle expects 2B with B>=2, got 2B={n}."
            )
        batch = n // 2
        if labels.dim() != 2 or labels.shape[0] != batch:
            raise ValueError(
                f"labels must be [B, C] with B={batch}, got {tuple(labels.shape)}."
            )

        if self.normalize_embeddings:
            z = F.normalize(z, dim=-1, eps=1e-12)

        sim_raw = z @ z.t()
        logits = sim_raw / self.temperature

        diag_mask = torch.eye(n, dtype=torch.bool, device=z.device)
        # log_prob[i, j] = logits[i, j] - logsumexp_{a != i} logits[i, a]
        masked_logits = logits.masked_fill(diag_mask, float("-inf"))
        log_denom = torch.logsumexp(masked_logits, dim=-1, keepdim=True)
        log_prob = masked_logits - log_denom  # [2B, 2B]
        # NUMERICAL: the diagonal is -inf and pos_mask zeroes its weight; but
        # ``0 * -inf == NaN`` in PyTorch, so neutralize the diagonal to 0
        # before the weighted sum (it carries zero weight either way).
        log_prob = log_prob.masked_fill(diag_mask, 0.0)

        # Positive mask (label-overlap + optional self-view), then weights.
        pos_mask = build_supcon_oracle_positive_mask(
            labels=labels,
            positive_mode=self.positive_mode,
            min_jaccard=self.min_jaccard,
            include_self_view_positive=self.include_self_view_positive,
        ).to(z.device)

        if self.positive_mode == "jaccard":
            labels_2v = expand_labels_for_two_views(labels).to(z.device)
            weights = multilabel_jaccard_matrix(labels_2v).detach()
            if self.include_self_view_positive:
                # Self-view positives carry weight 1.0 regardless of labels.
                sv = build_self_view_positive_mask(batch).to(z.device)
                weights = torch.where(sv, torch.ones_like(weights), weights)
        else:
            weights = pos_mask.float()  # uniform 1.0 over positives.

        weights = weights * pos_mask.float()  # zero outside positives.

        # Per-anchor weighted mean log-prob over positives.
        w_sum = weights.sum(dim=-1)  # [2B]
        has_pos = w_sum > 0

        # Anchors with zero-weight positives: apply policy.
        n_pos = pos_mask.sum(dim=-1)  # [2B] integer count of positives
        if self.no_positive_policy == "self_view_only":
            sv = build_self_view_positive_mask(batch).to(z.device)
            fallback_w = sv.float()
            need_fallback = ~has_pos
            weights = torch.where(
                need_fallback.unsqueeze(-1).expand_as(weights),
                fallback_w,
                weights,
            )
            w_sum = weights.sum(dim=-1)
            anchor_mask = w_sum > 0  # True everywhere if self-view restores it.
            dropped = ~anchor_mask
        else:  # drop_anchor
            anchor_mask = has_pos
            dropped = ~has_pos

        # Avoid 0/0; the loss for dropped anchors is masked out anyway.
        safe_w_sum = w_sum.clamp_min(1e-12)
        per_anchor = -(weights * log_prob).sum(dim=-1) / safe_w_sum  # [2B]

        if anchor_mask.any():
            loss = per_anchor[anchor_mask].mean()
        else:
            raise RuntimeError(
                "SupCon Oracle: every anchor was dropped (no positives, no "
                "self-view fallback). Check labels / positive_mode."
            )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"SupCon Oracle produced non-finite loss={loss.item()}. "
                f"temperature={self.temperature}, mode={self.positive_mode}."
            )

        neg_mask = ~pos_mask & ~diag_mask
        out = {
            "loss": loss,
            "num_oracle_positives_mean": n_pos.float().mean().detach(),
            "num_oracle_positives_min": n_pos.float().min().detach(),
            "num_oracle_positives_max": n_pos.float().max().detach(),
            "positive_pair_fraction": pos_mask.float().mean().detach(),
            "dropped_anchor_fraction": dropped.float().mean().detach(),
            "pos_sim_mean": (
                sim_raw[pos_mask].mean().detach()
                if pos_mask.any() else torch.tensor(float("nan"), device=z.device)
            ),
            "neg_sim_mean": (
                sim_raw[neg_mask].mean().detach()
                if neg_mask.any() else torch.tensor(float("nan"), device=z.device)
            ),
            "temperature": torch.tensor(self.temperature, device=z.device),
            "positive_mode": self.positive_mode,
            "min_jaccard": float(self.min_jaccard),
        }
        return out
