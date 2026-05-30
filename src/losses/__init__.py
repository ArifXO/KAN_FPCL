"""Loss functions for KAN-FPCL (R7: dict[str, Tensor] returns)."""

from .infonce import InfoNCELoss
from .fn_weighted_infonce import FNWeightedInfoNCELoss
from .edge_aware_fn_loss import EdgeAwareFNWeightedInfoNCELoss
from .debiased_contrastive import DebiasedContrastiveLoss
from .supcon_oracle import MultilabelSupConOracleLoss
from .edge_features import edge_fingerprint, edge_pair_similarity
from .masks import build_positive_mask

__all__ = [
    "InfoNCELoss",
    "FNWeightedInfoNCELoss",
    "EdgeAwareFNWeightedInfoNCELoss",
    "DebiasedContrastiveLoss",
    "MultilabelSupConOracleLoss",
    "edge_fingerprint",
    "edge_pair_similarity",
    "build_positive_mask",
]
