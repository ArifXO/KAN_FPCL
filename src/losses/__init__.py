"""Loss functions for KAN-FPCL (R7: dict[str, Tensor] returns)."""

from .infonce import InfoNCELoss
from .fn_weighted_infonce import FNWeightedInfoNCELoss
from .masks import build_positive_mask

__all__ = ["InfoNCELoss", "FNWeightedInfoNCELoss", "build_positive_mask"]
