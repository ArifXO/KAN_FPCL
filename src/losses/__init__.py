"""Loss functions for KAN-FPCL (R7: dict[str, Tensor] returns)."""

from .infonce import InfoNCELoss
from .masks import build_positive_mask

__all__ = ["InfoNCELoss", "build_positive_mask"]
