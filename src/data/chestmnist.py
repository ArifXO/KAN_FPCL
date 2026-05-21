"""ChestMNIST dataset wrapper (R4, R5, R9, R10).

MedMNIST's pre-defined train/val/test splits are patient-level by construction
(MedMNIST v2 paper, Yang et al. 2023). The npz does not expose original patient
IDs, so surrogate IDs of the form "cmnist_{split}_{idx}" are assigned — one per
image, reflecting the 1:1 image-to-patient-occurrence structure in each split.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import Dataset
import medmnist
from medmnist import ChestMNIST as _ChestMNIST


_VALID_SPLITS = ("train", "val", "test")

NUM_CLASSES = 14


class ChestMNISTDataset(Dataset):
    """Thin wrapper around medmnist.ChestMNIST.

    Args:
        split: One of "train", "val", "test".
        transform: Callable applied to the PIL image. Pass ``TwoViewTransform``
            for contrastive training; a single-view transform for eval.
        download: Download the dataset if not cached.
        size: Image side length in pixels (28 or 64).
        root: Override the medmnist cache root.
    """

    def __init__(
        self,
        split: str,
        transform=None,
        download: bool = False,
        size: int = 28,
        root: str | None = None,
    ) -> None:
        if split not in _VALID_SPLITS:
            raise ValueError(
                f"split must be one of {_VALID_SPLITS}, got {split!r}"
            )

        kwargs: dict = dict(split=split, download=download, size=size)
        if root is not None:
            kwargs["root"] = root

        self._inner = _ChestMNIST(transform=transform, **kwargs)
        self._split = split

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int) -> tuple[Tensor | tuple, Tensor, str]:
        """Return (image_or_views, multi_label_vector, patient_id).

        ``image_or_views`` is a single Tensor when a single-view transform is
        used, or a (view1, view2) tuple when TwoViewTransform is used.
        ``multi_label_vector`` is float32 of shape [14] for BCE-style losses.
        ``patient_id`` is a surrogate string unique within the medmnist split.
        """
        img, label = self._inner[idx]
        label_tensor = torch.from_numpy(label.astype("float32")).squeeze()
        patient_id = f"cmnist_{self._split}_{idx}"
        return img, label_tensor, patient_id

    @property
    def patient_ids(self) -> list[str]:
        """Return surrogate patient IDs for all samples (used by leakage check)."""
        return [f"cmnist_{self._split}_{i}" for i in range(len(self))]

    def label_info(self) -> dict:
        return self._inner.info.get("label", {})
