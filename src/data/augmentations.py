"""Augmentation transforms for contrastive learning on chest X-rays (R4)."""

from __future__ import annotations

import torchvision.transforms as T


def build_contrastive_transform(
    size: int = 28,
    mean: list[float] | None = None,
    std: list[float] | None = None,
    crop_scale: tuple[float, float] = (0.6, 1.0),
    rotation: int = 15,
    brightness: float = 0.3,
    contrast: float = 0.3,
    saturation: float = 0.0,
    hue: float = 0.0,
    flip_prob: float = 0.5,
) -> T.Compose:
    """Strong augmentation transform for contrastive pairs.

    No vertical flip — vertical orientation carries diagnostic meaning in CXR.
    Defaults match configs/data/chestmnist.yaml augmentation block (R6).
    """
    if mean is None:
        mean = [0.5]
    if std is None:
        std = [0.5]

    return T.Compose(
        [
            T.RandomResizedCrop(size, scale=crop_scale),
            T.RandomHorizontalFlip(p=flip_prob),
            T.RandomRotation(degrees=rotation),
            T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def build_eval_transform(
    size: int = 28,
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> T.Compose:
    """Deterministic transform for val/test evaluation."""
    if mean is None:
        mean = [0.5]
    if std is None:
        std = [0.5]

    return T.Compose(
        [
            T.Resize(size),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


class TwoViewTransform:
    """Apply the same transform twice with independent RNG draws.

    Returns a tuple (view1, view2) from the same source image.
    Used for contrastive training where two augmented views form a positive pair.
    """

    def __init__(self, transform: T.Compose) -> None:
        self.transform = transform

    def __call__(self, x):
        return self.transform(x), self.transform(x)
