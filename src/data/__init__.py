"""Data pipeline entry point (R6).

get_dataloader(cfg) is the single entry point for training and evaluation
scripts. It reads all parameters from the Hydra config — no hardcoded values.
"""

from __future__ import annotations

from torch.utils.data import DataLoader
from omegaconf import DictConfig

from .chestmnist import ChestMNISTDataset
from .augmentations import (
    TwoViewTransform,
    build_contrastive_transform,
    build_eval_transform,
)
from .splits import patient_level_split

__all__ = [
    "ChestMNISTDataset",
    "TwoViewTransform",
    "build_contrastive_transform",
    "build_eval_transform",
    "patient_level_split",
    "get_dataloader",
]


def get_dataloader(cfg: DictConfig) -> dict[str, DataLoader]:
    """Build train/val/test DataLoaders from a Hydra data config.

    Train split uses ``TwoViewTransform`` (contrastive pairs).
    Val/test splits use a single deterministic eval transform.

    Args:
        cfg: Hydra DictConfig with keys: size, batch_size, num_workers,
             download, normalize.mean, normalize.std, split_seed.

    Returns:
        dict with keys "train", "val", "test", each a DataLoader.
    """
    mean = list(cfg.normalize.mean)
    std = list(cfg.normalize.std)
    size = cfg.size
    download = cfg.get("download", False)

    train_transform = TwoViewTransform(
        build_contrastive_transform(size=size, mean=mean, std=std)
    )
    eval_transform = build_eval_transform(size=size, mean=mean, std=std)

    datasets = {
        "train": ChestMNISTDataset(
            split="train", transform=train_transform, download=download, size=size
        ),
        "val": ChestMNISTDataset(
            split="val", transform=eval_transform, download=download, size=size
        ),
        "test": ChestMNISTDataset(
            split="test", transform=eval_transform, download=download, size=size
        ),
    }

    loaders: dict[str, DataLoader] = {}
    for split_name, ds in datasets.items():
        loaders[split_name] = DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=(split_name == "train"),
            num_workers=cfg.num_workers,
            pin_memory=cfg.get("pin_memory", False),
            drop_last=(split_name == "train"),
        )

    return loaders
