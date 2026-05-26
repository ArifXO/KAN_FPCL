"""Data pipeline entry point (R6).

get_dataset() and get_dataloader() are the single entry points for training
and evaluation scripts. All dataset construction routes through get_dataset()
— no dataset class is constructed directly outside this module.
"""

from __future__ import annotations

from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig

from .chestmnist import ChestMNISTDataset
from .augmentations import (
    TwoViewTransform,
    build_contrastive_transform,
    build_eval_transform,
)

__all__ = [
    "ChestMNISTDataset",
    "TwoViewTransform",
    "build_contrastive_transform",
    "build_eval_transform",
    "get_dataset",
    "get_dataloader",
]


def get_dataset(data_cfg: DictConfig, split: str, transform) -> Dataset:
    """Return the correct Dataset for data_cfg.name."""
    name = data_cfg.name.lower()
    if name == "chestmnist":
        return ChestMNISTDataset(
            split=split,
            transform=transform,
            download=data_cfg.get("download", False),
            size=data_cfg.get("size", 28),
        )
    raise ValueError(
        f"Unknown dataset '{name}'. Supported: ['chestmnist']. "
        f"CheXpert is not yet integrated — see CLAUDE.md Dataset Scope."
    )


def get_dataloader(cfg: DictConfig) -> dict[str, DataLoader]:
    """Build train/val/test DataLoaders from a Hydra data config.

    Train split uses TwoViewTransform (contrastive pairs).
    Val/test splits use a single deterministic eval transform.

    Args:
        cfg: Hydra DictConfig with keys: name, size, batch_size, num_workers,
             download, normalize.mean, normalize.std.

    Returns:
        dict with keys "train", "val", "test", each a DataLoader.
    """
    mean = list(cfg.normalize.mean)
    std = list(cfg.normalize.std)
    size = cfg.size

    train_transform = TwoViewTransform(
        build_contrastive_transform(size=size, mean=mean, std=std)
    )
    eval_transform = build_eval_transform(size=size, mean=mean, std=std)

    datasets = {
        "train": get_dataset(cfg, "train", train_transform),
        "val": get_dataset(cfg, "val", eval_transform),
        "test": get_dataset(cfg, "test", eval_transform),
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
