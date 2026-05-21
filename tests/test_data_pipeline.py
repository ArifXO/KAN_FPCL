"""Tests for ChestMNIST dataset and augmentation pipeline (R4, R5, R9).

Verifies batch shapes, label dtype, two-view output, and dataset sizes
against known medmnist split counts.
"""

import torch
import pytest
from torch.utils.data import DataLoader

from src.data.chestmnist import ChestMNISTDataset, NUM_CLASSES
from src.data.augmentations import (
    TwoViewTransform,
    build_contrastive_transform,
    build_eval_transform,
)


# ---------------------------------------------------------------------------
# Dataset __getitem__ contract
# ---------------------------------------------------------------------------


class TestChestMNISTDataset:
    @pytest.fixture(scope="class")
    def eval_ds(self):
        transform = build_eval_transform(size=28)
        return ChestMNISTDataset(split="train", transform=transform, download=False)

    def test_len_matches_medmnist(self, eval_ds):
        # medmnist ChestMNIST train split: 78 468 samples
        assert len(eval_ds) == 78468

    def test_getitem_returns_three_elements(self, eval_ds):
        item = eval_ds[0]
        assert len(item) == 3, "Expected (image, label, patient_id)"

    def test_image_tensor_shape(self, eval_ds):
        img, _, _ = eval_ds[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (1, 28, 28), f"Unexpected shape: {img.shape}"

    def test_image_tensor_dtype_float(self, eval_ds):
        img, _, _ = eval_ds[0]
        assert img.dtype == torch.float32

    def test_image_range_after_normalize(self, eval_ds):
        img, _, _ = eval_ds[0]
        # Normalized with mean=0.5, std=0.5 → roughly [-1, 1]
        assert img.min() >= -3.0 and img.max() <= 3.0

    def test_label_tensor_shape(self, eval_ds):
        _, label, _ = eval_ds[0]
        assert isinstance(label, torch.Tensor)
        assert label.shape == (NUM_CLASSES,), f"Expected ({NUM_CLASSES},), got {label.shape}"

    def test_label_dtype_float32(self, eval_ds):
        _, label, _ = eval_ds[0]
        assert label.dtype == torch.float32, (
            f"Labels must be float32 for BCE losses, got {label.dtype}"
        )

    def test_label_values_binary(self, eval_ds):
        _, label, _ = eval_ds[0]
        unique = label.unique()
        assert all(v in (0.0, 1.0) for v in unique.tolist())

    def test_patient_id_format(self, eval_ds):
        _, _, pid = eval_ds[0]
        assert pid == "cmnist_train_0"
        _, _, pid2 = eval_ds[1]
        assert pid2 == "cmnist_train_1"

    def test_split_sizes(self):
        for split, expected_len in [("train", 78468), ("val", 11219), ("test", 22433)]:
            ds = ChestMNISTDataset(split=split, download=False)
            assert len(ds) == expected_len, (
                f"Split '{split}': expected {expected_len}, got {len(ds)}"
            )

    def test_invalid_split_raises(self):
        with pytest.raises(ValueError, match="split must be one of"):
            ChestMNISTDataset(split="bogus")


# ---------------------------------------------------------------------------
# TwoViewTransform
# ---------------------------------------------------------------------------


class TestTwoViewTransform:
    @pytest.fixture(scope="class")
    def two_view_ds(self):
        transform = TwoViewTransform(build_contrastive_transform(size=28))
        return ChestMNISTDataset(split="val", transform=transform, download=False)

    def test_getitem_returns_tuple_of_views(self, two_view_ds):
        views, label, pid = two_view_ds[0]
        assert isinstance(views, tuple), "TwoViewTransform should return a tuple"
        assert len(views) == 2

    def test_each_view_correct_shape(self, two_view_ds):
        (v1, v2), _, _ = two_view_ds[0]
        assert v1.shape == (1, 28, 28)
        assert v2.shape == (1, 28, 28)

    def test_views_differ_due_to_random_augmentation(self, two_view_ds):
        """Two views from the same image should usually differ (stochastic transform)."""
        diffs = []
        for i in range(10):
            (v1, v2), _, _ = two_view_ds[i]
            diffs.append((v1 - v2).abs().mean().item())
        # At least some views must differ — if all are identical the transform is broken
        assert any(d > 0.01 for d in diffs), (
            "TwoViewTransform produced identical views — check that augmentations are stochastic"
        )


# ---------------------------------------------------------------------------
# DataLoader batch shapes
# ---------------------------------------------------------------------------


class TestDataLoaderBatchShapes:
    @pytest.fixture(scope="class")
    def train_loader(self):
        transform = TwoViewTransform(build_contrastive_transform(size=28))
        ds = ChestMNISTDataset(split="train", transform=transform, download=False)
        return DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

    @pytest.fixture(scope="class")
    def eval_loader(self):
        transform = build_eval_transform(size=28)
        ds = ChestMNISTDataset(split="val", transform=transform, download=False)
        return DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

    def test_train_batch_two_views(self, train_loader):
        batch = next(iter(train_loader))
        views, labels, pids = batch
        assert isinstance(views, (list, tuple)) and len(views) == 2
        v1, v2 = views
        assert v1.shape == (8, 1, 28, 28)
        assert v2.shape == (8, 1, 28, 28)

    def test_train_labels_shape(self, train_loader):
        _, labels, _ = next(iter(train_loader))
        assert labels.shape == (8, NUM_CLASSES)
        assert labels.dtype == torch.float32

    def test_eval_batch_single_view(self, eval_loader):
        batch = next(iter(eval_loader))
        imgs, labels, pids = batch
        assert imgs.shape == (8, 1, 28, 28)

    def test_eval_labels_shape(self, eval_loader):
        _, labels, _ = next(iter(eval_loader))
        assert labels.shape == (8, NUM_CLASSES)
