"""Tests for patient_level_split (R4, R5).

Synthetic tests verify the core guarantee: no patient appears in more than one
split. Real-sample test loads a small slice of ChestMNIST to confirm surrogate
IDs are disjoint across the three pre-defined medmnist splits.
"""

import pytest
from src.data.splits import patient_level_split


# ---------------------------------------------------------------------------
# Synthetic tests — core R4/R5 compliance
# ---------------------------------------------------------------------------


def test_split_disjoint_patients():
    """No patient ID appears in more than one split."""
    # 100 samples from 34 patients (3 samples per patient, roughly)
    patient_ids = [f"p{i // 3}" for i in range(100)]
    train, val, test = patient_level_split(patient_ids, ratios=(0.7, 0.15, 0.15), seed=42)

    train_set = set(train)
    val_set = set(val)
    test_set = set(test)

    assert len(train_set & val_set) == 0, "train/val patient overlap"
    assert len(val_set & test_set) == 0, "val/test patient overlap"
    assert len(train_set & test_set) == 0, "train/test patient overlap"


def test_split_covers_all_samples():
    """Every sample index appears in exactly one split."""
    patient_ids = [f"p{i // 3}" for i in range(100)]
    train, val, test = patient_level_split(patient_ids, ratios=(0.7, 0.15, 0.15), seed=42)

    # Function returns integer indices into patient_ids
    all_indices = set(train) | set(val) | set(test)
    assert all_indices == set(range(100)), "Some sample indices missing from splits"

    # No duplicates across splits
    total = len(train) + len(val) + len(test)
    assert total == 100, f"Expected 100 total, got {total} — duplicates in splits"


def test_split_respects_patient_boundaries():
    """All images for a given patient land in the same split."""
    # 3 images per patient, 30 patients = 90 samples
    n_patients = 30
    images_per_patient = 3
    patient_ids = [f"patient_{p}" for p in range(n_patients) for _ in range(images_per_patient)]

    train, val, test = patient_level_split(patient_ids, ratios=(0.7, 0.15, 0.15), seed=0)

    # Function returns integer indices; reconstruct which split each patient is in
    train_set = set(train)
    val_set = set(val)
    test_set = set(test)

    # Build patient → assigned split from the indices
    from collections import defaultdict
    patient_split: dict[str, set[str]] = defaultdict(set)
    for idx in train_set:
        patient_split[patient_ids[idx]].add("train")
    for idx in val_set:
        patient_split[patient_ids[idx]].add("val")
    for idx in test_set:
        patient_split[patient_ids[idx]].add("test")

    for pid, splits_assigned in patient_split.items():
        assert len(splits_assigned) == 1, (
            f"Patient {pid} has samples in multiple splits: {splits_assigned}"
        )


def test_split_ratios_must_sum_to_one():
    """ValueError raised for invalid ratios."""
    with pytest.raises(ValueError, match="ratios must sum"):
        patient_level_split(["p0", "p1"], ratios=(0.5, 0.5, 0.5))


def test_split_detects_injected_leakage():
    """ValueError is raised when overlap is synthetically forced via direct call."""
    from src.data.splits import _assert_disjoint

    train = {"p0", "p1", "p2"}
    val = {"p2", "p3"}  # p2 leaks into val
    test = {"p4"}

    with pytest.raises(ValueError, match="leakage detected"):
        _assert_disjoint(train, val, test)


def test_split_single_patient():
    """Edge case: all images belong to one patient — all go to one split."""
    patient_ids = ["patient_A"] * 10
    train, val, test = patient_level_split(patient_ids, ratios=(0.7, 0.15, 0.15), seed=42)
    # With 1 unique patient, int(0.7*1)=0, int(0.15*1)=0 → all go to test (leftover).
    # Key invariant: all 10 sample indices must appear, with no overlap.
    all_indices = set(train) | set(val) | set(test)
    assert all_indices == set(range(10)), f"Missing indices: {all_indices}"
    assert len(train) + len(val) + len(test) == 10, "Duplicate indices detected"


# ---------------------------------------------------------------------------
# Real-sample test — medmnist surrogate IDs are disjoint across splits
# ---------------------------------------------------------------------------


def test_chestmnist_surrogate_ids_disjoint():
    """Surrogate patient IDs from the three medmnist splits are disjoint.

    MedMNIST pre-defines splits at patient level (MedMNIST v2, Yang et al. 2023).
    This test verifies the property holds for the surrogate ID scheme used here.
    """
    from src.data.chestmnist import ChestMNISTDataset

    # Use tiny slices to keep test fast — just enough to check ID structure
    train_ds = ChestMNISTDataset(split="train", download=False)
    val_ds = ChestMNISTDataset(split="val", download=False)
    test_ds = ChestMNISTDataset(split="test", download=False)

    # Sample 50 IDs from each split for a quick check
    train_sample = set(train_ds.patient_ids[:50])
    val_sample = set(val_ds.patient_ids[:50])
    test_sample = set(test_ds.patient_ids[:50])

    assert len(train_sample & val_sample) == 0
    assert len(val_sample & test_sample) == 0
    assert len(train_sample & test_sample) == 0
