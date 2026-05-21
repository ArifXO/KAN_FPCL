"""Patient-level dataset splitting utility (R4, R5)."""

from __future__ import annotations

import random
from collections import defaultdict


def patient_level_split(
    patient_ids: list[str],
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """Split sample indices into disjoint train/val/test by patient ID.

    Args:
        patient_ids: Per-sample patient ID, length N (multiple samples per
            patient allowed).
        ratios: (train, val, test) fractions — must sum to 1.0.
        seed: RNG seed for reproducible splits.

    Returns:
        (train_indices, val_indices, test_indices) — index lists into
        ``patient_ids``, guaranteed disjoint at patient level.

    Raises:
        ValueError: If any patient appears in more than one split.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {sum(ratios):.6f}")

    # Group sample indices by patient ID
    patient_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, pid in enumerate(patient_ids):
        patient_to_indices[pid].append(idx)

    unique_patients = list(patient_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(unique_patients)

    n = len(unique_patients)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)

    train_patients = set(unique_patients[:n_train])
    val_patients = set(unique_patients[n_train : n_train + n_val])
    test_patients = set(unique_patients[n_train + n_val :])

    # R5: verify disjointness before returning
    _assert_disjoint(train_patients, val_patients, test_patients)

    def collect(patients: set[str]) -> list[str]:
        idxs: list[str] = []
        for pid in patients:
            idxs.extend(patient_to_indices[pid])
        return idxs

    return collect(train_patients), collect(val_patients), collect(test_patients)


def _assert_disjoint(
    train_ids: set[str],
    val_ids: set[str],
    test_ids: set[str],
) -> None:
    """Raise ValueError if any patient appears in more than one split (R5)."""
    tv = train_ids & val_ids
    vt = val_ids & test_ids
    tt = train_ids & test_ids
    if tv or vt or tt:
        raise ValueError(
            f"Data leakage detected. Overlapping patient IDs: "
            f"train∩val={len(tv)}, val∩test={len(vt)}, train∩test={len(tt)}. "
            "Check patient ID extraction and split logic."
        )
