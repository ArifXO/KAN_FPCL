"""Tests for CheXpertDataset and CheXpert patient-ID extraction (R4, R5, R9).

These tests use a synthetic CSV fixture and synthetic JPEG files — they do
NOT require the real CheXpert corpus. The point is to verify the manifest
parser, uncertainty policies, view filter, and patient-level splitting
behavior in isolation.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.chexpert import CheXpertDataset
from src.data.splits import chexpert_patient_ids_from_paths, patient_level_split
from src.data.chexpert import (
    LABEL_COLUMNS,
    NUM_CLASSES,
    _apply_uncertainty_policy,
    patient_id_from_path,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_HEADER = (
    "Path,Sex,Age,Frontal/Lateral,AP/PA,"
    + ",".join(LABEL_COLUMNS)
)


def _row(path: str, view: str, labels: list[str]) -> str:
    cells = [path, "Female", "55", view, "AP", *labels]
    return ",".join(cells)


def _label_row(values: dict[str, str]) -> list[str]:
    """Render a 14-element label row from a {name: cell} dict (missing -> '')."""
    return [values.get(c, "") for c in LABEL_COLUMNS]


def _write_manifest(tmp_path: Path, rows: list[str]) -> Path:
    csv_path = tmp_path / "train.csv"
    csv_path.write_text(_HEADER + "\n" + "\n".join(rows) + "\n")
    return csv_path


def _write_image(root: Path, rel_path: str, size: int = 16) -> None:
    full = root / rel_path
    full.parent.mkdir(parents=True, exist_ok=True)
    arr = (np.random.RandomState(0).rand(size, size) * 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(full, format="JPEG")


@pytest.fixture
def synthetic_root(tmp_path: Path) -> Path:
    """Build a 5-row CheXpert manifest + matching JPEG files.

    Patients: patient00001 (2 studies), patient00002, patient00003 (1 lateral).
    Labels include one uncertain row (-1) and one with NaN cells.
    """
    rows = [
        # patient00001, frontal, clean positive
        _row(
            "CheXpert-v1.0/train/patient00001/study1/view1_frontal.jpg",
            "Frontal",
            _label_row({"Cardiomegaly": "1.0", "No Finding": "0.0"}),
        ),
        # patient00001, frontal, mostly negative
        _row(
            "CheXpert-v1.0/train/patient00001/study2/view1_frontal.jpg",
            "Frontal",
            _label_row({"No Finding": "1.0"}),
        ),
        # patient00002, frontal, has -1 uncertain
        _row(
            "CheXpert-v1.0/train/patient00002/study1/view1_frontal.jpg",
            "Frontal",
            _label_row({"Edema": "-1.0", "Cardiomegaly": "1.0"}),
        ),
        # patient00003, LATERAL view
        _row(
            "CheXpert-v1.0/train/patient00003/study1/view2_lateral.jpg",
            "Lateral",
            _label_row({"Atelectasis": "1.0"}),
        ),
        # patient00004, NaN cells (blank)
        _row(
            "CheXpert-v1.0/train/patient00004/study1/view1_frontal.jpg",
            "Frontal",
            _label_row({"Pneumonia": "1.0"}),
        ),
    ]
    _write_manifest(tmp_path, rows)
    for rel in [
        "CheXpert-v1.0/train/patient00001/study1/view1_frontal.jpg",
        "CheXpert-v1.0/train/patient00001/study2/view1_frontal.jpg",
        "CheXpert-v1.0/train/patient00002/study1/view1_frontal.jpg",
        "CheXpert-v1.0/train/patient00003/study1/view2_lateral.jpg",
        "CheXpert-v1.0/train/patient00004/study1/view1_frontal.jpg",
    ]:
        _write_image(tmp_path, rel)
    return tmp_path


# ---------------------------------------------------------------------------
# patient_id_from_path
# ---------------------------------------------------------------------------


def test_patient_id_extracted_from_path():
    assert (
        patient_id_from_path(
            "CheXpert-v1.0/train/patient00042/study3/view1_frontal.jpg"
        )
        == "patient00042"
    )


def test_patient_id_missing_raises():
    with pytest.raises(ValueError, match="No 'patient\\*' segment"):
        patient_id_from_path("CheXpert-v1.0/train/study1/img.jpg")


def test_chexpert_patient_ids_from_paths_batches():
    paths = [
        "CheXpert-v1.0/train/patient00001/study1/view1_frontal.jpg",
        "CheXpert-v1.0/train/patient00002/study1/view1_frontal.jpg",
    ]
    assert chexpert_patient_ids_from_paths(paths) == ["patient00001", "patient00002"]


def test_chexpert_patient_ids_from_paths_propagates_error():
    paths = ["CheXpert-v1.0/train/bogus/study1/img.jpg"]
    with pytest.raises(ValueError, match="lack a 'patient"):
        chexpert_patient_ids_from_paths(paths)


# ---------------------------------------------------------------------------
# _apply_uncertainty_policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "policy,expected_first",
    [("positive", 1.0), ("negative", 0.0), ("LSR", 0.5)],
)
def test_uncertainty_policies_map_negative_one(policy, expected_first):
    raw = [-1.0] + [0.0] * (NUM_CLASSES - 1)
    out = _apply_uncertainty_policy(raw, policy)
    assert out is not None
    assert out[0] == expected_first


def test_uncertainty_ignore_returns_none_when_any_uncertain():
    raw = [-1.0] + [0.0] * (NUM_CLASSES - 1)
    assert _apply_uncertainty_policy(raw, "ignore") is None


def test_uncertainty_ignore_keeps_clean_row():
    raw = [1.0] + [0.0] * (NUM_CLASSES - 1)
    out = _apply_uncertainty_policy(raw, "ignore")
    assert out is not None and out[0] == 1.0


def test_uncertainty_invalid_policy_rejected_at_construction(synthetic_root):
    with pytest.raises(ValueError, match="uncertainty_policy"):
        CheXpertDataset(
            root=synthetic_root,
            split="train",
            uncertainty_policy="bogus",
            image_size=16,
        )


def test_uncertainty_invalid_cell_value_raises():
    raw = [2.0] + [0.0] * (NUM_CLASSES - 1)
    with pytest.raises(ValueError, match="Unexpected CheXpert label cell"):
        _apply_uncertainty_policy(raw, "positive")


# ---------------------------------------------------------------------------
# CheXpertDataset construction
# ---------------------------------------------------------------------------


def test_dataset_loads_with_ignore_drops_uncertain(synthetic_root):
    ds = CheXpertDataset(
        root=synthetic_root,
        split="train",
        view="frontal",
        uncertainty_policy="ignore",
        image_size=16,
    )
    # 4 frontals total minus 1 uncertain row = 3
    assert len(ds) == 3
    assert "patient00002" not in ds.patient_ids
    assert "patient00003" not in ds.patient_ids  # filtered by view


def test_dataset_loads_with_positive_keeps_uncertain(synthetic_root):
    ds = CheXpertDataset(
        root=synthetic_root,
        split="train",
        view="frontal",
        uncertainty_policy="positive",
        image_size=16,
    )
    # All 4 frontal rows kept.
    assert len(ds) == 4
    # patient00002 had Edema=-1 -> 1 under 'positive'
    idx = ds.patient_ids.index("patient00002")
    _img, lbl, _pid = ds[idx]
    edema_col = LABEL_COLUMNS.index("Edema")
    assert lbl[edema_col] == 1.0


def test_dataset_lsr_maps_to_half(synthetic_root):
    ds = CheXpertDataset(
        root=synthetic_root,
        split="train",
        view="frontal",
        uncertainty_policy="LSR",
        image_size=16,
    )
    idx = ds.patient_ids.index("patient00002")
    _img, lbl, _pid = ds[idx]
    edema_col = LABEL_COLUMNS.index("Edema")
    assert lbl[edema_col] == 0.5


def test_dataset_view_filter_lateral(synthetic_root):
    ds = CheXpertDataset(
        root=synthetic_root,
        split="train",
        view="lateral",
        uncertainty_policy="positive",
        image_size=16,
    )
    assert len(ds) == 1
    assert ds.patient_ids == ["patient00003"]


def test_dataset_view_all_keeps_everyone(synthetic_root):
    ds = CheXpertDataset(
        root=synthetic_root,
        split="train",
        view="all",
        uncertainty_policy="positive",
        image_size=16,
    )
    assert len(ds) == 5


def test_dataset_invalid_split_raises(synthetic_root):
    with pytest.raises(ValueError, match="split must be one of"):
        CheXpertDataset(root=synthetic_root, split="bogus", image_size=16)


def test_dataset_invalid_view_raises(synthetic_root):
    with pytest.raises(ValueError, match="view must be one of"):
        CheXpertDataset(root=synthetic_root, view="oblique", image_size=16)


def test_dataset_missing_manifest_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest not found"):
        CheXpertDataset(root=tmp_path, split="train", image_size=16)


def test_dataset_missing_label_column_raises(tmp_path):
    bad_header = "Path,Frontal/Lateral," + ",".join(LABEL_COLUMNS[:-1])
    rows = [
        "CheXpert-v1.0/train/patient00001/study1/view1_frontal.jpg,Frontal,"
        + ",".join(["0.0"] * (NUM_CLASSES - 1))
    ]
    (tmp_path / "train.csv").write_text(bad_header + "\n" + "\n".join(rows) + "\n")
    with pytest.raises(ValueError, match="missing label columns"):
        CheXpertDataset(root=tmp_path, split="train", image_size=16)


# ---------------------------------------------------------------------------
# __getitem__ contract
# ---------------------------------------------------------------------------


def test_getitem_shapes_and_dtypes(synthetic_root):
    ds = CheXpertDataset(
        root=synthetic_root,
        split="train",
        view="frontal",
        uncertainty_policy="ignore",
        image_size=16,
    )
    img, label, pid = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (1, 16, 16)
    assert img.dtype == torch.float32
    assert label.shape == (NUM_CLASSES,)
    assert label.dtype == torch.float32
    assert pid.startswith("patient")


# ---------------------------------------------------------------------------
# R4/R5: patient-level splitting on CheXpert paths
# ---------------------------------------------------------------------------


def test_patient_level_split_on_synthetic_chexpert(synthetic_root):
    """Two studies of patient00001 must land in the SAME split."""
    ds = CheXpertDataset(
        root=synthetic_root,
        split="train",
        view="all",
        uncertainty_policy="positive",
        image_size=16,
    )
    pids = ds.patient_ids
    train, val, test = patient_level_split(pids, ratios=(0.6, 0.2, 0.2), seed=1)

    train_pids = {pids[i] for i in train}
    val_pids = {pids[i] for i in val}
    test_pids = {pids[i] for i in test}

    assert len(train_pids & val_pids) == 0
    assert len(val_pids & test_pids) == 0
    assert len(train_pids & test_pids) == 0


def test_patient_extraction_round_trips_through_splits(synthetic_root):
    """chexpert_patient_ids_from_paths(ds.paths) must equal ds.patient_ids."""
    ds = CheXpertDataset(
        root=synthetic_root,
        split="train",
        view="all",
        uncertainty_policy="positive",
        image_size=16,
    )
    assert chexpert_patient_ids_from_paths(ds.paths) == ds.patient_ids


# ---------------------------------------------------------------------------
# __getitem__: doubled-prefix fallback (R9)
# ---------------------------------------------------------------------------


def test_getitem_strips_doubled_prefix(tmp_path):
    """root=CheXpert-v1.0/, CSV paths start with CheXpert-v1.0/ -> strip and load."""
    dataset_dir = tmp_path / "CheXpert-v1.0"
    rel = "CheXpert-v1.0/train/patient00001/study1/view1_frontal.jpg"
    # Image lives at dataset_dir/train/...
    _write_image(dataset_dir, "train/patient00001/study1/view1_frontal.jpg", size=16)
    rows = [
        _row(rel, "Frontal", _label_row({"No Finding": "1.0"})),
    ]
    _write_manifest(dataset_dir, rows)
    ds = CheXpertDataset(
        root=dataset_dir,
        split="train",
        view="frontal",
        uncertainty_policy="ignore",
        image_size=16,
    )
    img, label, pid = ds[0]
    assert img.shape == (1, 16, 16)
    assert pid == "patient00001"


def test_getitem_filenotfounderror_shows_both_paths(tmp_path):
    """Missing image -> FileNotFoundError listing both attempted paths."""
    rel = "CheXpert-v1.0/train/patient00001/study1/view1_frontal.jpg"
    rows = [
        _row(rel, "Frontal", _label_row({"No Finding": "1.0"})),
    ]
    _write_manifest(tmp_path, rows)
    ds = CheXpertDataset(
        root=tmp_path,
        split="train",
        view="frontal",
        uncertainty_policy="ignore",
        image_size=16,
    )
    with pytest.raises(FileNotFoundError) as exc_info:
        ds[0]
    msg = str(exc_info.value)
    assert "Tried:" in msg
    assert "1." in msg
    assert "2." in msg
