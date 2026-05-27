"""CheXpert dataset wrapper (R4, R5, R9, R10).

Patient prefix is extracted from the ``Path`` column
(``.../patient00001/study1/view1_frontal.jpg``). Uncertainty handling
follows the four canonical CheXpert policies (Irvin et al., 2019).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


NUM_CLASSES = 14

LABEL_COLUMNS: tuple[str, ...] = (
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
)

_VALID_SPLITS = ("train", "val", "test")
_VALID_VIEWS = ("frontal", "lateral", "all")
_VALID_UNCERTAINTY = ("ignore", "positive", "negative", "LSR")


def patient_id_from_path(path: str) -> str:
    """Return the ``patientXXXXX`` token from a CheXpert image path."""
    for token in Path(path).parts:
        if token.startswith("patient"):
            return token
    raise ValueError(
        f"No 'patient*' segment found in CheXpert path {path!r}. "
        "Manifest is malformed or uses an unsupported layout."
    )


def _apply_uncertainty_policy(
    raw: list[Optional[float]], policy: str
) -> Optional[np.ndarray]:
    """Map raw label cells {1, 0, -1, NaN} to a binary vector under ``policy``.

    Returns ``None`` when ``policy='ignore'`` and any cell is -1 (caller
    drops the row).
    """
    out = np.zeros(NUM_CLASSES, dtype=np.float32)
    for i, cell in enumerate(raw):
        if cell is None or (isinstance(cell, float) and np.isnan(cell)):
            continue
        v = float(cell)
        if v == 1.0:
            out[i] = 1.0
        elif v == 0.0:
            out[i] = 0.0
        elif v == -1.0:
            if policy == "ignore":
                return None
            if policy == "positive":
                out[i] = 1.0
            elif policy == "negative":
                out[i] = 0.0
            elif policy == "LSR":
                out[i] = 0.5
        else:
            raise ValueError(
                f"Unexpected CheXpert label cell {v} (column index {i}); "
                "expected one of {1, 0, -1, NaN}."
            )
    return out


def _parse_cell(cell: str) -> Optional[float]:
    if cell == "" or cell.lower() == "nan":
        return None
    return float(cell)


class CheXpertDataset(Dataset):
    """CheXpert CXR dataset with configurable uncertainty handling.

    Args:
        root: Directory containing manifest CSV and the CheXpert image tree.
        split: ``train`` / ``val`` (maps to ``valid.csv``) / ``test``.
        view: ``frontal`` / ``lateral`` / ``all``.
        uncertainty_policy: ``ignore`` (drop row), ``positive`` (-1→1),
            ``negative`` (-1→0), ``LSR`` (-1→0.5).
        image_size: Target side in pixels for resize at load time.
        transform: Optional callable applied to the PIL image after open.
        manifest_override: Custom CSV path (test fixtures).
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        view: str = "frontal",
        uncertainty_policy: str = "ignore",
        image_size: int = 224,
        transform: Optional[Callable] = None,
        manifest_override: Optional[str | Path] = None,
    ) -> None:
        if split not in _VALID_SPLITS:
            raise ValueError(f"split must be one of {_VALID_SPLITS}, got {split!r}.")
        if view not in _VALID_VIEWS:
            raise ValueError(f"view must be one of {_VALID_VIEWS}, got {view!r}.")
        if uncertainty_policy not in _VALID_UNCERTAINTY:
            raise ValueError(
                f"uncertainty_policy must be one of {_VALID_UNCERTAINTY}, "
                f"got {uncertainty_policy!r}."
            )
        if image_size <= 0:
            raise ValueError(f"image_size must be > 0, got {image_size}.")

        self.root = Path(root)
        self.split = split
        self.view = view
        self.uncertainty_policy = uncertainty_policy
        self.image_size = int(image_size)
        self.transform = transform

        if manifest_override is not None:
            manifest = Path(manifest_override)
        else:
            csv_name = "valid.csv" if split == "val" else f"{split}.csv"
            manifest = self.root / csv_name

        if not manifest.exists():
            raise FileNotFoundError(
                f"CheXpert manifest not found at {manifest}. "
                "Set `root` to the directory containing train.csv/valid.csv."
            )

        self._records: list[tuple[str, str, np.ndarray]] = []
        with open(manifest, "r", newline="") as fh:
            reader = csv.DictReader(fh)
            missing = [c for c in LABEL_COLUMNS if c not in reader.fieldnames]
            if missing:
                raise ValueError(
                    f"CheXpert manifest {manifest} missing label columns: {missing}."
                )
            if "Path" not in reader.fieldnames:
                raise ValueError(f"CheXpert manifest {manifest} missing 'Path' column.")

            view_col = "Frontal/Lateral"
            has_view_col = view_col in reader.fieldnames

            for row in reader:
                if view != "all" and has_view_col:
                    cell = (row.get(view_col) or "").strip().lower()
                    if cell and cell != view.lower():
                        continue
                raw = [_parse_cell(row[c]) for c in LABEL_COLUMNS]
                labels = _apply_uncertainty_policy(raw, uncertainty_policy)
                if labels is None:
                    continue
                path = row["Path"]
                self._records.append((path, patient_id_from_path(path), labels))

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, str]:
        path, pid, labels = self._records[idx]
        full_path = self.root / path if not Path(path).is_absolute() else Path(path)
        if not full_path.exists():
            parts = Path(path).parts
            if len(parts) > 1:
                stripped = Path(*parts[1:])
                alt_path = self.root / stripped
                if alt_path.exists():
                    full_path = alt_path
                else:
                    raise FileNotFoundError(
                        f"CheXpert image not found. Tried:\n"
                        f"  1. {self.root / path}\n"
                        f"  2. {alt_path}\n"
                        f"Check that root='{self.root}' is correct."
                    )
            else:
                raise FileNotFoundError(f"CheXpert image not found: {full_path}")
        with Image.open(full_path) as im:
            img = im.convert("L")
            if (img.width, img.height) != (self.image_size, self.image_size):
                img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            if self.transform is not None:
                out = self.transform(img)
            else:
                arr = np.asarray(img, dtype=np.float32) / 255.0
                out = torch.from_numpy(arr).unsqueeze(0)
        return out, torch.from_numpy(labels), pid

    @property
    def patient_ids(self) -> list[str]:
        return [pid for (_p, pid, _l) in self._records]

    @property
    def paths(self) -> list[str]:
        return [p for (p, _pid, _l) in self._records]
