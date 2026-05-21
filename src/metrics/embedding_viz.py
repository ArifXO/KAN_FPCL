"""Small embedding visualization helper with UMAP/PCA reduction."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _as_2d_array(name: str, values) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{name} must have shape [N, D], got {arr.shape}.")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty, got {arr.shape}.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or Inf values.")
    return arr


def _label_ids(labels, n_rows: int) -> np.ndarray:
    if labels is None:
        return np.zeros(n_rows, dtype=np.int64)
    arr = np.asarray(labels)
    if arr.shape[0] != n_rows:
        raise ValueError(
            f"labels length must match embeddings rows, got {arr.shape[0]} vs {n_rows}."
        )
    if arr.ndim == 1:
        return arr.astype(np.int64)
    if arr.ndim == 2:
        return arr.argmax(axis=1).astype(np.int64)
    raise ValueError(f"labels must be 1D or 2D, got shape {arr.shape}.")


def _reduce(
    embeddings: np.ndarray,
    use_pca_fallback: bool,
    random_state: int,
) -> tuple[np.ndarray, str]:
    if embeddings.shape[0] == 1:
        return np.zeros((1, 2), dtype=np.float32), "single_point"
    try:
        from umap import UMAP
    except ImportError as exc:
        if not use_pca_fallback:
            raise ImportError(
                "umap-learn is not installed. Install it or set "
                "use_pca_fallback=True."
            ) from exc
        n_components = min(2, embeddings.shape[0], embeddings.shape[1])
        coords = PCA(n_components=n_components).fit_transform(embeddings)
        if n_components == 1:
            coords = np.column_stack([coords[:, 0], np.zeros(coords.shape[0])])
        return coords.astype(np.float32), "pca"

    coords = UMAP(n_components=2, random_state=random_state).fit_transform(embeddings)
    return coords.astype(np.float32), "umap"


def _scaled(coords: np.ndarray, width: int, height: int, pad: int) -> np.ndarray:
    mins = coords.min(axis=0)
    spans = np.maximum(coords.max(axis=0) - mins, 1e-12)
    xy = (coords - mins) / spans
    xy[:, 0] = pad + xy[:, 0] * (width - 2 * pad)
    xy[:, 1] = height - (pad + xy[:, 1] * (height - 2 * pad))
    return xy


def _save_csv(path: Path, coords: np.ndarray, labels: np.ndarray, method: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "label", "method"])
        for (x, y), label in zip(coords, labels):
            writer.writerow([float(x), float(y), int(label), method])


def _save_svg(path: Path, coords: np.ndarray, labels: np.ndarray, method: str) -> None:
    width, height, pad = 720, 520, 28
    points = _scaled(coords, width, height, pad)
    circles = []
    for (x, y), label in zip(points, labels):
        color = _PALETTE[int(label) % len(_PALETTE)]
        circles.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.2" '
            f'fill="{color}" fill-opacity="0.82" />'
        )
    path.write_text(
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
                f'height="{height}" viewBox="0 0 {width} {height}">',
                '<rect width="100%" height="100%" fill="white" />',
                f'<text x="{pad}" y="20" font-size="14" fill="#222">{method}</text>',
                *circles,
                "</svg>",
            ]
        )
    )


def _save_png(path: Path, coords: np.ndarray, labels: np.ndarray) -> None:
    try:
        from PIL import Image, ImageColor, ImageDraw
    except ImportError as exc:
        raise ImportError("Pillow is required to save PNG embedding plots.") from exc

    width, height, pad = 720, 520, 28
    points = _scaled(coords, width, height, pad)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    for (x, y), label in zip(points, labels):
        color = ImageColor.getrgb(_PALETTE[int(label) % len(_PALETTE)])
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
    image.save(path)


def save_umap(
    embeddings,
    labels,
    output_path: str | Path,
    use_pca_fallback: bool = True,
    random_state: int = 42,
) -> np.ndarray:
    """Save a 2D UMAP plot, falling back to PCA when ``umap-learn`` is absent.

    Supported output suffixes are ``.svg``, ``.png``, and ``.csv``. The returned
    value is the computed ``[N, 2]`` coordinate array for downstream inspection.
    """
    emb = _as_2d_array("embeddings", embeddings)
    label_ids = _label_ids(labels, emb.shape[0])
    coords, method = _reduce(emb, use_pca_fallback, random_state)

    path = Path(output_path)
    if path.suffix == "":
        path = path.with_suffix(".svg")
    path.parent.mkdir(parents=True, exist_ok=True)

    suffix = path.suffix.lower()
    if suffix == ".csv":
        _save_csv(path, coords, label_ids, method)
    elif suffix == ".svg":
        _save_svg(path, coords, label_ids, method)
    elif suffix == ".png":
        _save_png(path, coords, label_ids)
    else:
        raise ValueError(
            f"output_path suffix must be .svg, .png, or .csv, got {path.suffix!r}."
        )
    return coords


__all__ = ["save_umap"]
