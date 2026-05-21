"""Stage 8 geometry analysis on frozen checkpoint embeddings (R6)."""

from __future__ import annotations

import csv
import math
import os
import time
from pathlib import Path

os.environ.setdefault(
    "PROJECT_ROOT", str(Path(__file__).resolve().parent.parent).replace("\\", "/")
)

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.data import (
    ChestMNISTDataset,
    TwoViewTransform,
    build_contrastive_transform,
    build_eval_transform,
)
from src.metrics import (
    alignment,
    effective_rank,
    off_diagonal_covariance_norm,
    per_dim_std,
    uniformity,
)
from src.utils import set_seed


_CSV_COLUMNS = [
    "run_id", "encoder", "head", "loss", "seed", "alignment", "uniformity",
    "effective_rank", "per_dim_std_mean", "offdiag_covariance", "dataset",
]


def _load_model(ckpt_dir: Path, device: torch.device) -> tuple[nn.Module, nn.Module]:
    cfg_path = ckpt_dir / "config.yaml"
    model_path = ckpt_dir / "model.pt"
    if not ckpt_dir.exists():
        raise ValueError(f"checkpoint_dir does not exist: {ckpt_dir}.")
    if not cfg_path.exists():
        raise ValueError(f"No config.yaml found in checkpoint_dir: {ckpt_dir}.")
    if not model_path.exists():
        raise ValueError(f"No model.pt found in checkpoint_dir: {ckpt_dir}.")

    ckpt_cfg = OmegaConf.load(cfg_path)
    encoder = instantiate(ckpt_cfg.model.encoder).to(device)
    head = instantiate(ckpt_cfg.model.head).to(device)
    state = torch.load(model_path, map_location=device)
    encoder.load_state_dict(
        {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
    )
    head.load_state_dict(
        {k[len("head."):]: v for k, v in state.items() if k.startswith("head.")}
    )
    encoder.eval()
    head.eval()
    return encoder, head


def _single_loader(cfg: DictConfig, split: str) -> DataLoader:
    mean = list(cfg.data.normalize.mean)
    std = list(cfg.data.normalize.std)
    transform = build_eval_transform(size=cfg.data.size, mean=mean, std=std)
    dataset = ChestMNISTDataset(
        split=split,
        transform=transform,
        download=cfg.data.get("download", False),
        size=cfg.data.size,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )


def _pair_loader(cfg: DictConfig, split: str) -> DataLoader:
    mean = list(cfg.data.normalize.mean)
    std = list(cfg.data.normalize.std)
    transform = TwoViewTransform(
        build_contrastive_transform(size=cfg.data.size, mean=mean, std=std)
    )
    dataset = ChestMNISTDataset(
        split=split,
        transform=transform,
        download=cfg.data.get("download", False),
        size=cfg.data.size,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )


def _embed(encoder: nn.Module, head: nn.Module, images: torch.Tensor) -> torch.Tensor:
    return F.normalize(head(encoder(images)), dim=-1, eps=1e-12)


def _extract_embeddings(
    encoder: nn.Module,
    head: nn.Module,
    cfg: DictConfig,
    device: torch.device,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    max_samples = cfg.geometry.get("max_samples")
    split_names = [str(split) for split in cfg.geometry.eval_splits]
    per_split = None
    if max_samples is not None:
        per_split = max(1, math.ceil(int(max_samples) / len(split_names)))

    with torch.no_grad():
        for split in split_names:
            seen = 0
            for images, _labels, _pids in _single_loader(cfg, split):
                z_batch = _embed(encoder, head, images.to(device)).cpu()
                if per_split is not None:
                    remaining = per_split - seen
                    if remaining <= 0:
                        break
                    z_batch = z_batch[:remaining]
                chunks.append(z_batch)
                seen += z_batch.shape[0]
                if per_split is not None and seen >= per_split:
                    break
    z = torch.cat(chunks, dim=0)
    if max_samples is not None and z.shape[0] > int(max_samples):
        generator = torch.Generator().manual_seed(int(cfg.geometry.seed))
        idx = torch.randperm(z.shape[0], generator=generator)[: int(max_samples)]
        z = z[idx]
    return z


def _extract_alignment(
    encoder: nn.Module,
    head: nn.Module,
    cfg: DictConfig,
    device: torch.device,
) -> torch.Tensor:
    z_a, z_b = [], []
    max_samples = cfg.geometry.get("max_samples")
    seen = 0
    with torch.no_grad():
        for (v1, v2), _labels, _pids in _pair_loader(
            cfg, str(cfg.geometry.alignment_split)
        ):
            z1 = _embed(encoder, head, v1.to(device)).cpu()
            z2 = _embed(encoder, head, v2.to(device)).cpu()
            if max_samples is not None:
                remaining = int(max_samples) - seen
                if remaining <= 0:
                    break
                z1, z2 = z1[:remaining], z2[:remaining]
            z_a.append(z1)
            z_b.append(z2)
            seen += z1.shape[0]
            if max_samples is not None and seen >= int(max_samples):
                break
    return alignment(torch.cat(z_a, dim=0), torch.cat(z_b, dim=0))


def _append_csv_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


@hydra.main(version_base=None, config_path="../configs/experiment", config_name="geometry")
def main(cfg: DictConfig) -> None:
    if len(cfg.geometry.eval_splits) == 0:
        raise ValueError("geometry.eval_splits must contain at least one split.")
    max_samples = cfg.geometry.get("max_samples")
    if max_samples is not None and int(max_samples) < 2:
        raise ValueError("geometry.max_samples must be at least 2 when set.")

    set_seed(cfg.geometry.seed)
    device = torch.device(cfg.geometry.device)
    ckpt_dir = Path(cfg.geometry.checkpoint_dir)
    encoder, head = _load_model(ckpt_dir, device)

    t0 = time.time()
    z = _extract_embeddings(encoder, head, cfg, device)
    align = _extract_alignment(encoder, head, cfg, device)

    row = {
        "run_id": ckpt_dir.name,
        "encoder": cfg.meta.encoder,
        "head": cfg.meta.head,
        "loss": cfg.meta.loss,
        "seed": int(cfg.geometry.seed),
        "alignment": round(float(align.item()), 6),
        "uniformity": round(float(uniformity(z).item()), 6),
        "effective_rank": round(float(effective_rank(z).item()), 6),
        "per_dim_std_mean": round(float(per_dim_std(z).mean().item()), 6),
        "offdiag_covariance": round(float(off_diagonal_covariance_norm(z).item()), 6),
        "dataset": cfg.meta.dataset,
    }
    _append_csv_row(Path(cfg.geometry.output_csv), row)

    runtime = time.time() - t0
    print(
        f"[geometry] alignment={row['alignment']:.4f} "
        f"uniformity={row['uniformity']:.4f} "
        f"effective_rank={row['effective_rank']:.2f} runtime={runtime:.1f}s"
    )
    print(f"[geometry] Row appended to {cfg.geometry.output_csv}")


if __name__ == "__main__":
    main()
