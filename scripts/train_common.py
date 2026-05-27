"""Shared training utilities for the Stage 2 / 6 / 7.5 entry points.

Validation note (important): contrastive losses score positive *pairs*, so
validation needs two augmented views per image — the loss expects the
``[2B, D]`` view-concatenated layout (positive of ``i`` is ``i + B``). A single
deterministic ``build_eval_transform`` yields one view per image and cannot
form that layout, so :func:`build_val_loader` uses ``TwoViewTransform`` over
the contrastive augmentation pipeline on the disjoint patient-level ``val``
split. :func:`run_validation` seeds the augmentation RNG per pass (restoring
the training RNG afterwards) so ``val_loss`` is comparable step-to-step and
does not perturb the training augmentation stream.
"""

from __future__ import annotations

from typing import Callable

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.data import TwoViewTransform, build_contrastive_transform, get_dataset


def build_val_loader(cfg: DictConfig) -> DataLoader:
    """Two-view val loader over the disjoint ``val`` split (shuffle off)."""
    mean = list(cfg.data.normalize.mean)
    std = list(cfg.data.normalize.std)
    transform = TwoViewTransform(
        build_contrastive_transform(size=cfg.data.size, mean=mean, std=std)
    )
    dataset = get_dataset(cfg.data, "val", transform)
    return DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        drop_last=True,
    )


def snapshot_cpu_state(model: torch.nn.Module) -> dict:
    """Detached CPU clone of ``model.state_dict()`` for best-checkpoint saving."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def base_step_metrics(step: int, lr: float, out: dict, epoch: int = 0) -> dict:
    """One per-step row from a loss dict; NaN for keys a loss does not emit.

    Single source of truth for the R7 logging keys shared by all three train
    scripts. ``out["loss"]`` carries grad, so it is detached before casting.
    """
    return {
        "step": step,
        "epoch": epoch,
        "lr": lr,
        "loss": float(out["loss"].detach()),
        "pos_sim_mean": float(out.get("pos_sim_mean", float("nan"))),
        "neg_sim_mean": float(out.get("neg_sim_mean", float("nan"))),
        "p_fn_mean": float(out.get("p_fn_mean", float("nan"))),
        "p_fn_max": float(out.get("p_fn_max", float("nan"))),
        "downweighted_fraction": float(out.get("downweighted_fraction", float("nan"))),
        "fn_loss": float(out.get("fn_loss", float("nan"))),
        "edge_contrastive_loss": float(out.get("edge_contrastive_loss", float("nan"))),
        "edge_align_loss": float(out.get("edge_align_loss", float("nan"))),
        "p_fn_reg_loss": float(out.get("p_fn_reg_loss", float("nan"))),
        "total_loss": float(out.get("total_loss", float("nan"))),
        "contrastive_loss": float(out.get("contrastive_loss", float("nan"))),
        "alpha": float(out.get("alpha", float("nan"))),
    }


def run_validation(
    val_loader: DataLoader,
    device: torch.device,
    forward_fn: Callable[[torch.Tensor, torch.Tensor], dict],
    max_batches: int | None,
    val_seed: int,
) -> float:
    """Mean ``loss`` over (optionally capped) val batches.

    ``forward_fn(v1, v2)`` must return the loss dict for the two views. The
    caller is responsible for setting modules to ``eval()`` before and
    ``train()`` after. Augmentation RNG is fixed to ``val_seed`` for the pass
    and the prior RNG state is restored on exit.
    """
    rng_state = torch.get_rng_state()
    torch.manual_seed(val_seed)
    total, count = 0.0, 0
    try:
        with torch.no_grad():
            for (v1, v2), _labels, _pids in val_loader:
                if max_batches is not None and count >= max_batches:
                    break
                out = forward_fn(v1.to(device), v2.to(device))
                total += float(out["loss"])
                count += 1
    finally:
        torch.set_rng_state(rng_state)

    if count == 0:
        raise RuntimeError(
            "Validation produced 0 batches. val split too small for "
            f"batch_size={val_loader.batch_size} with drop_last=True, or "
            f"val_max_batches misconfigured (max_batches={max_batches})."
        )
    return total / count
