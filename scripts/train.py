"""Stage 2 training entry point (Hydra, R6/R8).

Runs a contrastive training loop using the resolved Hydra config and saves the
R8 artifact set to ``<output_root>/run_<run_id>/``. Adds a cosine LR schedule,
gradient clipping, per-step metric logging (``step_metrics.json``), and
validation-loss monitoring with a ``model_best.pt`` checkpoint.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault(
    "PROJECT_ROOT", str(Path(__file__).resolve().parent.parent).replace("\\", "/")
)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import hydra
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.data import get_dataloader
from src.utils import make_run_id, save_run_artifacts, set_seed

from train_common import (
    base_step_metrics,
    build_val_loader,
    run_validation,
    snapshot_cpu_state,
)


@hydra.main(version_base=None, config_path="../configs/experiment", config_name="smoke_mlp")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.run.seed, deterministic=cfg.run.get("deterministic", False))
    device = torch.device(cfg.run.device)

    encoder = instantiate(cfg.model.encoder).to(device)
    head = instantiate(cfg.model.head).to(device)
    loss_fn = instantiate(cfg.loss).to(device)
    full_model = nn.ModuleDict({"encoder": encoder, "head": head})

    loaders = get_dataloader(cfg.data)
    train_loader = loaders["train"]
    val_loader = build_val_loader(cfg)

    params = list(encoder.parameters()) + list(head.parameters())
    optim = torch.optim.Adam(params, lr=cfg.train.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=cfg.train.max_steps, eta_min=cfg.train.get("lr_min", 1e-6)
    )

    grad_clip = cfg.train.get("grad_clip", 1.0)
    val_every = cfg.train.get("val_every", 200)
    val_max_batches = cfg.train.get("val_max_batches", None)
    val_seed = int(cfg.run.seed) + 10000

    def val_forward(v1: torch.Tensor, v2: torch.Tensor) -> dict:
        z = head(encoder(torch.cat([v1, v2], dim=0)))
        return loss_fn(z)

    losses: list[float] = []
    step_metrics: list[dict] = []
    val_loss_curve: list[dict] = []
    best_val_loss = float("inf")
    best_val_step = -1
    best_state: dict | None = None

    t0 = time.time()
    encoder.train()
    head.train()
    step = 0
    epoch = 0
    while step < cfg.train.max_steps:
        epoch += 1
        for (v1, v2), _labels, _pids in train_loader:
            if step >= cfg.train.max_steps:
                break
            v1, v2 = v1.to(device), v2.to(device)
            z = head(encoder(torch.cat([v1, v2], dim=0)))
            out = loss_fn(z)
            loss = out["loss"]

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            optim.step()
            scheduler.step()

            losses.append(float(loss.item()))
            step_dict = base_step_metrics(step, scheduler.get_last_lr()[0], out, epoch=epoch)
            if hasattr(head, "alpha") and head.alpha is not None:
                step_dict["alpha"] = float(head.alpha.detach())
            step_metrics.append(step_dict)

            if step % cfg.train.log_every == 0:
                alpha_str = ""
                if hasattr(head, "alpha") and head.alpha is not None:
                    alpha_str = f" alpha={float(head.alpha.detach()):.4f}"
                print(
                    f"[step {step}] loss={loss.item():.4f} "
                    f"pos_sim={out['pos_sim_mean'].item():.4f} "
                    f"neg_sim={out['neg_sim_mean'].item():.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}{alpha_str}"
                )

            if step % val_every == 0:
                encoder.eval()
                head.eval()
                val_loss = run_validation(val_loader, device, val_forward, val_max_batches, val_seed)
                encoder.train()
                head.train()
                val_loss_curve.append({"step": step, "val_loss": val_loss})
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_step = step
                    best_state = snapshot_cpu_state(full_model)
                print(f"[val step {step}] val_loss={val_loss:.4f} best={best_val_loss:.4f}")

            step += 1

    runtime = time.time() - t0

    run_id = make_run_id()
    run_dir = Path(cfg.run.output_root) / f"run_{cfg.run.name}_{run_id}"

    if best_state is None:
        best_state = snapshot_cpu_state(full_model)
        best_val_loss = float("nan")

    metrics = {
        "train_loss_final": losses[-1] if losses else float("nan"),
        "train_loss_curve": losses,
        "val_loss_curve": val_loss_curve,
        "best_val_loss": best_val_loss,
        "best_val_step": best_val_step,
        "train_time_sec": runtime,
        "steps": step,
        "total_epochs": epoch,
        "run_id": run_id,
        "seed": int(cfg.run.seed),
    }

    save_run_artifacts(
        run_dir=run_dir,
        cfg=cfg,
        model=full_model,
        named_modules={"encoder": encoder, "head": head},
        metrics=metrics,
    )
    (run_dir / "step_metrics.json").write_text(json.dumps(step_metrics, indent=2))
    torch.save(best_state, run_dir / "model_best.pt")
    print(f"Run artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
