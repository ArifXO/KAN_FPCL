"""Stage 2 training entry point (Hydra, R6/R8).

Runs a short contrastive training loop using the resolved Hydra config and
saves the R8 artifact set to ``<output_root>/run_<run_id>/``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault(
    "PROJECT_ROOT", str(Path(__file__).resolve().parent.parent).replace("\\", "/")
)

import hydra
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.data import get_dataloader
from src.utils import make_run_id, save_run_artifacts, set_seed


@hydra.main(version_base=None, config_path="../configs/experiment", config_name="smoke_mlp")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.run.seed, deterministic=cfg.run.get("deterministic", False))
    device = torch.device(cfg.run.device)

    encoder = instantiate(cfg.model.encoder).to(device)
    head = instantiate(cfg.model.head).to(device)
    loss_fn = instantiate(cfg.loss).to(device)

    loaders = get_dataloader(cfg.data)
    train_loader = loaders["train"]

    params = list(encoder.parameters()) + list(head.parameters())
    optim = torch.optim.Adam(params, lr=cfg.train.lr)

    losses: list[float] = []
    t0 = time.time()
    encoder.train()
    head.train()
    step = 0
    for (v1, v2), _labels, _pids in train_loader:
        if step >= cfg.train.max_steps:
            break
        v1, v2 = v1.to(device), v2.to(device)
        x = torch.cat([v1, v2], dim=0)
        feats = encoder(x)
        z = head(feats)
        out = loss_fn(z)
        loss = out["loss"]

        optim.zero_grad()
        loss.backward()
        optim.step()

        losses.append(float(loss.item()))
        if step % cfg.train.log_every == 0:
            print(
                f"[step {step}] loss={loss.item():.4f} "
                f"pos_sim={out['pos_sim_mean'].item():.4f} "
                f"neg_sim={out['neg_sim_mean'].item():.4f}"
            )
        step += 1

    runtime = time.time() - t0

    run_id = make_run_id()
    run_dir = Path(cfg.run.output_root) / f"run_{cfg.run.name}_{run_id}"

    metrics = {
        "train_loss": losses,
        "train_time_sec": runtime,
        "steps": step,
        "run_id": run_id,
    }

    full_model = nn.ModuleDict({"encoder": encoder, "head": head})
    save_run_artifacts(
        run_dir=run_dir,
        cfg=cfg,
        model=full_model,
        named_modules={"encoder": encoder, "head": head},
        metrics=metrics,
    )
    print(f"Run artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
