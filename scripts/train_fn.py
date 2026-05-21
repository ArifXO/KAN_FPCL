"""Stage 6 training entry point (Hydra, R6/R8): FN-weighted InfoNCE + scorer.

Trains encoder + projection head + MLP pair scorer jointly with
:class:`src.losses.FNWeightedInfoNCELoss`. The pair scorer's ``p_fn`` matrix
is fed into the loss denominator to downweight likely false negatives.

Saves the full R8 artifact bundle (config, weights, metrics, params, git) to
``<output_root>/run_<name>_<run_id>/``. The encoder+head sub-state is keyed
so :mod:`scripts.probe` can load them while ignoring scorer weights.
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


@hydra.main(version_base=None, config_path="../configs/experiment", config_name="smoke_fn_mlp")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.run.seed, deterministic=cfg.run.get("deterministic", False))
    device = torch.device(cfg.run.device)

    encoder = instantiate(cfg.model.encoder).to(device)
    head = instantiate(cfg.model.head).to(device)
    scorer = instantiate(cfg.model.scorer).to(device)
    loss_fn = instantiate(cfg.loss).to(device)

    loaders = get_dataloader(cfg.data)
    train_loader = loaders["train"]

    params = (
        list(encoder.parameters())
        + list(head.parameters())
        + list(scorer.parameters())
    )
    optim = torch.optim.Adam(params, lr=cfg.train.lr)

    losses: list[float] = []
    t0 = time.time()
    encoder.train()
    head.train()
    scorer.train()
    step = 0
    for (v1, v2), _labels, _pids in train_loader:
        if step >= cfg.train.max_steps:
            break
        v1, v2 = v1.to(device), v2.to(device)
        x = torch.cat([v1, v2], dim=0)
        feats = encoder(x)
        z = head(feats)

        batch = v1.shape[0]
        z_view1 = z[:batch]
        p_fn = scorer(z_view1.detach())  # scorer trained via loss grad below

        out = loss_fn(z, p_fn)
        loss = out["loss"]

        optim.zero_grad()
        loss.backward()
        optim.step()

        losses.append(float(loss.item()))
        if step % cfg.train.log_every == 0:
            print(
                f"[step {step}] loss={loss.item():.4f} "
                f"pos_sim={out['pos_sim_mean'].item():.4f} "
                f"neg_sim={out['neg_sim_mean'].item():.4f} "
                f"p_fn_mean={out['p_fn_mean'].item():.4f}"
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

    full_model = nn.ModuleDict({"encoder": encoder, "head": head, "scorer": scorer})
    save_run_artifacts(
        run_dir=run_dir,
        cfg=cfg,
        model=full_model,
        named_modules={"encoder": encoder, "head": head, "scorer": scorer},
        metrics=metrics,
    )
    print(f"Run artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
