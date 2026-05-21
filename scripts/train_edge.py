"""Stage 7.5 training entry point (Hydra, R6/R8): edge-aware FN-weighted InfoNCE.

Trains encoder + KAN projector head + EdgeAwarePairScorer jointly with
:class:`src.losses.EdgeAwareFNWeightedInfoNCELoss`. Edge fingerprints are
extracted from the projector's last-layer ``phi`` tensor via
:func:`src.losses.edge_fingerprint` and fed to both the scorer (when
``use_edge_features=True``) and the loss (when ``lambda_edge`` or
``lambda_edge_align`` is positive).

When the loss config has ``edge_aware: false``, the script still runs but
skips the edge-fingerprint computation entirely — behaviorally identical
to Stage 6's ``train_fn.py``.

We chose a new entrypoint (rather than overloading ``scripts/train.py``)
to keep each stage's script independently auditable (R10 spirit).
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
from omegaconf import OmegaConf

from src.data import get_dataloader
from src.losses import edge_fingerprint
from src.utils import make_run_id, save_run_artifacts, set_seed


@hydra.main(
    version_base=None,
    config_path="../configs/experiment",
    config_name="smoke_edge_aware",
)
def main(cfg: DictConfig) -> None:
    set_seed(cfg.run.seed, deterministic=cfg.run.get("deterministic", False))
    device = torch.device(cfg.run.device)

    encoder = instantiate(cfg.model.encoder).to(device)
    head = instantiate(cfg.model.head).to(device)
    scorer = instantiate(cfg.model.scorer).to(device)

    # Build the loss WITHOUT the marker flag (it's a script-level concern).
    loss_cfg = OmegaConf.to_container(cfg.loss, resolve=True)
    edge_aware = bool(loss_cfg.pop("edge_aware", True))
    loss_target = loss_cfg.pop("_target_")
    loss_fn = hydra.utils.get_class(loss_target)(**loss_cfg).to(device)

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

        if edge_aware:
            z, phi = head(feats, return_edges=True)        # phi: [2B, O, I]
            edge_feats = edge_fingerprint(phi)              # [2B, 256]
        else:
            z = head(feats)
            edge_feats = None

        batch = v1.shape[0]
        z_view1 = z[:batch]
        scorer_edge_arg = (
            edge_feats[:batch] if (edge_aware and scorer.use_edge_features) else None
        )
        p_fn = scorer(z_view1.detach(), scorer_edge_arg.detach() if scorer_edge_arg is not None else None)

        out = loss_fn(z, p_fn, edge_feats if edge_aware else None)
        loss = out["loss"]

        optim.zero_grad()
        loss.backward()
        optim.step()

        losses.append(float(loss.item()))
        if step % cfg.train.log_every == 0:
            print(
                f"[step {step}] loss={loss.item():.4f} "
                f"fn={float(out['fn_loss']):.4f} "
                f"edge_c={float(out['edge_contrastive_loss']):.4f} "
                f"edge_a={float(out['edge_align_loss']):.4f} "
                f"p_fn_mean={float(out['p_fn_mean']):.4f}"
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
        "edge_aware": edge_aware,
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
