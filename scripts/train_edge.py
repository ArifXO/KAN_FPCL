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

Adds a cosine LR schedule, gradient clipping, per-step metric logging
(``step_metrics.json``, including ``alpha`` when the head exposes it), and
validation-loss monitoring with a ``model_best.pt`` checkpoint. We chose a
new entrypoint (rather than overloading ``scripts/train.py``) to keep each
stage's script independently auditable (R10 spirit).
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
from omegaconf import OmegaConf

from src.data import get_dataloader
from src.losses import edge_fingerprint
from src.utils import make_run_id, save_run_artifacts, set_seed

from train_common import (
    base_step_metrics,
    build_val_loader,
    run_validation,
    snapshot_cpu_state,
)


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
    full_model = nn.ModuleDict({"encoder": encoder, "head": head, "scorer": scorer})

    # Build the loss WITHOUT the marker flag (it's a script-level concern).
    loss_cfg = OmegaConf.to_container(cfg.loss, resolve=True)
    edge_aware = bool(loss_cfg.pop("edge_aware", True))
    loss_target = loss_cfg.pop("_target_")
    loss_fn = hydra.utils.get_class(loss_target)(**loss_cfg).to(device)

    loaders = get_dataloader(cfg.data)
    train_loader = loaders["train"]
    val_loader = build_val_loader(cfg)

    params = (
        list(encoder.parameters())
        + list(head.parameters())
        + list(scorer.parameters())
    )
    optim = torch.optim.Adam(params, lr=cfg.train.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=cfg.train.max_steps, eta_min=cfg.train.get("lr_min", 1e-6)
    )

    grad_clip = cfg.train.get("grad_clip", 1.0)
    val_every = cfg.train.get("val_every", 200)
    val_max_batches = cfg.train.get("val_max_batches", None)
    val_seed = int(cfg.run.seed) + 10000
    lambda_pfn_reg = cfg.train.get("lambda_pfn_reg", 0.0)
    detach_scorer_edge = cfg.train.get("detach_scorer_edge", False)

    def forward_batch(x: torch.Tensor):
        if edge_aware:
            z, phi = head(x, return_edges=True)        # phi: [2B, O, I]
            return z, edge_fingerprint(phi)             # edge_feats: [2B, 256]
        return head(x), None

    def val_forward(v1: torch.Tensor, v2: torch.Tensor) -> dict:
        # Validation uses raw contrastive loss without p_fn regularization.
        # p_fn_reg is a training stabilizer, not a metric.
        x = torch.cat([v1, v2], dim=0)
        feats = encoder(x)
        z, edge_feats = forward_batch(feats)
        b = v1.shape[0]
        scorer_edge_arg = (
            edge_feats[:b] if (edge_aware and scorer.use_edge_features) else None
        )
        p_fn = scorer(z[:b], scorer_edge_arg)
        return loss_fn(z, p_fn, edge_feats if edge_aware else None)

    losses: list[float] = []
    step_metrics: list[dict] = []
    val_loss_curve: list[dict] = []
    best_val_loss = float("inf")
    best_val_step = -1
    best_state: dict | None = None

    t0 = time.time()
    encoder.train()
    head.train()
    scorer.train()
    step = 0
    epoch = 0
    while step < cfg.train.max_steps:
        epoch += 1
        for (v1, v2), _labels, _pids in train_loader:
            if step >= cfg.train.max_steps:
                break
            v1, v2 = v1.to(device), v2.to(device)
            feats = encoder(torch.cat([v1, v2], dim=0))
            z, edge_feats = forward_batch(feats)

            batch = v1.shape[0]
            scorer_edge_arg = (
                edge_feats[:batch] if (edge_aware and scorer.use_edge_features) else None
            )
            # z.detach(): scorer must not reshape main embeddings (SimCLR convention).
            # edge_features NOT detached: H4 requires gradient flow from the scorer's
            # p_fn decisions back through edge_fingerprint into the KAN projector
            # weights. Set train.detach_scorer_edge=true for the ablation that
            # isolates aux-loss-only training of edges from scorer-driven training.
            if scorer_edge_arg is not None and detach_scorer_edge:
                scorer_edge_arg = scorer_edge_arg.detach()
            p_fn = scorer(z[:batch].detach(), scorer_edge_arg)

            out = loss_fn(z, p_fn, edge_feats if edge_aware else None)
            contrastive_loss = out["loss"]
            p_fn_reg = lambda_pfn_reg * p_fn.mean()
            total_loss = contrastive_loss + p_fn_reg

            optim.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            optim.step()
            scheduler.step()

            losses.append(float(contrastive_loss.item()))
            step_dict = base_step_metrics(step, scheduler.get_last_lr()[0], out, epoch=epoch)
            step_dict["p_fn_reg_loss"] = float(p_fn_reg.detach())
            step_dict["total_loss"] = float(total_loss.detach())
            if hasattr(head, "alpha"):
                step_dict["alpha"] = float(head.alpha.detach())
            step_metrics.append(step_dict)

            if step % cfg.train.log_every == 0:
                print(
                    f"[step {step}] loss={total_loss.item():.4f} "
                    f"fn={float(out['fn_loss']):.4f} "
                    f"edge_c={float(out['edge_contrastive_loss']):.4f} "
                    f"edge_a={float(out['edge_align_loss']):.4f} "
                    f"p_fn_mean={float(out['p_fn_mean']):.4f} "
                    f"p_fn_max={out['p_fn_max'].item():.4f} "
                    f"downweighted={out['downweighted_fraction'].item():.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

            if step % val_every == 0:
                encoder.eval()
                head.eval()
                scorer.eval()
                val_loss = run_validation(val_loader, device, val_forward, val_max_batches, val_seed)
                encoder.train()
                head.train()
                scorer.train()
                val_loss_curve.append({"step": step, "val_loss": val_loss})
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_step = step
                    best_state = snapshot_cpu_state(full_model)
                print(f"[val step {step}] val_loss={val_loss:.4f} best={best_val_loss:.4f}")

            step += 1

    runtime = time.time() - t0

    run_id = make_run_id()
    run_dir = Path(cfg.run.output_root) / f"{cfg.run.name}_{run_id}"

    if best_state is None:
        best_state = snapshot_cpu_state(full_model)
        best_val_loss = float("nan")

    metrics = {
        "train_loss_final": losses[-1] if losses else float("nan"),
        "train_loss_curve": losses,
        "train_loss_type": "contrastive",
        "val_loss_curve": val_loss_curve,
        "best_val_loss": best_val_loss,
        "best_val_step": best_val_step,
        "train_time_sec": runtime,
        "steps": step,
        "total_epochs": epoch,
        "run_id": run_id,
        "seed": int(cfg.run.seed),
        "edge_aware": edge_aware,
    }

    save_run_artifacts(
        run_dir=run_dir,
        cfg=cfg,
        model=full_model,
        named_modules={"encoder": encoder, "head": head, "scorer": scorer},
        metrics=metrics,
    )
    (run_dir / "step_metrics.json").write_text(json.dumps(step_metrics, indent=2))
    if step_metrics:
        import csv as csv_mod
        csv_path = run_dir / "step_metrics.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv_mod.DictWriter(f, fieldnames=list(step_metrics[0].keys()))
            writer.writeheader()
            writer.writerows(step_metrics)
    torch.save(best_state, run_dir / "model_best.pt")
    print(f"Run artifacts saved to: {run_dir}")
    print(f"  step_metrics: {len(step_metrics)} steps ({epoch} epochs)")


if __name__ == "__main__":
    main()
