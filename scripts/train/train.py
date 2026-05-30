"""Stage 2 training entry point (Hydra, R6/R8).

Runs a contrastive training loop using the resolved Hydra config and saves the
R8 artifact set to ``<output_root>/run_<run_id>/``. Adds a cosine LR schedule,
gradient clipping, per-step metric logging (``step_metrics.json``), and
validation-loss monitoring with a ``model_best.pt`` checkpoint.

Routes by ``cfg.loss.name`` (metadata only — never passed as a constructor
kwarg):
* ``infonce`` / ``debiased`` (or no name): label-free path, ``loss_fn(z)``.
* ``supcon_oracle``: label-aware path, ``loss_fn(z, labels)``. Labels are
  the original ``[B, C]`` multi-hot training labels — never duplicated.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault(
    "PROJECT_ROOT", str(Path(__file__).resolve().parents[2]).replace("\\", "/")
)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from src.data import get_dataloader
from src.utils import make_run_id, save_run_artifacts, set_seed

from train_common import (
    base_step_metrics,
    build_val_loader,
    snapshot_cpu_state,
)


@hydra.main(version_base=None, config_path="../../configs/experiment", config_name="smoke_mlp")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.run.seed, deterministic=cfg.run.get("deterministic", False))
    device = torch.device(cfg.run.device)

    encoder = hydra.utils.instantiate(cfg.model.encoder).to(device)
    head = hydra.utils.instantiate(cfg.model.head).to(device)
    # Pop metadata-only fields (name) before constructing the loss — these
    # drive routing and probe-row tagging but are not constructor kwargs.
    loss_cfg = OmegaConf.to_container(cfg.loss, resolve=True)
    loss_name = loss_cfg.pop("name", "infonce")
    loss_target = loss_cfg.pop("_target_")
    loss_fn = hydra.utils.get_class(loss_target)(**loss_cfg).to(device)
    requires_labels = loss_name == "supcon_oracle"
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

    def _run_val(epoch_for_seed_only: int) -> float:
        """Mean ``loss`` over (optionally capped) val batches.

        Inlined (not :func:`train_common.run_validation`) because SupCon
        Oracle needs per-batch labels and the shared helper's forward_fn
        signature is ``(v1, v2)`` only.
        """
        rng_state = torch.get_rng_state()
        torch.manual_seed(val_seed)
        total, count = 0.0, 0
        try:
            with torch.no_grad():
                for (v1, v2), lbls, _pids in val_loader:
                    if val_max_batches is not None and count >= val_max_batches:
                        break
                    v1, v2 = v1.to(device), v2.to(device)
                    z = head(encoder(torch.cat([v1, v2], dim=0)))
                    if requires_labels:
                        out = loss_fn(z, lbls.to(device))
                    else:
                        out = loss_fn(z)
                    total += float(out["loss"])
                    count += 1
        finally:
            torch.set_rng_state(rng_state)
        if count == 0:
            raise RuntimeError(
                "Validation produced 0 batches. val split too small for "
                f"batch_size={val_loader.batch_size} with drop_last=True, or "
                f"val_max_batches misconfigured (max_batches={val_max_batches})."
            )
        return total / count

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
        for (v1, v2), labels, _pids in train_loader:
            if step >= cfg.train.max_steps:
                break
            v1, v2 = v1.to(device), v2.to(device)
            z = head(encoder(torch.cat([v1, v2], dim=0)))
            if requires_labels:
                out = loss_fn(z, labels.to(device))
            else:
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
                val_loss = _run_val(epoch)
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
    run_dir = Path(cfg.run.output_root) / f"{cfg.run.name}_{run_id}"

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
