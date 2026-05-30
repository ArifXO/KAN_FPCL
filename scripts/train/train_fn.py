"""Stage 6 training entry point (Hydra, R6/R8): FN-weighted InfoNCE + scorer.

Trains encoder + projection head + MLP pair scorer jointly with
:class:`src.losses.FNWeightedInfoNCELoss`. The pair scorer's ``p_fn`` matrix
is fed into the loss denominator to downweight likely false negatives.

Adds a cosine LR schedule, gradient clipping, per-step metric logging
(``step_metrics.json``), and validation-loss monitoring with a
``model_best.pt`` checkpoint. Saves the R8 artifact bundle to
``<output_root>/run_<name>_<run_id>/``; the encoder+head sub-state is keyed
so :mod:`scripts.analysis.probe` can load them while ignoring scorer weights.
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
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.data import get_dataloader
from src.utils import make_run_id, save_run_artifacts, set_seed

from train_common import (
    base_step_metrics,
    build_val_loader,
    compute_max_fn_weight_current,
    run_validation_dict,
    snapshot_cpu_state,
)


@hydra.main(version_base=None, config_path="../../configs/experiment", config_name="smoke_fn_mlp")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.run.seed, deterministic=cfg.run.get("deterministic", False))
    device = torch.device(cfg.run.device)

    encoder = instantiate(cfg.model.encoder).to(device)
    head = instantiate(cfg.model.head).to(device)
    scorer = instantiate(cfg.model.scorer).to(device)
    # Pop fn_schedule (train-loop concern) and name (probe-row metadata) so
    # neither is passed to FNWeightedInfoNCELoss.__init__ as an unexpected kwarg.
    loss_cfg = OmegaConf.to_container(cfg.loss, resolve=True)
    loss_cfg.pop("fn_schedule", None)
    loss_cfg.pop("name", None)
    loss_target = loss_cfg.pop("_target_")
    loss_fn = hydra.utils.get_class(loss_target)(**loss_cfg).to(device)
    full_model = nn.ModuleDict({"encoder": encoder, "head": head, "scorer": scorer})

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
    lambda_pfn_reg = cfg.train.get("lambda_pfn_reg", 0.0)  # legacy term
    static_max_fn_weight = float(cfg.loss.get("max_fn_weight", 1.0))
    fn_schedule_cfg = (
        OmegaConf.to_container(cfg.loss.fn_schedule, resolve=True)
        if "fn_schedule" in cfg.loss else None
    )

    current_max_cap = {"val": static_max_fn_weight}  # closure-shared, updated per epoch

    def val_forward(v1: torch.Tensor, v2: torch.Tensor) -> dict:
        # val_total = contrastive loss + pfn_reg_total + legacy collapse
        # penalty, so a saturated scorer cannot win checkpoint selection by
        # gaming the FN denominator.
        z = head(encoder(torch.cat([v1, v2], dim=0)))
        p_fn = scorer(z[: v1.shape[0]])
        fn_out = loss_fn(z, p_fn, max_fn_weight_override=current_max_cap["val"])
        legacy_reg = lambda_pfn_reg * p_fn.mean()
        return {
            "loss": fn_out["loss"],
            "val_total": fn_out["loss"] + fn_out["pfn_reg_total"] + legacy_reg,
            "p_fn_at_cap_fraction": fn_out["p_fn_at_cap_fraction"],
        }

    losses: list[float] = []
    step_metrics: list[dict] = []
    val_loss_curve: list[dict] = []
    best_val_loss = float("inf")
    best_val_total = float("inf")
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
        # Schedule update — recomputed once per epoch. None falls through to
        # the loss's constructor-time max_fn_weight (legacy behavior).
        sched_cap = compute_max_fn_weight_current(
            epoch, fn_schedule_cfg, static_max_fn_weight
        )
        current_max_cap["val"] = (
            sched_cap if sched_cap is not None else static_max_fn_weight
        )
        for (v1, v2), _labels, _pids in train_loader:
            if step >= cfg.train.max_steps:
                break
            v1, v2 = v1.to(device), v2.to(device)
            z = head(encoder(torch.cat([v1, v2], dim=0)))

            batch = v1.shape[0]
            # Scorer detaches internally now (detach_inputs=True default for
            # MLPPairScorer/KANPairScorer); .detach() here is redundant but
            # cheap and keeps the call signature explicit for readers.
            p_fn = scorer(z[:batch])

            out = loss_fn(z, p_fn, max_fn_weight_override=sched_cap)
            contrastive_loss = out["loss"]
            pfn_reg_total = out["pfn_reg_total"]
            legacy_reg = lambda_pfn_reg * p_fn.mean()
            total_loss = contrastive_loss + pfn_reg_total + legacy_reg

            optim.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            optim.step()
            scheduler.step()

            losses.append(float(contrastive_loss.item()))
            step_dict = base_step_metrics(step, scheduler.get_last_lr()[0], out, epoch=epoch)
            step_dict["p_fn_reg_loss"] = float(legacy_reg.detach())
            step_dict["total_loss"] = float(total_loss.detach())
            step_metrics.append(step_dict)

            if step % cfg.train.log_every == 0:
                print(
                    f"[step {step}] loss={total_loss.item():.4f} "
                    f"raw_mean={out['p_fn_raw_mean'].item():.4f} "
                    f"raw_max={out['p_fn_raw_max'].item():.4f} "
                    f"raw_std={out['p_fn_raw_std'].item():.4f} "
                    f"at_cap={out['p_fn_at_cap_fraction'].item():.4f} "
                    f"max_cap={out['max_fn_weight_current'].item():.3f} "
                    f"pfn_reg={out['pfn_reg_total'].detach().item():.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

            if step % val_every == 0:
                encoder.eval()
                head.eval()
                scorer.eval()
                val_out = run_validation_dict(
                    val_loader, device, val_forward, val_max_batches, val_seed,
                    keys=("loss", "val_total", "p_fn_at_cap_fraction"),
                )
                encoder.train()
                head.train()
                scorer.train()
                val_loss = val_out["loss"]
                val_total = val_out["val_total"]
                val_loss_curve.append(
                    {"step": step, "val_loss": val_loss, "val_total": val_total}
                )
                if val_total < best_val_total:
                    best_val_total = val_total
                    best_val_loss = val_loss
                    best_val_step = step
                    best_state = snapshot_cpu_state(full_model)
                cap_frac = val_out.get("p_fn_at_cap_fraction", 0.0)
                if cap_frac > 0.5:
                    print(
                        f"[WARNING step {step}] p_fn saturation: {cap_frac:.1%} of pairs at "
                        f"max_fn_weight cap. Scorer may be collapsing."
                    )
                print(
                    f"[val step {step}] val_loss={val_loss:.4f} "
                    f"val_total={val_total:.4f} best={best_val_total:.4f}"
                )

            step += 1

    runtime = time.time() - t0

    run_id = make_run_id()
    run_dir = Path(cfg.run.output_root) / f"{cfg.run.name}_{run_id}"

    if best_state is None:
        best_state = snapshot_cpu_state(full_model)
        best_val_loss = float("nan")
        best_val_total = float("nan")

    metrics = {
        "train_loss_final": losses[-1] if losses else float("nan"),
        "train_loss_curve": losses,
        "train_loss_type": "contrastive",
        "val_loss_curve": val_loss_curve,
        "best_val_loss": best_val_loss,
        "best_val_total": best_val_total,
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
