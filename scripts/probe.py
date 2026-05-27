"""Stage 3 evaluation entry point: linear probe + kNN on a saved checkpoint (R6, R8).

Loads encoder+head from a run directory, extracts frozen embeddings on train/val
splits, runs LinearProbe and kNN, then appends a result row to probe_results.csv.
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

os.environ.setdefault(
    "PROJECT_ROOT", str(Path(__file__).resolve().parent.parent).replace("\\", "/")
)

import hydra
import numpy as np
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from torch.utils.data import DataLoader

from src.data import get_dataset, build_eval_transform
from src.metrics import linear_probe, knn_eval
from src.utils import make_run_id, set_seed
from src.utils.param_count import count_parameters


# ChestMNIST 14-class label names in medmnist order (index 0-13).
# Used to name per-class AUROC keys so make_paper_tables.py can look
# up rare classes by name rather than integer index.
CHESTMNIST_CLASS_NAMES = [
    "atelectasis", "cardiomegaly", "effusion", "infiltration",
    "mass", "nodule", "pneumonia", "pneumothorax",
    "consolidation", "edema", "emphysema", "fibrosis",
    "pleural_thickening", "hernia",
]

_CSV_COLUMNS = [
    "run_id", "encoder", "head", "loss", "scorer", "dataset", "seed",
    "params_total", "macro_auroc_linear", "macro_auroc_knn", "mAP",
    "per_class_auroc_linear_json",
    "n_classes_valid",
    "runtime_sec",
]


def _extract_embeddings(
    encoder: nn.Module,
    head: nn.Module,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (embeddings [N, D], labels [N, C]) from a DataLoader."""
    encoder.eval()
    head.eval()
    all_emb, all_lbl = [], []

    with torch.no_grad():
        for batch in loader:
            imgs, labels, _pids = batch
            imgs = imgs.to(device)
            feats = encoder(imgs)
            z = head(feats)
            all_emb.append(z.cpu().numpy())
            all_lbl.append(labels.numpy())

    return np.concatenate(all_emb, axis=0), np.concatenate(all_lbl, axis=0)


def _append_csv_row(csv_path: Path, row: dict) -> None:
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


@hydra.main(version_base=None, config_path="../configs/experiment", config_name="probe")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.probe.seed)
    device = torch.device(cfg.probe.device)

    ckpt_dir = Path(cfg.probe.checkpoint_dir)
    if not ckpt_dir.exists():
        raise ValueError(
            f"checkpoint_dir does not exist: {ckpt_dir}. "
            "Pass probe.checkpoint_dir=<path> on the command line."
        )

    # Load model architecture from checkpoint config then restore weights.
    ckpt_cfg_path = ckpt_dir / "config.yaml"
    if not ckpt_cfg_path.exists():
        raise ValueError(f"No config.yaml found in checkpoint_dir: {ckpt_dir}")

    ckpt_cfg = OmegaConf.load(ckpt_cfg_path)
    encoder = instantiate(ckpt_cfg.model.encoder).to(device)
    head = instantiate(ckpt_cfg.model.head).to(device)

    best_path = ckpt_dir / "model_best.pt"
    fallback_path = ckpt_dir / "model.pt"
    if best_path.exists():
        state = torch.load(best_path, map_location=device)
        print(f"[probe] Loaded best-val checkpoint: {best_path}")
    elif fallback_path.exists():
        state = torch.load(fallback_path, map_location=device)
        print(f"[probe] Warning: model_best.pt not found, using model.pt (last step)")
    else:
        raise ValueError(
            f"No model checkpoint found in {ckpt_dir}. "
            "Expected model_best.pt or model.pt."
        )
    encoder.load_state_dict({k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")})
    head.load_state_dict({k[len("head."):]: v for k, v in state.items() if k.startswith("head.")})

    params_total = count_parameters(encoder) + count_parameters(head)

    # Build eval-transform loaders (single view) for both train and val.
    # Cannot reuse get_dataloader() because the train split uses TwoViewTransform.
    mean = list(cfg.data.normalize.mean)
    std = list(cfg.data.normalize.std)
    eval_tf = build_eval_transform(size=cfg.data.size, mean=mean, std=std)

    train_ds = get_dataset(cfg.data, "train", eval_tf)
    val_ds = get_dataset(cfg.data, "val", eval_tf)

    train_loader = DataLoader(train_ds, batch_size=cfg.data.batch_size,
                              shuffle=False, num_workers=cfg.data.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.data.batch_size,
                            shuffle=False, num_workers=cfg.data.num_workers)

    t0 = time.time()
    train_emb, train_lbl = _extract_embeddings(encoder, head, train_loader, device)
    val_emb, val_lbl = _extract_embeddings(encoder, head, val_loader, device)

    # Identify zero-positive classes in val. Log explicitly; do not silently drop.
    valid_classes = [c for c in range(val_lbl.shape[1]) if val_lbl[:, c].sum() > 0]
    n_classes_valid = len(valid_classes)
    if n_classes_valid < val_lbl.shape[1]:
        dropped_ids = [c for c in range(val_lbl.shape[1]) if val_lbl[:, c].sum() == 0]
        print(
            f"[probe] Warning: {len(dropped_ids)} class(es) have 0 val positives "
            f"and are excluded from AUROC (class ids: {dropped_ids}). "
            f"This is expected for fresh/random encoders."
        )

    tr_lbl_f = train_lbl[:, valid_classes].astype(np.int32)
    vl_lbl_f = val_lbl[:, valid_classes].astype(np.int32)

    probe_out = linear_probe(
        train_emb, tr_lbl_f, val_emb, vl_lbl_f,
        max_iter=cfg.probe.probe_max_iter,
        C=cfg.probe.probe_C,
    )
    knn_out = knn_eval(
        train_emb, tr_lbl_f, val_emb, vl_lbl_f,
        k=min(cfg.probe.knn_k, train_emb.shape[0]),
    )

    runtime = time.time() - t0
    run_id = make_run_id()

    names = (
        CHESTMNIST_CLASS_NAMES
        if cfg.data.name == "chestmnist"
        else [str(i) for i in range(val_lbl.shape[1])]
    )
    per_class_json = json.dumps({
        names[i]: round(float(v), 6)
        for i, v in zip(valid_classes, probe_out["per_class_auroc"])
    })

    row = {
        "run_id": run_id,
        "encoder": cfg.meta.encoder,
        "head": cfg.meta.head,
        "loss": cfg.meta.loss,
        "scorer": cfg.meta.scorer,
        "dataset": cfg.meta.dataset,
        "seed": cfg.probe.seed,
        "params_total": params_total,
        "macro_auroc_linear": round(probe_out["macro_auroc"], 6),
        "macro_auroc_knn": round(knn_out["macro_auroc"], 6),
        "mAP": round(probe_out["mAP"], 6),
        "per_class_auroc_linear_json": per_class_json,
        "n_classes_valid": n_classes_valid,
        "runtime_sec": round(runtime, 2),
    }

    csv_path = Path(cfg.probe.output_csv)
    _append_csv_row(csv_path, row)

    print(f"[probe] macro_auroc_linear={row['macro_auroc_linear']:.4f}  "
          f"macro_auroc_knn={row['macro_auroc_knn']:.4f}  "
          f"mAP={row['mAP']:.4f}  "
          f"n_classes_valid={n_classes_valid}  runtime={runtime:.1f}s")
    print(f"[probe] Row appended to {csv_path}")


if __name__ == "__main__":
    main()
