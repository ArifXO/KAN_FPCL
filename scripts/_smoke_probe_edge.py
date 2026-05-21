"""Small-subset probe for Stage 7.5 smoke runs.

The default probe materializes a full [N_val, N_train] cosine matrix that
overflows memory on this host. For smoke checkpoints (which only ran 5
steps anyway), the meaningful signal is just that the encoder forward +
probe + CSV append path runs end-to-end at random-baseline AUROC. So we
subsample to ~5k/1k train/val.

This script is not a stage deliverable — it's a smoke-helper. Real probes
should go through scripts/probe.py on a machine with enough memory.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from src.data import ChestMNISTDataset, build_eval_transform
from src.metrics import linear_probe, knn_eval
from src.utils import make_run_id, set_seed
from src.utils.param_count import count_parameters


_CSV_COLUMNS = [
    "run_id", "encoder", "head", "loss", "scorer", "dataset", "seed",
    "params_total", "macro_auroc_linear", "macro_auroc_knn", "mAP", "runtime_sec",
]


def main(ckpt_dir: str, scorer_tag: str, loss_tag: str = "edge_aware_fn") -> None:
    set_seed(42)
    device = torch.device("cpu")
    ckpt = Path(ckpt_dir)
    cfg = OmegaConf.load(ckpt / "config.yaml")

    encoder = instantiate(cfg.model.encoder).to(device)
    head = instantiate(cfg.model.head).to(device)
    state = torch.load(ckpt / "model.pt", map_location=device)
    encoder.load_state_dict({k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")})
    head.load_state_dict({k[len("head."):]: v for k, v in state.items() if k.startswith("head.")})
    params_total = count_parameters(encoder) + count_parameters(head)

    mean = list(cfg.data.normalize.mean)
    std = list(cfg.data.normalize.std)
    eval_tf = build_eval_transform(size=cfg.data.size, mean=mean, std=std)

    train_ds = ChestMNISTDataset(split="train", transform=eval_tf, download=False, size=cfg.data.size)
    val_ds = ChestMNISTDataset(split="val", transform=eval_tf, download=False, size=cfg.data.size)

    # Deterministic subset for smoke probes (memory limits).
    rng = np.random.default_rng(42)
    train_idx = rng.choice(len(train_ds), size=min(5000, len(train_ds)), replace=False)
    val_idx = rng.choice(len(val_ds), size=min(1000, len(val_ds)), replace=False)

    tr_loader = DataLoader(Subset(train_ds, train_idx.tolist()), batch_size=256, shuffle=False, num_workers=0)
    vl_loader = DataLoader(Subset(val_ds, val_idx.tolist()), batch_size=256, shuffle=False, num_workers=0)

    def _emb(loader):
        encoder.eval(); head.eval()
        embs, lbls = [], []
        with torch.no_grad():
            for imgs, labels, _pids in loader:
                imgs = imgs.to(device)
                feats = encoder(imgs)
                # Head may return (z, phi) tuple if return_edges=True is forced;
                # call without that flag to keep eval path simple.
                z = head(feats)
                if isinstance(z, tuple):
                    z = z[0]
                embs.append(z.cpu().numpy())
                lbls.append(labels.numpy())
        return np.concatenate(embs), np.concatenate(lbls)

    t0 = time.time()
    train_emb, train_lbl = _emb(tr_loader)
    val_emb, val_lbl = _emb(vl_loader)

    valid_c = [c for c in range(val_lbl.shape[1]) if val_lbl[:, c].sum() > 0]
    tr_lbl_f = train_lbl[:, valid_c].astype(np.int32)
    vl_lbl_f = val_lbl[:, valid_c].astype(np.int32)

    pout = linear_probe(train_emb, tr_lbl_f, val_emb, vl_lbl_f, max_iter=1000, C=1.0)
    kout = knn_eval(train_emb, tr_lbl_f, val_emb, vl_lbl_f, k=20)
    runtime = time.time() - t0

    row = {
        "run_id": f"smoke_{make_run_id()}",
        "encoder": "resnet18", "head": "fastkan",
        "loss": loss_tag, "scorer": scorer_tag,
        "dataset": "chestmnist", "seed": 42,
        "params_total": params_total,
        "macro_auroc_linear": round(pout["macro_auroc"], 6),
        "macro_auroc_knn": round(kout["macro_auroc"], 6),
        "mAP": round(pout["mAP"], 6),
        "runtime_sec": round(runtime, 2),
    }

    csv_path = Path("probe_results.csv")
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"Appended smoke probe row: {row}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python -m scripts._smoke_probe_edge <ckpt_dir> <scorer_tag>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
