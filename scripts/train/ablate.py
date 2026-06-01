"""Cell × seed ablation orchestrator (Stage 10, R6/R8/R9).

For each (cell, seed) in ``configs/experiment/ablation.yaml``:
    train -> probe -> analyze_geometry
then append one merged row to ``ablation_master.csv``.

A failure in any sub-step is captured as ``status=FAILED: <exception>`` so
the next cell still runs — one OOM must not lose the rest of the matrix.

Sub-scripts are invoked as subprocesses (``python -m scripts.<group>.<name>``) and
re-use their existing Hydra configs; this script only orchestrates and
merges the per-step CSV rows.
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault(
    "PROJECT_ROOT", str(Path(__file__).resolve().parents[2]).replace("\\", "/")
)

import hydra
from omegaconf import DictConfig

_COLUMNS = [
    "cell_id", "head", "loss", "scorer", "lambda_edge", "lambda_edge_align",
    "dataset", "seed", "params_total", "params_scorer", "final_loss", "best_loss",
    "macro_auroc_linear", "macro_auroc_knn",
    "mAP", "per_class_auroc_linear_json",
    # Test-split metrics: model_best.pt is selected on val, so val numbers are
    # optimistically biased. Carry the test columns probe.py emits so downstream
    # tables can report the unbiased headline split.
    "macro_auroc_linear_test", "macro_auroc_knn_test", "mAP_test",
    "per_class_auroc_linear_test_json",
    "alignment", "uniformity", "effective_rank",
    "runtime_sec", "status",
]


def _append_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Replace any prior master row sharing the (cell_id, seed, dataset) identity
    # so re-running a cell overwrites rather than appending a duplicate.
    key = (str(row.get("cell_id")), str(row.get("seed")), str(row.get("dataset")))
    kept: list[dict] = []
    if csv_path.exists():
        with open(csv_path, "r", newline="") as f:
            for r in csv.DictReader(f):
                r_key = (str(r.get("cell_id")), str(r.get("seed")), str(r.get("dataset")))
                if r_key != key:
                    kept.append({k: r.get(k, "") for k in _COLUMNS})
    kept.append({k: row.get(k, "") for k in _COLUMNS})
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_COLUMNS)
        w.writeheader()
        w.writerows(kept)


def _run(cmd: list[str], cwd: Path) -> str:
    print(f"[ablate] running: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        output.append(line)
    returncode = proc.wait()
    stdout = "".join(output)
    if returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {returncode}): {' '.join(cmd)}\n"
            f"--- output ---\n{stdout}"
        )
    return stdout


def _find_run_dir(stdout: str) -> Path:
    m = re.search(r"Run artifacts saved to:\s*(.+)", stdout)
    if m is None:
        raise RuntimeError(
            "Could not parse 'Run artifacts saved to:' line from train stdout. "
            "Ensure the chosen train_script prints that sentinel."
        )
    return Path(m.group(1).strip())


def _last_csv_row(csv_path: Path) -> dict[str, str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected CSV not produced: {csv_path}")
    with open(csv_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"CSV {csv_path} has a header but no data rows.")
    return rows[-1]


def _loss_metrics(ckpt_dir: Path) -> dict[str, object]:
    metrics_path = ckpt_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Expected train metrics not produced: {metrics_path}")
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    return {
        "final_loss": metrics.get("train_loss_final", ""),
        "best_loss": metrics.get("best_val_loss", ""),
    }


_HYPOTHESIS_DIR = {
    "h1": "h1_architecture",
    "h2": "h2_fn_weighted",
    "h3": "h3_kan_scorer",
    "h3_cross": "h3_kan_scorer",
    "h4": "h4_edge_signal",
    "baselines": "baselines",
}


def _train(
    cell: DictConfig,
    seed: int,
    output_root: str,
    project_root: Path,
    global_overrides: list[str] | None = None,
) -> Path:
    overrides = list(cell.get("overrides", []))
    if global_overrides:
        overrides.extend(global_overrides)
    cmd = [
        sys.executable, "-m", str(cell.train_script),
        f"--config-name={cell.train_config}",
        f"run.seed={seed}",
        f"run.name={cell.cell_id}_s{seed}",
        f"run.output_root={output_root}",
        *overrides,
    ]
    return _find_run_dir(_run(cmd, project_root))


def _probe(
    ckpt_dir: Path, cell: DictConfig, seed: int, csv_path: Path, project_root: Path
) -> dict[str, str]:
    cmd = [
        sys.executable, "-m", "scripts.analysis.probe",
        f"probe.checkpoint_dir={ckpt_dir}",
        f"probe.seed={seed}",
        f"probe.output_csv={csv_path}",
        f"meta.head={cell.head}",
        f"meta.loss={cell.loss}",
        f"meta.scorer={cell.scorer}",
        f"meta.dataset={cell.dataset}",
    ]
    _run(cmd, project_root)
    return _last_csv_row(csv_path)


def _geometry(
    ckpt_dir: Path, cell: DictConfig, seed: int, csv_path: Path, project_root: Path
) -> dict[str, str]:
    cmd = [
        sys.executable, "-m", "scripts.analysis.analyze_geometry",
        f"geometry.checkpoint_dir={ckpt_dir}",
        f"geometry.seed={seed}",
        f"geometry.output_csv={csv_path}",
        f"meta.head={cell.head}",
        f"meta.loss={cell.loss}",
        f"meta.dataset={cell.dataset}",
    ]
    _run(cmd, project_root)
    return _last_csv_row(csv_path)


def _run_cell(
    cell: DictConfig,
    seed: int,
    out_csv: Path,
    probe_csv: Path,
    geom_csv: Path,
    project_root: Path,
    global_overrides: list[str] | None = None,
) -> None:
    row = {
        "cell_id": cell.cell_id,
        "head": cell.head,
        "loss": cell.loss,
        "scorer": cell.scorer,
        "lambda_edge": cell.get("lambda_edge", ""),
        "lambda_edge_align": cell.get("lambda_edge_align", ""),
        "dataset": cell.dataset,
        "seed": seed,
        "status": "FAILED",
    }
    if cell.cell_id.startswith("smoke_"):
        output_root = "runs/smoke"
    else:
        hyp = cell.get("hypothesis", "h1")
        output_root = f"runs/ablation/{_HYPOTHESIS_DIR[hyp]}"

    t0 = time.time()
    try:
        ckpt_dir = _train(cell, seed, output_root, project_root, global_overrides)
        loss_row = _loss_metrics(ckpt_dir)
        probe_row = _probe(ckpt_dir, cell, seed, probe_csv, project_root)
        geom_row = _geometry(ckpt_dir, cell, seed, geom_csv, project_root)
        row.update({
            "params_total": probe_row.get("params_total", ""),
            "params_scorer": probe_row.get("params_scorer", "0"),
            "final_loss": loss_row.get("final_loss", ""),
            "best_loss": loss_row.get("best_loss", ""),
            "macro_auroc_linear": probe_row.get("macro_auroc_linear", ""),
            "macro_auroc_knn": probe_row.get("macro_auroc_knn", ""),
            "mAP": probe_row.get("mAP", ""),
            "per_class_auroc_linear_json": probe_row.get("per_class_auroc_linear_json", ""),
            "macro_auroc_linear_test": probe_row.get("macro_auroc_linear_test", ""),
            "macro_auroc_knn_test": probe_row.get("macro_auroc_knn_test", ""),
            "mAP_test": probe_row.get("mAP_test", ""),
            "per_class_auroc_linear_test_json": probe_row.get("per_class_auroc_linear_test_json", ""),
            "alignment": geom_row.get("alignment", ""),
            "uniformity": geom_row.get("uniformity", ""),
            "effective_rank": geom_row.get("effective_rank", ""),
            "status": "OK",
        })
    except Exception as e:
        msg = f"{type(e).__name__}: {e}".replace("\n", " | ").replace("\r", " ")
        row["status"] = f"FAILED: {msg}"[:512]
        traceback.print_exc()
    finally:
        row["runtime_sec"] = round(time.time() - t0, 2)
        _append_row(out_csv, row)
        print(
            f"[ablate] cell={cell.cell_id} seed={seed} "
            f"status={row['status'][:40]} runtime={row['runtime_sec']}s"
        )


@hydra.main(
    version_base=None, config_path="../../configs/experiment", config_name="ablation"
)
def main(cfg: DictConfig) -> None:
    if not cfg.ablate.cells:
        raise ValueError("ablate.cells is empty — nothing to run.")
    seeds = list(cfg.ablate.seeds)
    if not seeds:
        raise ValueError("ablate.seeds is empty — at least one seed required.")

    project_root = Path(os.environ["PROJECT_ROOT"])
    # Anchor the master CSV to the project root (like probe/geom below) so it
    # lands with the other artifacts regardless of the launch directory.
    out_csv = Path(cfg.ablate.output_csv)
    if not out_csv.is_absolute():
        out_csv = project_root / out_csv
    probe_csv = project_root / cfg.ablate.probe_csv
    geom_csv = project_root / cfg.ablate.geometry_csv
    global_overrides = [str(item) for item in cfg.ablate.get("global_overrides", [])]

    n = sum(1 for _ in cfg.ablate.cells) * len(seeds)
    print(f"[ablate] {n} (cell x seed) combinations -> {out_csv}")
    for cell in cfg.ablate.cells:
        for seed in seeds:
            _run_cell(
                cell,
                int(seed),
                out_csv,
                probe_csv,
                geom_csv,
                project_root,
                global_overrides,
            )


if __name__ == "__main__":
    main()
