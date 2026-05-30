"""Post-hoc saturation diagnosis for an FN-weighted training run.

Reads ``step_metrics.csv`` (or ``.json``) from a run directory and reports:
* p_fn raw + clipped mean/std over training,
* effective negative weight curve,
* p_fn_at_cap_fraction trajectory,
* a histogram of the final p_fn distribution if the run logged one (we do
  not yet — the histogram block is a placeholder for future telemetry),
* warnings on sustained saturation or collapse-to-constant.

Usage::

    python -m scripts.analysis.analyze_pfn_saturation \\
        --run-dir runs/checkpoints/smoke_fn_mlp_<id> [--plot]

The script is intentionally lightweight (no Hydra dep) so it can be run on a
checkpoint dir copied off the cluster.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path


def _read_step_metrics(run_dir: Path) -> list[dict]:
    csv_path = run_dir / "step_metrics.csv"
    json_path = run_dir / "step_metrics.json"
    if csv_path.exists():
        with open(csv_path, "r", newline="") as fh:
            return [
                {k: (float(v) if v not in ("", "nan", "NaN") else float("nan"))
                 for k, v in row.items()}
                for row in csv.DictReader(fh)
            ]
    if json_path.exists():
        return json.loads(json_path.read_text())
    raise FileNotFoundError(
        f"Neither step_metrics.csv nor step_metrics.json found under {run_dir}."
    )


def _series(rows: list[dict], key: str) -> list[float]:
    return [float(r.get(key, float("nan"))) for r in rows]


def _mean(xs: list[float]) -> float:
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def _last_n_mean(xs: list[float], n: int) -> float:
    return _mean(xs[-n:])


def _epochs_above(xs: list[float], threshold: float, by_epoch: list[int]) -> set[int]:
    epochs = set()
    for x, e in zip(xs, by_epoch):
        if not math.isnan(x) and x > threshold:
            epochs.add(int(e))
    return epochs


def report(run_dir: Path) -> int:
    rows = _read_step_metrics(run_dir)
    if not rows:
        print(f"[WARN] no step_metrics rows in {run_dir}")
        return 1

    epochs = [int(r.get("epoch", 0)) for r in rows]
    cap_frac = _series(rows, "p_fn_at_cap_fraction")
    raw_mean = _series(rows, "p_fn_raw_mean")
    raw_std = _series(rows, "p_fn_raw_std")
    eff_neg_mean = _series(rows, "effective_neg_weight_mean")
    pfn_reg = _series(rows, "pfn_reg_total")
    max_cap = _series(rows, "max_fn_weight_current")

    print(f"Run: {run_dir}")
    print(f"  steps: {len(rows)} (final epoch={epochs[-1]})")
    print()
    print("Last-100-step means (or run mean if shorter):")
    print(f"  p_fn_at_cap_fraction       = {_last_n_mean(cap_frac, 100):.4f}")
    print(f"  p_fn_raw_mean              = {_last_n_mean(raw_mean, 100):.4f}")
    print(f"  p_fn_raw_std               = {_last_n_mean(raw_std, 100):.4f}")
    print(f"  effective_neg_weight_mean  = {_last_n_mean(eff_neg_mean, 100):.4f}")
    print(f"  pfn_reg_total              = {_last_n_mean(pfn_reg, 100):.4f}")
    print(f"  max_fn_weight_current      = {_last_n_mean(max_cap, 100):.4f}")
    print()

    warnings: list[str] = []
    # Sustained saturation: at_cap > 0.25 in ≥ 2 epochs (after warmup).
    bad_epochs = _epochs_above(cap_frac, 0.25, epochs)
    if len(bad_epochs) >= 2:
        warnings.append(
            f"p_fn_at_cap_fraction > 0.25 in {len(bad_epochs)} epoch(s): "
            f"{sorted(bad_epochs)[:8]}{'...' if len(bad_epochs) > 8 else ''}. "
            "Scorer is saturating against the cap — consider higher lambda_cap "
            "or a slower ramp."
        )

    # Collapse-to-constant: tail std < 0.02 while mean is high.
    tail_std = _last_n_mean(raw_std, 100)
    tail_mean = _last_n_mean(raw_mean, 100)
    if not math.isnan(tail_std) and tail_std < 0.02 and tail_mean > 0.3:
        warnings.append(
            f"p_fn_raw_std={tail_std:.4f} with raw_mean={tail_mean:.4f}: scorer "
            "appears collapsed to a near-constant high value. Increase "
            "lambda_mean or lower target_mean."
        )

    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("No saturation warnings.")
    print()

    # Optional plot: only attempt if matplotlib is available.
    if "--plot" in sys.argv:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[plot] matplotlib not available; skipping.")
            return 0
        fig, axs = plt.subplots(2, 2, figsize=(10, 7))
        axs[0, 0].plot(raw_mean, label="raw_mean")
        axs[0, 0].plot(eff_neg_mean, label="eff_neg_mean")
        axs[0, 0].set_title("p_fn raw mean / effective neg weight")
        axs[0, 0].legend()
        axs[0, 1].plot(cap_frac, color="red")
        axs[0, 1].axhline(0.25, color="grey", linestyle="--")
        axs[0, 1].set_title("p_fn_at_cap_fraction")
        axs[1, 0].plot(raw_std)
        axs[1, 0].set_title("p_fn_raw_std")
        axs[1, 1].plot(max_cap)
        axs[1, 1].set_title("max_fn_weight_current (schedule)")
        for ax in axs.flat:
            ax.set_xlabel("step")
        fig.suptitle(f"p_fn saturation diagnostics — {run_dir.name}")
        fig.tight_layout()
        out = run_dir / "pfn_saturation.png"
        fig.savefig(out, dpi=120)
        print(f"[plot] wrote {out}")

    return 0 if not warnings else 2


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument(
        "--plot", action="store_true",
        help="Write pfn_saturation.png alongside the metrics CSV.",
    )
    args = p.parse_args(argv)
    return report(args.run_dir)


if __name__ == "__main__":
    sys.exit(main())
