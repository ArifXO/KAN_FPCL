"""Generate H1-H4 Markdown tables from ``ablation_master.csv``.

The ablation CSV is expected to come from ``scripts/train/ablate.py``. Failed rows
and smoke rows are excluded by default so thesis tables only average completed
full-run cells.

Headline metrics default to the **test** split: ``model_best.pt`` is selected on
validation loss, so val AUROC/mAP are optimistically biased. Pass ``--split val``
to reproduce the (biased) validation numbers. ``--publish`` additionally copies
the generated tables into ``reports/tables/`` for git commit.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np

import pandas as pd

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))

# ChestMNIST: rarest 3 classes by frequency (hernia ~0.2%, emphysema ~2%, fibrosis ~1.5%)
RARE_CLASSES = ("hernia", "emphysema", "fibrosis")

CORE_COLUMNS = {
    "cell_id",
    "head",
    "loss",
    "scorer",
    "seed",
    "params_total",
}

# H1/H2/H3/H4: minimum seeds needed to count as "enough data"
_MIN_SEEDS = 3


def _metric_aliases(split: str) -> dict[str, tuple[str, ...]]:
    """Canonical metric name -> source-column candidates, in preference order.

    For ``split='test'`` the ``*_test`` columns are tried first and fall back to
    the val columns so a run that pre-dates the test columns still tabulates.
    """
    if split == "test":
        return {
            "macro_auroc": ("macro_auroc_linear_test", "macro_auroc_linear", "macro_auroc"),
            "macro_auroc_knn": ("macro_auroc_knn_test", "macro_auroc_knn"),
            "mAP": ("mAP_test", "mAP", "map"),
            "rare_disease_auroc": ("rare_disease_auroc", "rare_auroc"),
        }
    return {
        "macro_auroc": ("macro_auroc_linear", "macro_auroc"),
        "macro_auroc_knn": ("macro_auroc_knn",),
        "mAP": ("mAP", "map", "mean_average_precision"),
        "rare_disease_auroc": ("rare_disease_auroc", "rare_disease_auroc_linear", "rare_auroc"),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="runs/results/ablation_master.csv",
                   help="Ablation CSV path.")
    p.add_argument("--output-dir", default="runs/tables", help="Directory for tables.")
    p.add_argument("--split", choices=["test", "val"], default="test",
                   help="Metric split for headline numbers (default: test).")
    p.add_argument("--publish", action="store_true",
                   help="Also copy generated tables into reports/tables/ for commit.")
    p.add_argument("--publish-dir", default="reports/tables",
                   help="Destination for --publish (default: reports/tables).")
    p.add_argument("--include-smoke", action="store_true", help="Keep smoke_* rows.")
    p.add_argument("--include-failed", action="store_true", help="Keep FAILED rows.")
    return p.parse_args()


def _read_ablation(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Expected ablation CSV at {path}.\n"
            f"Run scripts/train/ablate.py first, or pass --input <path> to point to a different file."
        )
    df = pd.read_csv(path)
    if len(df) == 0:
        raise ValueError(
            f"{path} exists but has no data rows.\n"
            f"Run scripts/train/ablate.py first to populate it."
        )
    missing = sorted(CORE_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def _resolve_split(df: pd.DataFrame, requested: str) -> str:
    """Return the split actually usable, downgrading test->val with a warning."""
    if requested == "val":
        return "val"
    has_test = (
        "macro_auroc_linear_test" in df.columns
        and pd.to_numeric(df["macro_auroc_linear_test"], errors="coerce").notna().any()
    )
    if has_test:
        return "test"
    print(
        "[make_paper_tables WARNING] --split test requested but no test metrics "
        "found in the CSV (pre-dates ablate.py test columns). Falling back to "
        "VALIDATION metrics - these are optimistically biased."
    )
    return "val"


def _per_class_cols(df: pd.DataFrame, split: str) -> tuple[str, ...]:
    """Per-class AUROC JSON columns in preference order (test first, then val).

    Returns every candidate present so rare-AUROC extraction can fall back
    row-by-row, matching the per-row coalescing in ``_metric_column``.
    """
    candidates = []
    if split == "test":
        candidates.append("per_class_auroc_linear_test_json")
    candidates.append("per_class_auroc_linear_json")
    return tuple(c for c in candidates if c in df.columns)


def _metric_column(df: pd.DataFrame, canonical: str, aliases: dict[str, tuple[str, ...]]) -> str:
    # Coalesce per-row across alias columns in preference order: take the first
    # alias's value for each row, filling blanks from later aliases. This makes
    # the test->val fallback row-level, so in a mixed CSV a row that lacks a
    # test value still reports its val number instead of collapsing to NA.
    series = None
    for name in aliases[canonical]:
        if name in df.columns:
            col = pd.to_numeric(df[name], errors="coerce")
            series = col if series is None else series.fillna(col)
    df[canonical] = series if series is not None else math.nan
    return canonical


def _prepare(
    df: pd.DataFrame,
    include_smoke: bool,
    include_failed: bool,
    aliases: dict[str, tuple[str, ...]],
) -> pd.DataFrame:
    out = df.copy()
    if "status" in out.columns and not include_failed:
        out = out[out["status"].fillna("OK").eq("OK")]
    dups = out.duplicated(subset=["cell_id", "dataset", "seed"], keep=False)
    if dups.any():
        print(
            f"[make_paper_tables WARNING] {dups.sum()} duplicate (cell_id, dataset, seed) "
            f"rows. Keeping last. Unique duplicated combos:\n"
            f"{out.loc[dups, ['cell_id', 'dataset', 'seed']].drop_duplicates().to_string()}"
        )
        out = out.drop_duplicates(subset=["cell_id", "dataset", "seed"], keep="last")
    if not include_smoke:
        cell = out["cell_id"].astype(str).str.lower()
        out = out[~cell.str.startswith("smoke")]
    for col in (
        "params_total",
        "alignment",
        "uniformity",
        "effective_rank",
        "lambda_edge",
        "lambda_edge_align",
    ):
        if col not in out.columns:
            out[col] = 0.0 if col.startswith("lambda_") else math.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for metric in aliases:
        _metric_column(out, metric, aliases)
    return out.reset_index(drop=True)


def _fmt_metric(values: pd.Series, digits: int = 4) -> str:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return "NA"
    mean = vals.mean()
    # n=1: no variance to report. Tag as (n=1) so the absence of error bars is
    # explicit rather than printing "± 0.0000", which reads as zero variance.
    if len(vals) == 1:
        return f"{mean:.{digits}f} (n=1)"
    std = vals.std(ddof=1)
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _fmt_params(values: pd.Series) -> str:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return "NA"
    if vals.nunique() == 1:
        return f"{int(vals.iloc[0]):,}"
    return _fmt_metric(vals, digits=0)


def _seed_count(values: pd.Series) -> int:
    return int(values.dropna().nunique())


def _markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        display = df.astype({c: "object" for c, t in df.dtypes.items() if str(t) == "category"}).fillna("NA").astype(str)
        cols = list(display.columns)
        rows = display.to_dict("records")
        widths = {
            col: max(len(col), *(len(row[col]) for row in rows)) if rows else len(col)
            for col in cols
        }
        header = "| " + " | ".join(col.ljust(widths[col]) for col in cols) + " |"
        sep = "| " + " | ".join("-" * widths[col] for col in cols) + " |"
        body = [
            "| " + " | ".join(row[col].ljust(widths[col]) for col in cols) + " |"
            for row in rows
        ]
        return "\n".join([header, sep, *body])


def _write_table(path: Path, title: str, table: pd.DataFrame) -> None:
    text = f"# {title}\n\n{_markdown(table)}\n"
    path.write_text(text, encoding="utf-8")


def _extract_rare_auroc(json_str: str) -> float:
    """Mean AUROC across rare classes from a per-class AUROC JSON string."""
    if not json_str or pd.isna(json_str):
        return float("nan")
    try:
        data = json.loads(json_str)
        vals = [float(data[c]) for c in RARE_CLASSES if c in data]
        return float(np.mean(vals)) if vals else float("nan")
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return float("nan")


def _build_tables(df: pd.DataFrame, per_class_cols: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    # Each hypothesis isolates EXACTLY ONE experimental factor:
    # H1: head architecture     (fixed: InfoNCE loss, no scorer)
    # H2: loss function         (fixed: MLP head)
    # H3: scorer architecture   (fixed: MLP head, FN-weighted loss)
    # H4: edge feature usage    (fixed: FastKAN head, edge-aware loss)
    #
    # Rows are selected by explicit cell_id so each table varies one factor
    # while holding the rest constant. A cell absent from the CSV is simply
    # omitted (a partial ablation yields a partial table); the per-hypothesis
    # coverage summary reports which required cells are still missing.
    tables: dict[str, pd.DataFrame] = {}

    def _auroc(values: pd.Series) -> str:
        return _fmt_metric(values, digits=4)

    def _rank(values: pd.Series) -> str:
        return _fmt_metric(values, digits=2)

    def _cell_slice(cells: list[str]) -> pd.DataFrame:
        sub = df[df["cell_id"].astype(str).isin(cells)].copy()
        if not sub.empty:
            sub["cell_id"] = pd.Categorical(
                sub["cell_id"].astype(str), categories=cells, ordered=True
            )
        return sub

    def _rare_source(sub: pd.DataFrame) -> str:
        present = [c for c in per_class_cols if c in sub.columns]
        if present:
            # Coalesce per row across the candidate JSON columns (test then val)
            # so a row lacking a test fingerprint still reports its val rare AUROC.
            rare = None
            for col in present:
                vals = sub[col].apply(_extract_rare_auroc)
                rare = vals if rare is None else rare.fillna(vals)
            sub["rare_auroc"] = rare
            return "rare_auroc"
        return "rare_disease_auroc"

    # H1: head architecture — InfoNCE rows only, no scorer.
    # kan_wide_infonce is a labeled parameter ablation (hidden>=output, NOT
    # R1-matched); it appears as an extra context row to disentangle the
    # narrow-hidden bottleneck from the KAN architecture, and must not be read
    # as a parity KAN-vs-MLP comparison.
    h1 = _cell_slice(["mlp_infonce", "kan_infonce", "kan_wide_infonce", "reskan_infonce"])
    if not h1.empty:
        agg = {
            "head": ("head", "first"),
            "params_total": ("params_total", _fmt_params),
            "n_seeds": ("seed", "nunique"),
            "macro_auroc": ("macro_auroc", _auroc),
            "alignment": ("alignment", _auroc),
            "uniformity": ("uniformity", _auroc),
            "effective_rank": ("effective_rank", _rank),
        }
        if "macro_auroc_knn" in h1.columns:
            agg["macro_auroc_knn"] = ("macro_auroc_knn", _auroc)
        tables["table_h1.md"] = (
            h1.groupby("cell_id", observed=True).agg(**agg).reset_index()
        )

    # H2: loss function — MLP head rows only (InfoNCE vs FN-weighted).
    h2 = _cell_slice(["mlp_infonce", "mlp_fn_mlp"])
    if not h2.empty:
        rare_src = _rare_source(h2)
        tables["table_h2.md"] = (
            h2.groupby("cell_id", observed=True)
            .agg(
                head=("head", "first"),
                loss=("loss", "first"),
                scorer=("scorer", "first"),
                n_seeds=("seed", "nunique"),
                macro_auroc=("macro_auroc", _auroc),
                rare_disease_auroc=(rare_src, _auroc),
                mAP=("mAP", _auroc),
            )
            .reset_index()
        )

    # H3: scorer architecture — MLP head + FN-weighted (MLP vs KAN scorer);
    # cross-checks append KAN-head and edge MLP-vs-KAN scorer pairs.
    h3 = _cell_slice(
        ["mlp_fn_mlp", "mlp_fn_kan", "kan_fn_kan",
         "edge_contrastive", "edge_contrastive_kan"]
    )
    if not h3.empty:
        tables["table_h3.md"] = (
            h3.groupby("cell_id", observed=True)
            .agg(
                head=("head", "first"),
                scorer=("scorer", "first"),
                params_total=("params_total", _fmt_params),
                n_seeds=("seed", "nunique"),
                macro_auroc=("macro_auroc", _auroc),
                mAP=("mAP", _auroc),
            )
            .reset_index()
        )

    # H4: edge feature usage — FastKAN head, varying edge features / aux loss.
    h4 = _cell_slice(
        ["zonly_fn", "edge_scorer_no_aux", "edge_contrastive", "edge_align"]
    )
    if not h4.empty:
        rare_src = _rare_source(h4)
        tables["table_h4.md"] = (
            h4.groupby("cell_id", observed=True)
            .agg(
                scorer=("scorer", "first"),
                lambda_edge=("lambda_edge", "first"),
                lambda_edge_align=("lambda_edge_align", "first"),
                n_seeds=("seed", "nunique"),
                macro_auroc=("macro_auroc", _auroc),
                rare_disease_auroc=(rare_src, _auroc),
                mAP=("mAP", _auroc),
            )
            .reset_index()
        )

    return tables


def _print_hypothesis_summary(df: pd.DataFrame) -> None:
    """Report per-hypothesis cell coverage and seed depth from the source rows.

    Coverage is judged against the exact cell_id list each hypothesis isolates,
    so a partially complete ablation CSV prints precisely which cells are still
    missing rather than a derived-label guess.
    """
    print("\n-- Hypothesis Data Coverage ------------------------------------------")
    cell_id = df["cell_id"].astype(str)

    def _gate(tag: str, label: str, required_cells: list[str]) -> None:
        seeds = {
            c: int(df.loc[cell_id == c, "seed"].nunique()) for c in required_cells
        }
        missing = [c for c, n in seeds.items() if n == 0]
        n_present = len(required_cells) - len(missing)
        if len(missing) == len(required_cells):
            print(
                f"  [{tag}] WARNING: Missing cells: {', '.join(missing)} "
                f"(0/{len(required_cells)} present) — run ablate.py"
            )
        elif missing:
            print(
                f"  [{tag}] WARNING: Missing cells: {', '.join(missing)} "
                f"({n_present}/{len(required_cells)} present)"
            )
        elif min(seeds.values()) < _MIN_SEEDS:
            low = {c: n for c, n in seeds.items() if n < _MIN_SEEDS}
            print(
                f"  [{tag}] WARNING: Cells below {_MIN_SEEDS} seeds: {low} — "
                f"({n_present}/{len(required_cells)} present)"
            )
        else:
            print(
                f"  [{tag}] READY — {label} "
                f"({min(seeds.values())}+ seeds, {n_present}/{len(required_cells)} cells)"
            )

    _gate("H1", "head architecture",
          ["mlp_infonce", "kan_infonce", "kan_wide_infonce", "reskan_infonce"])
    _gate("H2", "FN-weighted loss", ["mlp_infonce", "mlp_fn_mlp"])
    _gate("H3", "KAN scorer", ["mlp_fn_mlp", "mlp_fn_kan"])
    _gate(
        "H4", "edge signals",
        ["zonly_fn", "edge_scorer_no_aux", "edge_contrastive", "edge_align"],
    )
    print("----------------------------------------------------------------------\n")


def _publish(out_dir: Path, written: list[str], publish_dir: Path) -> None:
    publish_dir.mkdir(parents=True, exist_ok=True)
    for filename in written:
        shutil.copy2(out_dir / filename, publish_dir / filename)
        print(f"[tables] published {filename} -> {publish_dir}")


def main() -> None:
    args = _parse_args()
    in_path = Path(args.input)
    if not in_path.is_absolute():
        in_path = PROJECT_ROOT / in_path
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = _read_ablation(in_path)
    split = _resolve_split(raw, args.split)
    df = _prepare(raw, args.include_smoke, args.include_failed, _metric_aliases(split))
    per_class_cols = _per_class_cols(df, split)
    tables = _build_tables(df, per_class_cols)
    split_tag = "test split" if split == "test" else "VAL split (biased)"
    titles = {
        "table_h1.md": f"H1 Geometry: MLP vs KAN vs res_KAN ({split_tag})",
        "table_h2.md": f"H2 InfoNCE vs FN-weighted, with rare-class AUROC ({split_tag})",
        "table_h3.md": f"H3 FN Scorer Comparison ({split_tag})",
        "table_h4.md": f"H4 Edge-aware Scorer Ablation ({split_tag})",
    }
    for filename, table in tables.items():
        _write_table(out_dir / filename, titles[filename], table)
    print(f"[tables] wrote {len(tables)} tables to {out_dir} (metrics: {split} split)")

    if args.publish:
        publish_dir = Path(args.publish_dir)
        if not publish_dir.is_absolute():
            publish_dir = PROJECT_ROOT / publish_dir
        _publish(out_dir, list(tables.keys()), publish_dir)

    _print_hypothesis_summary(df)


if __name__ == "__main__":
    main()
