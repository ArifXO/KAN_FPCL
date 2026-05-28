"""Generate H1-H4 Markdown tables from ``ablation_master.csv``.

The ablation CSV is expected to come from ``scripts/ablate.py``. Failed rows
and smoke rows are excluded by default so thesis tables only average completed
full-run cells.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

import pandas as pd

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))

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
METRIC_ALIASES = {
    "macro_auroc": ("macro_auroc_linear", "macro_auroc"),
    "rare_disease_auroc": (
        "rare_disease_auroc",
        "rare_disease_auroc_linear",
        "rare_auroc",
        "rare_auroc_linear",
    ),
    "mAP": ("mAP", "map", "mean_average_precision"),
}

# H1/H2/H3/H4: minimum seeds needed to count as "enough data"
_MIN_SEEDS = 3


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="runs/results/ablation_master.csv",
                   help="Ablation CSV path.")
    p.add_argument("--output-dir", default="runs/tables", help="Directory for tables.")
    p.add_argument("--include-smoke", action="store_true", help="Keep smoke_* rows.")
    p.add_argument("--include-failed", action="store_true", help="Keep FAILED rows.")
    return p.parse_args()


def _read_ablation(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Expected ablation CSV at {path}.\n"
            f"Run scripts/ablate.py first, or pass --input <path> to point to a different file."
        )
    df = pd.read_csv(path)
    if len(df) == 0:
        raise ValueError(
            f"{path} exists but has no data rows.\n"
            f"Run scripts/ablate.py first to populate it."
        )
    missing = sorted(CORE_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def _metric_column(df: pd.DataFrame, canonical: str) -> str:
    for name in METRIC_ALIASES[canonical]:
        if name in df.columns:
            df[canonical] = pd.to_numeric(df[name], errors="coerce")
            return canonical
    df[canonical] = math.nan
    return canonical


def _prepare(df: pd.DataFrame, include_smoke: bool, include_failed: bool) -> pd.DataFrame:
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
    for metric in METRIC_ALIASES:
        _metric_column(out, metric)
    return out.reset_index(drop=True)


def _norm(value: object) -> str:
    return str(value).strip().lower().replace("-", "_")


def _head_family(value: object) -> str:
    head = _norm(value)
    if "res" in head and "kan" in head:
        return "res_KAN"
    if "kan" in head:
        return "KAN"
    if "mlp" in head:
        return "MLP"
    return str(value)


def _loss_family(value: object) -> str | None:
    loss = _norm(value)
    if loss in {"infonce", "info_nce"}:
        return "InfoNCE"
    if "fn" in loss:
        return "FN-weighted"
    return None


def _scorer_family(value: object) -> str | None:
    scorer = _norm(value)
    if "edge" in scorer:
        return None
    if "kan" in scorer:
        return "FN+KAN_scorer"
    if "mlp" in scorer:
        return "FN+MLP_scorer"
    return None


def _edge_mode(row: pd.Series) -> str:
    lam = row.get("lambda_edge", 0.0)
    align = row.get("lambda_edge_align", 0.0)
    lam = 0.0 if pd.isna(lam) else float(lam)
    align = 0.0 if pd.isna(align) else float(align)
    if lam == 0.0 and align == 0.0:
        return "z-only"
    if lam > 0.0 and align > 0.0:
        return "edge-aware + edge-align"
    if lam > 0.0:
        return "edge-aware"
    return "edge-align"


def _fmt_metric(values: pd.Series, digits: int = 4) -> str:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return "NA"
    mean = vals.mean()
    std = vals.std(ddof=1) if len(vals) > 1 else 0.0
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


def _edge_off(df: pd.DataFrame) -> pd.Series:
    return df["lambda_edge"].fillna(0).eq(0) & df["lambda_edge_align"].fillna(0).eq(0)


def _parse_per_class_auroc(cell_df: pd.DataFrame) -> dict[str, list[float]]:
    """Parse per_class_auroc_linear_json across seeds into {class_name: [auroc, ...]}."""
    col = "per_class_auroc_linear_json"
    if col not in cell_df.columns:
        return {}
    per_class: dict[str, list[float]] = {}
    for raw in cell_df[col].dropna():
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(parsed, dict):
            continue
        for cls, val in parsed.items():
            try:
                per_class.setdefault(cls.lower(), []).append(float(val))
            except (TypeError, ValueError):
                continue
    return per_class


def _summarize(
    df: pd.DataFrame,
    group_cols: list[str],
    metrics: list[str],
    label_order: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    if df.empty:
        columns = [*group_cols, "params_total", "n_seeds", *metrics]
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for keys, group in df.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(group_cols, keys)}
        row["params_total"] = _fmt_params(group["params_total"])
        row["n_seeds"] = _seed_count(group["seed"])
        for metric in metrics:
            row[metric] = _fmt_metric(group[metric])
        rows.append(row)
    out = pd.DataFrame(rows)
    if label_order:
        order_cols = []
        for col, order in label_order.items():
            if col in out.columns:
                order_col = f"__order_{col}"
                ranks = {label: idx for idx, label in enumerate(order)}
                out[order_col] = out[col].map(ranks).fillna(len(order))
                order_cols.append(order_col)
        if order_cols:
            out = out.sort_values(order_cols, na_position="last").drop(columns=order_cols)
    return out.reset_index(drop=True)


def _summarize_h2_with_rare(df: pd.DataFrame) -> pd.DataFrame:
    """Build H2 table: macro_auroc + rare_disease_auroc + 3 rarest per-class columns."""
    h2 = df[_edge_off(df)].copy()
    h2["loss_family"] = h2["loss"].map(_loss_family)
    h2 = h2[h2["loss_family"].notna()]

    rows: list[dict[str, object]] = []
    for loss_fam, group in h2.groupby("loss_family", sort=False):
        row: dict[str, object] = {
            "loss_family": loss_fam,
            "params_total": _fmt_params(group["params_total"]),
            "n_seeds": _seed_count(group["seed"]),
            "macro_auroc": _fmt_metric(group["macro_auroc"]),
            "rare_disease_auroc": _fmt_metric(group["rare_disease_auroc"]),
            "mAP": _fmt_metric(group["mAP"]),
        }
        # Rare per-class AUROC from JSON column
        per_class = _parse_per_class_auroc(group)
        for cls in RARE_CLASSES:
            vals = per_class.get(cls, [])
            col_name = f"auroc_{cls}"
            if vals:
                s = pd.Series(vals)
                mean = s.mean()
                std = s.std(ddof=1) if len(vals) > 1 else 0.0
                row[col_name] = f"{mean:.4f} ± {std:.4f}"
            else:
                row[col_name] = "NA"
        rows.append(row)

    out = pd.DataFrame(rows)
    # Sort InfoNCE before FN-weighted
    order = {"InfoNCE": 0, "FN-weighted": 1}
    out["__ord"] = out["loss_family"].map(order).fillna(2)
    out = out.sort_values("__ord").drop(columns="__ord").reset_index(drop=True)
    return out


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
    """Mean AUROC across rare classes from per_class_auroc_linear_json."""
    if not json_str or pd.isna(json_str):
        return float("nan")
    try:
        data = json.loads(json_str)
        vals = [data[c] for c in RARE_CLASSES if c in data]
        return float(np.mean(vals)) if vals else float("nan")
    except (json.JSONDecodeError, KeyError):
        return float("nan")


def _build_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
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

    # H1: head architecture — InfoNCE rows only, no scorer.
    h1 = _cell_slice(["mlp_infonce", "kan_infonce", "reskan_infonce"])
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
        if "per_class_auroc_linear_json" in h2.columns:
            h2["rare_auroc"] = h2["per_class_auroc_linear_json"].apply(_extract_rare_auroc)
            rare_src = "rare_auroc"
        else:
            rare_src = "rare_disease_auroc"
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
        tables["table_h4.md"] = (
            h4.groupby("cell_id", observed=True)
            .agg(
                scorer=("scorer", "first"),
                lambda_edge=("lambda_edge", "first"),
                lambda_edge_align=("lambda_edge_align", "first"),
                n_seeds=("seed", "nunique"),
                macro_auroc=("macro_auroc", _auroc),
                rare_disease_auroc=("rare_disease_auroc", _auroc),
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

    _gate("H1", "head architecture", ["mlp_infonce", "kan_infonce", "reskan_infonce"])
    _gate("H2", "FN-weighted loss", ["mlp_infonce", "mlp_fn_mlp"])
    _gate("H3", "KAN scorer", ["mlp_fn_mlp", "mlp_fn_kan"])
    _gate(
        "H4", "edge signals",
        ["zonly_fn", "edge_scorer_no_aux", "edge_contrastive", "edge_align"],
    )
    print("----------------------------------------------------------------------\n")


def main() -> None:
    args = _parse_args()
    in_path = Path(args.input)
    if not in_path.is_absolute():
        in_path = PROJECT_ROOT / in_path
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _prepare(_read_ablation(in_path), args.include_smoke, args.include_failed)
    tables = _build_tables(df)
    titles = {
        "table_h1.md": "H1 Geometry: MLP vs KAN vs res_KAN",
        "table_h2.md": "H2 InfoNCE vs FN-weighted (with rare-class AUROC)",
        "table_h3.md": "H3 FN Scorer Comparison",
        "table_h4.md": "H4 Edge-aware Scorer Ablation",
    }
    for filename, table in tables.items():
        _write_table(out_dir / filename, titles[filename], table)
    print(f"[tables] wrote {len(tables)} tables to {out_dir}")

    _print_hypothesis_summary(df)


if __name__ == "__main__":
    main()
