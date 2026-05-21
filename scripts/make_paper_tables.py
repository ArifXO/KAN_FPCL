"""Generate H1-H4 Markdown tables from ``ablation_master.csv``.

The ablation CSV is expected to come from ``scripts/ablate.py``. Failed rows
and smoke rows are excluded by default so thesis tables only average completed
full-run cells.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))

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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="ablation_master.csv", help="Ablation CSV path.")
    p.add_argument("--output-dir", default="reports/tables", help="Directory for tables.")
    p.add_argument("--include-smoke", action="store_true", help="Keep smoke_* rows.")
    p.add_argument("--include-failed", action="store_true", help="Keep FAILED rows.")
    return p.parse_args()


def _read_ablation(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Expected ablation CSV at {path}. Run scripts/ablate.py first or pass --input."
        )
    df = pd.read_csv(path)
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
    return f"{mean:.{digits}f} +- {std:.{digits}f}"


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


def _markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        display = df.fillna("NA").astype(str)
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


def _build_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    h1 = df.dropna(subset=["alignment", "uniformity", "effective_rank"], how="all").copy()
    h1["projector"] = h1["head"].map(_head_family)
    table_h1 = _summarize(
        h1,
        ["projector"],
        ["alignment", "uniformity", "effective_rank"],
        {"projector": ["MLP", "KAN", "res_KAN"]},
    )

    h2 = df[_edge_off(df)].copy()
    h2["loss_family"] = h2["loss"].map(_loss_family)
    h2 = h2[h2["loss_family"].notna()]
    table_h2 = _summarize(
        h2,
        ["loss_family"],
        ["macro_auroc", "rare_disease_auroc", "mAP"],
        {"loss_family": ["InfoNCE", "FN-weighted"]},
    )

    h3 = df[_edge_off(df)].copy()
    h3["loss_family"] = h3["loss"].map(_loss_family)
    h3["scorer_variant"] = h3["scorer"].map(_scorer_family)
    h3 = h3[(h3["loss_family"] == "FN-weighted") & h3["scorer_variant"].notna()]
    table_h3 = _summarize(
        h3,
        ["scorer_variant"],
        ["macro_auroc"],
        {"scorer_variant": ["FN+MLP_scorer", "FN+KAN_scorer"]},
    )

    h4 = df.copy()
    h4["edge_mode"] = h4.apply(_edge_mode, axis=1)
    is_edge = (
        h4["loss"].astype(str).str.contains("edge", case=False, na=False)
        | h4["scorer"].astype(str).str.contains("edge", case=False, na=False)
        | h4["cell_id"].astype(str).str.contains("edge|z_only", case=False, na=False)
        | h4["lambda_edge"].fillna(0).ne(0)
        | h4["lambda_edge_align"].fillna(0).ne(0)
    )
    h4 = h4[is_edge]
    h4["edge_align"] = h4["lambda_edge_align"].fillna(0).gt(0).map({True: "on", False: "off"})
    table_h4 = _summarize(
        h4,
        ["edge_mode", "lambda_edge", "edge_align"],
        ["macro_auroc", "rare_disease_auroc"],
        {"edge_mode": ["z-only", "edge-aware", "edge-align", "edge-aware + edge-align"]},
    )
    return {
        "table_h1.md": table_h1,
        "table_h2.md": table_h2,
        "table_h3.md": table_h3,
        "table_h4.md": table_h4,
    }


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
        "table_h2.md": "H2 InfoNCE vs FN-weighted",
        "table_h3.md": "H3 FN Scorer Comparison",
        "table_h4.md": "H4 Edge-aware Scorer Ablation",
    }
    for filename, table in tables.items():
        _write_table(out_dir / filename, titles[filename], table)
    print(f"[tables] wrote {len(tables)} tables to {out_dir}")


if __name__ == "__main__":
    main()
