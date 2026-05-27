"""Centralized path resolution for all run artifacts (R8)."""

from __future__ import annotations

from pathlib import Path


class RunPaths:
    """Resolve every artifact path from a single base directory."""

    def __init__(self, base_dir: str = "runs") -> None:
        self.base = Path(base_dir)
        self.results = self.base / "results"
        self.tables = self.base / "tables"
        self.figures = self.base / "figures"

    def run_dir(self, name: str, run_id: str) -> Path:
        return self.base / f"{name}_{run_id}"

    def smoke_dir(self) -> Path:
        return self.base / "smoke"

    def ablation_dir(self, hypothesis: str) -> Path:
        return self.base / "ablation" / hypothesis

    def probe_csv(self) -> Path:
        return self.results / "probe_results.csv"

    def ablation_csv(self) -> Path:
        return self.results / "ablation_master.csv"

    def geometry_csv(self) -> Path:
        return self.results / "geometry.csv"

    def ensure_dirs(self) -> None:
        for d in [
            self.smoke_dir(),
            self.ablation_dir("h1_architecture"),
            self.ablation_dir("h2_fn_weighted"),
            self.ablation_dir("h3_kan_scorer"),
            self.ablation_dir("h4_edge_signal"),
            self.results,
            self.tables,
            self.figures,
            self.figures / "loss_curves",
            self.figures / "umap",
            self.figures / "geometry",
        ]:
            d.mkdir(parents=True, exist_ok=True)
