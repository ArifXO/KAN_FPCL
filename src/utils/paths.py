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

    def probe_csv(self) -> Path:
        return self.results / "probe_results.csv"

    def ablation_csv(self) -> Path:
        return self.results / "ablation_master.csv"

    def geometry_csv(self) -> Path:
        return self.results / "geometry.csv"

    def ensure_dirs(self) -> None:
        for d in [
            self.results,
            self.tables,
            self.figures,
            self.figures / "loss_curves",
            self.figures / "umap",
            self.figures / "geometry",
        ]:
            d.mkdir(parents=True, exist_ok=True)
