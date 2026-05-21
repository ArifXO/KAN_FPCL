"""Run-artifact persistence (R8)."""

from __future__ import annotations

import json
import subprocess
import time
import uuid
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from .param_count import format_param_report


def make_run_id() -> str:
    """Timestamp + short uuid — never user input (R8)."""
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def _git_info() -> str:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        return f"commit: {commit}\nbranch: {branch}\n"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "commit: <unavailable: not a git repo or git not installed>\n"


def save_run_artifacts(
    run_dir: Path,
    cfg: DictConfig,
    model: nn.Module,
    named_modules: dict[str, nn.Module],
    metrics: dict,
) -> None:
    """Persist the R8 artifact set into ``run_dir`` (created if missing).

    Files written:
        config.yaml, model.pt, metrics.json, param_count.txt, git_info.txt
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    OmegaConf.save(cfg, run_dir / "config.yaml")
    torch.save(model.state_dict(), run_dir / "model.pt")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "param_count.txt").write_text(format_param_report(named_modules))
    (run_dir / "git_info.txt").write_text(_git_info())
