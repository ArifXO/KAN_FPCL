"""Guards Hydra defaults-list ordering (R6).

`_self_` must be LAST in every experiment config's defaults list so the
experiment file's own `loss:` overrides win over the loss group's zero
defaults. With `_self_` first, `/loss: edge_aware` (lambda_edge=0.0) silently
clobbered the experiment's lambda_edge=0.05 — turning H4 treatment configs
into controls. These tests fail if that regression returns.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = REPO_ROOT / "configs" / "experiment"

# Experiment configs resolve their `/data`, `/model`, `/loss` group defaults
# via a searchpath that interpolates ${oc.env:PROJECT_ROOT}.
os.environ.setdefault("PROJECT_ROOT", str(REPO_ROOT).replace("\\", "/"))


def _compose(config_name: str, overrides=None):
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(EXP_DIR)):
        return compose(config_name=config_name, overrides=overrides or [])


def test_edge_l005_lambda_edge_resolves_to_treatment():
    cfg = _compose("full_edge_l005")
    assert cfg.loss.lambda_edge == 0.05


def test_edge_align_l005_lambda_edge_align_resolves_to_treatment():
    cfg = _compose("full_edge_align_l005")
    assert cfg.loss.lambda_edge_align == 0.05


def test_edge_scorer_no_aux_lambda_edge_stays_zero():
    cfg = _compose("full_edge_scorer_no_aux")
    assert cfg.loss.lambda_edge == 0.0


def test_mlp_infonce_temperature_from_loss_group():
    cfg = _compose("full_mlp_infonce")
    assert cfg.loss.temperature == 0.1


@pytest.mark.parametrize(
    "config_name",
    sorted(p.stem for p in EXP_DIR.glob("full_*.yaml")),
)
def test_full_configs_compose_and_resolve(config_name):
    cfg = _compose(config_name, overrides=["train.max_steps=0"])
    # Force interpolation of every node; raises on any ConfigAttributeError
    # or unresolved reference.
    OmegaConf.to_container(cfg, resolve=True)
    assert cfg.train.max_steps == 0
