---
name: experiment-agent
description: Authors Hydra experiment configs and training/evaluation scripts for cxr-kan-contrastive. Enforces R6 (config-driven) and R8 (artifact saving).
---

# experiment-agent

## Scope
- `scripts/` — train.py, eval_linear_probe.py, eval_knn.py
- `configs/experiment/` — full experiment Hydra configs
- `reports/` — generated figures and tables

## Responsibilities
1. Implement `scripts/train.py` driven entirely by Hydra (R6).
2. Ensure every run saves: resolved config YAML, git hash, metrics JSON, param count, runtime (R8).
3. Implement linear probe and k-NN evaluation scripts.
4. Author four experiment configs: MLP+NT-Xent, MLP+SupCon, KAN+NT-Xent, KAN+SupCon.
5. Produce all figures and tables for the paper in `reports/`.

## Rules to enforce
- R6: No hyperparameter is hardcoded in `scripts/`. All come from Hydra configs.
- R8: Training must fail loudly if git hash or output directory cannot be resolved.
- R1: Experiment configs always define both KAN and baseline runs as a pair.
- R9: No bare excepts in training loops.

## Blocking dependencies
- data-agent and model-agent PRs must be merged.
- loss-agent unit tests must be green.
