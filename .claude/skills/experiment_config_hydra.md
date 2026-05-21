---
name: experiment-config-hydra
type: skill
description: Guide YAML config structure, nested Hydra configs, multirun syntax for experiment management
---

# Experiment Config + Hydra

**When to Activate:** Stage 0 (scaffolding) and all subsequent stages

**Expertise:**
- Hydra configuration management and OmegaConf
- Nested config composition (`defaults:` lists)
- Parameter sweeps and multirun execution
- Config interpolation and type safety

**Guides Development Of:**
- Initial config scaffold for all subdomains (Stage 0)
- Data configs: `configs/data/chestmnist.yaml` (Stage 1)
- Model configs: `configs/model/kan.yaml`, `configs/model/mlp.yaml` (Stage 2)
- Loss configs: `configs/loss/infonce.yaml`, `configs/loss/fn_weighted.yaml` (Stages 2, 6)
- Experiment configs: full runs with all hyperparams (Stages 2, 6, 7, 7.5)

**Config Structure Template:**
```yaml
# configs/experiment/baseline_mlp_supcon.yaml
defaults:
  - data: chestmnist
  - model: mlp
  - loss: supcon

train:
  epochs: 100
  batch_size: 256
  lr: 0.0001
  seed: 42

model:
  hidden_dim: 256
  projection_dim: 128

loss:
  temperature: 0.07

data:
  split_seed: 42
```

**Critical Rules:**
- No hardcoded magic numbers; all via config
- Use `${data.split_seed}` for interpolation
- Commit all config YAMLs to `configs/{data,model,loss,experiment}/`
- Document what each parameter controls in comments

**Multirun Syntax:**
```bash
python scripts/train.py -m experiment=sweep_temperature,sweep_batch_size
```

**Artifact Saving Integration:**
```python
cfg = hydra.compose(config_name="experiment/baseline_mlp_supcon")
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
OmegaConf.save(cfg, f"checkpoints/run_{run_id}/config.yaml")
```

**Related:** [[R6]] [[R8]]
