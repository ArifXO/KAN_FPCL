---
name: stage-gate
description: Run the stage gate checks before advancing to the next development stage. Verifies that all prerequisite tests pass and rule compliance holds.
---

# stage-gate command

Runs the prerequisite checks for the requested development stage.

## Usage
```
/stage-gate <stage>
```
Stages: `data`, `models`, `losses`, `experiments`, `reporting`

## Gate definitions

### /stage-gate data
- [ ] `pytest tests/test_data_*.py -v` — all pass
- [ ] No patient ID overlap between splits (asserted in tests)
- [ ] `/check-rules` R4, R5 — PASS

### /stage-gate models
- [ ] `data` gate has passed
- [ ] `pytest tests/test_model_*.py -v` — all pass
- [ ] `/count-params` — all KAN/MLP pairs within 5%
- [ ] `/check-rules` R1, R10 — PASS

### /stage-gate losses
- [ ] `pytest tests/test_loss_*.py -v` — all pass
- [ ] All mask functions have positive/negative/FN tests
- [ ] `/check-rules` R3, R7, R9 — PASS

### /stage-gate experiments
- [ ] `losses` gate has passed
- [ ] `models` gate has passed
- [ ] Smoke run: `python scripts/train.py experiment=baseline_mlp_ntxent +trainer.max_epochs=1`
- [ ] Output directory contains: config.yaml, git_hash.txt, metrics.json
- [ ] `/check-rules` R6, R8 — PASS

### /stage-gate reporting
- [ ] `experiments` gate has passed
- [ ] All four experiment configs have completed runs
- [ ] `reports/figures/` and `reports/tables/` contain expected outputs
- [ ] review-agent checklist: all R1-R10 PASS
