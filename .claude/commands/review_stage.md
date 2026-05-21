---
name: review-stage
type: command
description: Run experiment-auditor on Stage N, reports 0 FAILs or lists issues
usage: /review-stage <N>
---

# Review Stage

**Command:** `/review-stage <N>`

**Purpose:** Audit all experiment and artifact compliance for a given stage before advancing.

**What It Does:**
1. Identifies all commits/PRs for Stage N
2. Runs experiment-auditor (R6, R8 checks) on all new code
3. Verifies all artifacts saved: config YAML, git hash, metrics JSON, param count
4. Reports results: 0 FAILs (pass) or lists issues

**Stage Gate Definitions:**

| **Stage** | **R6 Checks** | **R8 Checks** | **Blocking Tests** |
|---|---|---|---|
| 0 (Repo Bootstrap) | YAML present | N/A | `pytest tests/test_smoke.py -v` |
| 1 (Data Pipeline) | data/*.yaml | artifact dirs exist | `pytest tests/test_data_*.py -v` |
| 2 (Baseline Models) | model/*.yaml | checkpoints saved | `pytest tests/ -v --cov=src` |
| 6 (FN Loss + Scorer) | loss/*.yaml | metrics.json valid | `pytest tests/test_loss_*.py -v` |
| 7 (KAN Scorer) | experiment/*.yaml | full artifact set | `pytest tests/ -v --cov=src` |
| 9 (Metrics + Audit) | N/A | final reports | `pytest tests/ -v` |

**Exit Codes:**
- `0`: All checks pass (0 FAILs)
- `1`: One or more FAILs; lists them

**Example:**
```bash
/review-stage 2
# Output:
# [PASS] R6 model/mlp.yaml and model/kan.yaml present
# [PASS] R8 checkpoints/run_*/config.yaml verified for all runs
# [PASS] pytest tests/ all green
# Gate 2 READY for merge
```
