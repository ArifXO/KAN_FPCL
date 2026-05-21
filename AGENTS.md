# AGENTS.md KAN-FPCL

Subagent specializations and responsibilities for this project.

---

## Agent Roster

### `data-agent`
**File:** `.claude/agents/data_agent.md`
**Scope:** `src/data/`, `configs/data/`, `tests/test_data_*.py`
**Responsibilities:**
- Implement MedMNIST dataset wrappers with patient-level splits (R4, R5)
- Validate split disjointness and raise on leakage (R5)
- Write unit tests for all masking utilities (R3)

### `model-agent`
**File:** `.claude/agents/model_agent.md`
**Scope:** `src/models/`, `configs/model/`
**Responsibilities:**
- Implement KAN backbone and parameter-matched MLP baseline (R1)
- Enforce R2: MLP baseline must pass tests before KAN work begins
- Keep all modules under 200 lines (R10)

### `loss-agent`
**File:** `.claude/agents/loss_agent.md`
**Scope:** `src/losses/`, `configs/loss/`, `tests/test_loss_*.py`
**Responsibilities:**
- Implement all loss functions returning `dict[str, Tensor]` (R7)
- Write positive/negative/FN unit tests for all contrastive masks (R3)
- No combined losses until individual components pass (R2)

### `experiment-agent`
**File:** `.claude/agents/experiment_agent.md`
**Scope:** `scripts/`, `configs/experiment/`, `reports/`
**Responsibilities:**
- Author Hydra experiment configs (R6)
- Ensure every run saves required artifacts (R8)
- Produce figures and tables in `reports/`

### `review-agent`
**File:** `.claude/agents/review_agent.md`
**Scope:** Entire codebase (read-only audit)
**Responsibilities:**
- Audit compliance with R1–R10 before any merge
- Check for data leakage, bare excepts, hardcoded hyperparams
- Verify baseline always accompanies KAN results

---

## Hand-Off Protocol
1. A data-agent pull request must be merged before model-agent begins dataset usage.
2. Loss-agent unit tests must be green before experiment-agent wires them into training loops.
3. Review-agent must sign off on every PR that touches `src/` or `configs/`.
