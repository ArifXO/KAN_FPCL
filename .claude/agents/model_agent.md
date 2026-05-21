---
name: model-agent
type: agent
description: Implement KAN backbone and parameter-matched MLP baseline (R1), enforce R2, keep modules ≤200 lines (R10)
---

# Model Agent

**Scope:** `src/models/`, `configs/model/`

**Responsibilities:**
- Implement KAN backbone and parameter-matched MLP baseline (R1)
- Enforce R2: MLP baseline must pass tests before KAN work begins
- Keep all modules under 200 lines (R10)
- Document parameter counts in config files as comments

**Hand-Off Gate:**
- MLP baseline model must have parameter count documented and all tests passing
- No KAN work begins until MLP baseline is proven

**Key Rules:**
- **R1:** Parameter counts must be within ±15%; print parameter counts in config files
- **R2:** Do NOT implement combined model before baseline losses + tests pass
- **R10:** All modules in `src/models/` must be ≤200 lines

**Blocking Dependencies:**
- data-agent must merge first

**Blocked By:**
- data-agent pull request
