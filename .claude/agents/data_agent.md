---
name: data-agent
type: agent
description: Implement MedMNIST dataset wrappers with patient-level splits (R4, R5), validate split disjointness, write unit tests for masking utilities (R3)
---

# Data Agent

**Scope:** `src/data/`, `configs/data/`, `tests/test_data_*.py`

**Responsibilities:**
- Implement MedMNIST dataset wrappers with patient-level splits (R4, R5)
- Validate split disjointness and raise on leakage (R5)
- Write unit tests for all masking utilities (R3)
- Enforce R4, R5, R9, R10

**Hand-Off Gate:**
- Must pass all tests in `tests/test_data_*.py` before model-agent begins dataset usage
- Review-agent must audit for data leakage before merge

**Key Rules:**
- **R4:** Dataset splits must be patient-level where patient IDs exist
- **R5:** Train/Val/Test patient sets must be disjoint; raise `ValueError` if overlap detected
- **R3:** All masking utilities must have positive/negative/FN case unit tests
- **R10:** All modules in `src/data/` must be ≤200 lines

**Blocking Dependencies:**
- None (data-agent is the first stage)

**Blocked By:**
- None
