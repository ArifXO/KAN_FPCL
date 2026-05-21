---
name: review-agent
type: agent
description: Audit entire codebase for compliance with R1–R10 before any merge; check data leakage, bare excepts, hardcoded hyperparams
---

# Review Agent

**Scope:** Entire codebase (read-only audit)

**Responsibilities:**
- Audit compliance with R1–R10 before any merge
- Check for data leakage, bare excepts, hardcoded hyperparams
- Verify baseline always accompanies KAN results
- Generate PASS/FAIL/WARN report for each rule

**Hand-Off Gate:**
- Must sign off on every PR that touches `src/` or `configs/`
- Output format: JSON with PASS/FAIL/WARN for each rule (R1–R10)

**Audit Checklist (R1–R10):**

- **R1:** Parameter counts within ±15%; comment in config files
- **R2:** Combined models only after baseline tests pass
- **R3:** All contrastive masks have pos/neg/FN unit tests
- **R4:** Dataset splits are patient-level (not row-level)
- **R5:** Train/Val/Test patient sets disjoint; no overlap
- **R6:** Hydra configs for all hyperparams; no hardcoded magic numbers
- **R7:** All losses return `dict[str, Tensor]` with named components
- **R8:** Every run saves config YAML, git hash, metrics JSON, param count, runtime
- **R9:** No bare `except: pass`; descriptive errors with context
- **R10:** All modules in `src/` are ≤200 lines; split if larger

**Blocking Dependencies:**
- None (review-agent blocks others; not blocked)

**Blocked By:**
- None
