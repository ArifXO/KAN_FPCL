---
name: check-rules
description: Scan the codebase for violations of R1-R10 from CLAUDE.md and report a compliance summary.
---

# check-rules skill

Scans the working directory for rule violations and prints a compliance report.

## Steps
1. **R6** — Grep `src/` and `scripts/` for numeric literals that look like hyperparameters (learning rates, batch sizes, temperatures). Flag any not behind a config reference.
2. **R7** — Grep `src/losses/` for functions returning `torch.Tensor` directly without wrapping in a dict.
3. **R9** — Grep for `except:` with no exception type and for `except Exception: pass`.
4. **R10** — Count lines in every `.py` file under `src/`. Flag files over 180 lines (warn) or 200 lines (error).
5. **R3** — For each mask function found in `src/losses/` or `src/data/`, check that `tests/` contains a test file with "positive", "negative", and "fn" test cases.

## Output format
```
R1  [SKIP — no results yet]
R2  [SKIP — no combined models yet]
R3  PASS | FAIL: <file>:<line> missing FN test for <function>
R4  [SKIP — no dataset code yet]
R5  [SKIP — no dataset code yet]
R6  PASS | FAIL: <file>:<line> hardcoded value <value>
R7  PASS | FAIL: <file>:<line> bare tensor return
R8  [SKIP — no training scripts yet]
R9  PASS | FAIL: <file>:<line> bare except
R10 PASS | WARN: <file> <N> lines | FAIL: <file> <N> lines
```
