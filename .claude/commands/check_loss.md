---
name: check-loss
type: command
description: Run loss-auditor on current src/losses/ changes, verify R3/R7 compliance
usage: /check-loss
---

# Check Loss

**Command:** `/check-loss`

**Purpose:** Audit loss function implementation for compliance with R3 (unit tests), R7 (dict return), and numerical stability.

**What It Does:**
1. Finds all modified files in `src/losses/` and `src/models/pair_scorer.py`
2. Checks R7 compliance: all losses return `dict[str, Tensor]`
3. Checks R3 compliance: all masks have pos/neg/FN unit tests
4. Checks numerical stability: no overflow in log(1 - p_FN), bounds validation
5. Reports PASS/FAIL for each loss function

**Compliance Checks:**

| **Rule** | **Check** | **Fail If** |
|---|---|---|
| R7 | Return type is dict[str, Tensor] | Return is scalar or other type |
| R7 | Dict has "loss" key | Missing "loss" key |
| R7 | All component names documented | Undocumented keys |
| R3 | test_loss_*_positive.py exists | No positive test |
| R3 | test_loss_*_negative.py exists | No negative test |
| R3 | test_loss_*_fn_case.py exists | No FN case test |
| Stability | log(1 - p_FN) has clamp_min(1e-10) | Direct log without clamp |
| Stability | p_FN bounds validated [0, 1] | No bounds check |
| Stability | L2-norm has clamp_min(1e-12) | Direct division by norm |

**Exit Codes:**
- `0`: All checks pass
- `1`: One or more FAILs

**Example:**
```bash
/check-loss
# Output:
# [PASS] src/losses/infonce.py returns dict[str, Tensor]
# [PASS] test_loss_infonce_positive.py green
# [PASS] test_loss_infonce_negative.py green
# [PASS] test_loss_infonce_fn_case.py green
# [PASS] Numerical stability: log(1-p_fn) clamped correctly
# Loss auditor: 0 FAILs
```

**When to Run:**
- After implementing any loss function
- After modifying a pair scorer
- Before committing loss-agent PR
