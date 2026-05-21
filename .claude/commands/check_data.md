---
name: check-data
type: command
description: Run dataset-leakage-checker on current src/data/ changes, verify R4/R5 compliance
usage: /check-data
---

# Check Data

**Command:** `/check-data`

**Purpose:** Audit dataset split implementation for patient-level splits (R4) and leakage prevention (R5).

**What It Does:**
1. Finds all modified files in `src/data/`
2. Checks R4 compliance: splits are patient-level (not row-level)
3. Checks R5 compliance: train/val/test patient sets are disjoint
4. Verifies no overlap and raises `ValueError` if leakage detected
5. Documents split composition (image counts per patient group)

**Compliance Checks:**

| **Rule** | **Check** | **Fail If** |
|---|---|---|
| R4 | Splits use patient IDs, not row indices | Uses row-level random split |
| R4 | Patient ID extraction documented | Unclear how patient ID derived |
| R5 | len(train_ids & val_ids) == 0 | Any overlap in train/val |
| R5 | len(val_ids & test_ids) == 0 | Any overlap in val/test |
| R5 | len(train_ids & test_ids) == 0 | Any overlap in train/test |
| R5 | ValueError raised on leakage | Silent acceptance of overlap |
| R9 | Error messages descriptive | Bare assert with no context |
| Documentation | Split stats logged | No train/val/test composition shown |

**Exit Codes:**
- `0`: All checks pass
- `1`: One or more FAILs (likely data leakage detected)

**Example:**
```bash
/check-data
# Output:
# [PASS] ChestMNIST splits use patient-level IDs
# [PASS] Train patient IDs: 3250 unique
# [PASS] Val patient IDs: 812 unique
# [PASS] Test patient IDs: 908 unique
# [PASS] Disjointness verified: no overlap
# Dataset auditor: 0 FAILs
```

**When to Run:**
- After implementing dataset split logic
- After modifying patient ID extraction
- Before committing data-agent PR

**Critical:** If `/check-data` reports any leakage, the run is invalid and all downstream results are unpublishable.
