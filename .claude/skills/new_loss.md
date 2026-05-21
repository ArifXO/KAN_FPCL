---
name: new-loss
description: Scaffold a new loss function module in src/losses/ with the correct dict[str, Tensor] return signature and corresponding test stubs.
---

# new-loss skill

Scaffolds a compliant loss function and its test file.

## Usage
```
/new-loss <loss_name>
```
Example: `/new-loss focal_supcon`

## Steps
1. Create `src/losses/<loss_name>.py` with:
   - A class `<CamelCase>Loss(torch.nn.Module)` 
   - `forward()` returning `dict[str, torch.Tensor]` with at least keys `"loss"` and named components (R7)
   - A `NotImplementedError` body so R9 is satisfied until implemented
2. Create `tests/test_loss_<loss_name>.py` with three stub test functions:
   - `test_<loss_name>_positive_pairs` — assert loss decreases for same-class pairs
   - `test_<loss_name>_negative_pairs` — assert loss decreases when cross-class pairs repelled
   - `test_<loss_name>_fn_exclusion` — assert same-class negatives are excluded from denominator (R3)
3. Add an entry to `configs/loss/<loss_name>.yaml` stub.
4. Print reminder to implement and make tests pass before wiring into experiments (R2).
