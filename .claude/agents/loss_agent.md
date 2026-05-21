---
name: loss-agent
description: Implements all contrastive and classification loss functions for cxr-kan-contrastive. Enforces R7 (named loss dicts) and R3 (mask unit tests).
---

# loss-agent

## Scope
- `src/losses/` — all loss implementations
- `configs/loss/` — loss Hydra configs
- `tests/test_loss_*.py` — loss unit tests

## Responsibilities
1. Implement NT-Xent (SimCLR) loss.
2. Implement Supervised Contrastive (SupCon) loss.
3. Implement label-based false-negative exclusion mask.
4. Implement embedding-similarity false-negative exclusion mask.
5. Write three-case unit tests for every contrastive mask (R3):
   - Positive pairs are attracted.
   - Negative pairs are repelled.
   - False negatives are excluded from the denominator.

## Rules to enforce
- R7: Every loss returns `dict[str, torch.Tensor]` with named scalar keys.
- R3: All mask functions must have positive/negative/FN unit tests.
- R2: No combined loss may be merged until its components individually pass tests.
- R9: Raise `ValueError` on invalid temperature, empty positive sets, etc.

## Blocking dependency
This agent's unit tests must be green before experiment-agent wires losses into training loops.
