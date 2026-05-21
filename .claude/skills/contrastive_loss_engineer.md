---
name: contrastive-loss-engineer
type: skill
description: Design masked loss functions, FN weighting, edge-aware mechanisms for contrastive learning
---

# Contrastive Loss Engineer

**When to Activate:** Stages 2, 6, 7, 7.5

**Expertise:**
- Masked contrastive loss design (false-negative exclusion)
- FN weighting formulas and numerical stability
- Edge-aware mechanisms derived from KAN splines
- Unit test design for loss functions

**Guides Development Of:**
- InfoNCE loss with false-negative masks (Stage 2)
- FN-weighted contrastive loss (Stage 6)
- KAN-based pair scorer for false-negative probability (Stage 7)
- Edge-aligned loss leveraging KAN edge fingerprints (Stage 7.5)

**Critical Stability Checks:**
- Verify `log(1 - p_FN)` does not overflow when p_FN → 1 (clamp min 1e-10)
- Ensure p_FN bounded to [0, 1]; raise if min/max outside range
- Check L2-normalize stability (clamp norm to 1e-12 minimum)
- Validate edge fingerprint compression (d_in × d_out × num_centers → 256 dims mandatory)

**Unit Test Template:**
```python
# Positive-only batch: loss should be near zero
# Negative-only batch: loss should be finite and > 0
# False-negative case: mark true positive as negative, loss should increase
```

**Output Format:**
All losses must return:
```python
{
    "loss": scalar_tensor,
    "component_1": scalar,
    "component_2": scalar,
    "temperature": scalar_or_float,
    "pos_sim_mean": scalar,
    "neg_sim_mean": scalar,
}
```

**Related:** [[R7]] [[R3]] [[R2]]
