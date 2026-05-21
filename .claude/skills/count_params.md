---
name: count-params
description: Count and compare trainable parameters between all registered model configs to verify R1 (parameter-matched baselines).
---

# count-params skill

Instantiates models from Hydra configs and prints parameter counts for compliance with R1.

## Steps
1. For each YAML in `configs/model/`, instantiate the model using Hydra's `instantiate` with a dummy input size.
2. Count trainable parameters using `sum(p.numel() for p in model.parameters() if p.requires_grad)`.
3. Group results by encoder type (KAN vs MLP) and find pairs with the same config base name.
4. For each KAN/MLP pair, compute the parameter ratio.
5. Print a table:

```
Model                   Params      vs Baseline   Status
mlp_encoder_d4_w256     1,234,567   —             baseline
kan_encoder_d4_w256     1,198,432   -2.9%         PASS (within 5%)
kan_encoder_d4_w512     2,441,000   +97.8%        FAIL (not matched)
```

6. Exit with code 1 if any KAN model deviates more than 5% from its baseline pair.

## Rule
R1 — every KAN result must pair with a parameter-matched MLP baseline.
