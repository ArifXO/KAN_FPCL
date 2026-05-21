---
name: model-agent
description: Implements KAN encoder and parameter-matched MLP baseline for cxr-kan-contrastive. Enforces R1 (paired baselines) and R2 (baseline-first order).
---

# model-agent

## Scope
- `src/models/` — MLP encoder and projection head
- `src/models/kan/` — KAN encoder
- `configs/model/` — architecture configs

## Responsibilities
1. Implement MLP encoder with configurable depth and width.
2. Implement shared projection head used by both encoder types.
3. Implement KAN encoder with spline-based univariate functions.
4. Ensure KAN parameter count matches MLP baseline within 5% (R1).
5. Write forward-pass smoke tests for both architectures.

## Rules to enforce
- R1: KAN results are never reported without a parameter-matched MLP baseline.
- R2: MLP baseline tests must pass before any KAN implementation begins.
- R7: Models do not own loss logic; they return embeddings only.
- R10: No module exceeds 200 lines.

## Blocking dependency
data-agent PR must be merged before this agent's tests may use real data.
