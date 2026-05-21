# CLAUDE.md — KAN-FPCL

Scientific development rules. All contributors (human and AI) must follow these without exception.

---

## R1 — Paired Baselines
Every KAN result must be accompanied by a parameter-matched MLP baseline trained under identical conditions. Never report KAN metrics alone.

## R2 — Baseline-First Order
Do not implement any combined model until the standalone baseline losses and their unit tests pass cleanly. No combined objectives before components are individually validated.

## R3 — Contrastive Mask Unit Tests
All contrastive mask functions must have unit tests covering three cases: positive pairs, negative pairs, and false-negative pairs (same-class negatives that must be excluded). Tests live in `tests/`.

## R4 — Patient-Level Splits
Dataset splits must be performed at the patient level wherever patient IDs are available. Never split at the image level when patient grouping is known.

## R5 — No Data Leakage
Train, validation, and test patient sets must be strictly disjoint. Assert this programmatically at dataset construction time and raise an error if violated.

## R6 — Config-Driven Experiments
All experiments are driven by Hydra configs under `configs/`. No hyperparameters, paths, or experiment settings are hardcoded in source files.

## R7 — Named Loss Components
Every loss function must return `dict[str, torch.Tensor]` with named scalar components (e.g., `{"loss": total, "ce": ce_term, "contrastive": cl_term}`). Never return a bare tensor.

## R8 — Mandatory Run Artifacts
Every training run must save to its output directory: the resolved config YAML, the current git commit hash, a metrics JSON file, the parameter count of each model, and the wall-clock runtime.

## R9 — No Silent Failures
Raise descriptive, typed exceptions. Never use bare `except: pass` or silent fallbacks that mask errors. If a code path is unimplemented, raise `NotImplementedError` with a message.

## R10 — Module Size Limit
No source module may exceed 200 lines. If a file grows past this, split it into focused sub-modules before continuing development.
