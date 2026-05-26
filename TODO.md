# TODO.md - KAN-FPCL Current State

This file tracks the real repo state after Prompts 0-9. A stage is marked
`☑ Complete` only when its gate condition is met. Smoke runs prove execution
only; they are not evidence for H1-H4.

---

## Fix Log (May 2026)

Changes made or reconciled during Prompts 0-9:

- CLAUDE.md: Dataset scope section added. ChestMNIST is the active dataset for
  now; CheXpert training/probe/geometry integration is deferred.
- probe_results.csv: Moved to `reports/tables/probe_results.csv`; fake CheXpert
  result rows removed.
- ablation.yaml: Expanded to 9 ChestMNIST cells, `lambda_edge` fixed at `0.05`
  for edge treatment, and 3 seeds configured: `[42, 1337, 2024]`.
- KANPairScorer: Implemented and tested, so H3 is now testable.
- Training scripts: Cosine LR, gradient clipping, validation monitoring, and
  full `step_metrics` logging added.
- probe.py: Per-class AUROC serialization added, with ChestMNIST class names;
  dataset routing is explicit and does not silently route CheXpert through
  ChestMNIST.
- Full training configs: 9 `full_*.yaml` configs added, each set to 10K steps.
- p_fn collapse guard: Full FN configs include `lambda_pfn_reg=0.01`; edge-aware
  loss config uses `max_fn_weight=0.5`. Caveat: `configs/loss/fn_weighted_mlp.yaml`
  still shows `max_fn_weight: 1.0`, so verify/fix this before full FN ablations.
- make_paper_tables.py: Guard clauses and per-class H2 rare-disease rows added.
- All smoke tests passing: `pytest tests/test_smoke.py -q` -> 5 passed on
  2026-05-26.

Stage status summary:

- Stages 0-3: ☑ Complete - gate conditions met.
- Stage 4 (FastKAN): ☑ Complete.
- Stage 5 (Residual KAN): ☑ Complete.
- Stage 6 (FN-weighted + MLP scorer): ☑ Complete.
- Stage 7 (KAN scorer): ☑ Complete after Prompt 2.
- Stage 7.5 (Edge-aware): ☑ Complete.
- Stage 8 (Geometry): ☑ Complete.
- Stage 9 (CheXpert): ☐ Not Started - deferred; see `CLAUDE.md` Dataset Scope.
- Stage 10 (Ablation): ☐ In Progress - configs/scripts ready, full runs not yet
  executed.

---

## Stage 0: Repository Bootstrap

**Status:** ☑ Complete

**Gate:** `pytest tests/test_smoke.py -v` passes and required files exist.

**Current evidence:**

- Repo structure, `CLAUDE.md`, `AGENTS.md`, `README.md`, `TODO.md`,
  `pyproject.toml`, and smoke tests exist.
- Smoke test verified on 2026-05-26: 5 passed.

**Open items:**

- No gate blocker recorded.

---

## Stage 1: ChestMNIST Smoke Data Pipeline

**Status:** ☑ Complete

**Gate:** `tests/test_data_*.py` pass, patient-level ChestMNIST split exists, and
train/val/test patient sets are disjoint.

**Current evidence:**

- ChestMNIST dataset, transforms, split helpers, and data pipeline tests exist.
- `CLAUDE.md` confirms ChestMNIST is the active development dataset.

**Open items:**

- CheXpert is outside this stage and remains deferred.

---

## Stage 2: MLP Encoder + InfoNCE Baseline

**Status:** ☑ Complete

**Gate:** InfoNCE/MLP tests pass, Hydra training script saves artifacts, and an
MLP baseline checkpoint can be probed.

**Current evidence:**

- ResNet18 encoder, MLP projector, InfoNCE loss, mask tests, checkpoint utilities,
  and `scripts/train.py` are implemented.
- Training script includes cosine LR, grad clipping, validation monitoring, and
  step metrics.
- Baseline probe rows exist in `reports/tables/probe_results.csv`.

**Open items:**

- Full thesis claims still require 3-seed ablation rows, not smoke rows.

---

## Stage 3: Linear Probe + kNN Evaluation

**Status:** ☑ Complete

**Gate:** Probe harness appends AUROC rows for checkpoints.

**Current evidence:**

- `src/metrics/{auc,linear_probe,knn}.py` and `scripts/probe.py` are implemented.
- Probe output now includes `per_class_auroc_linear_json`.
- ChestMNIST per-class AUROC JSON keys are class names, including `hernia`.
- Output path is `reports/tables/probe_results.csv`.

**Open items:**

- Older rows may contain integer per-class keys; new rows use class names.

---

## Stage 4: FastKAN Projector

**Status:** ☑ Complete

**Gate:** FastKAN tests pass, including shape, fixed centers, L2 norm, gradients,
return_edges, and validation errors.

**Current evidence:**

- `src/models/kan/fastkan.py` implements `FastKANLayer` and `FastKANProjector`.
- `return_edges=True` returns `(output, phi)` with `phi[B, out, in]`.
- `configs/model/kan_head.yaml` exists.
- `tests/test_fastkan.py` exists.

**Open items:**

- Full H1 evidence still requires 3-seed ablation/geometry runs.

---

## Stage 5: Residual FastKAN Warp

**Status:** ☑ Complete

**Gate:** Four residual warp invariants pass.

**Current evidence:**

- `src/models/kan/residual_warp.py` implements `ResidualFastKANWarp`.
- `configs/model/residual_fastkan_warp.yaml` exists.
- `tests/test_residual_kan.py` covers identity at alpha 0, gradient flow, alpha
  clamp, and L2-normalized output.

**Open items:**

- Full Res-KAN result comparison waits on Stage 10 runs.

---

## Stage 6: FN-Weighted InfoNCE + MLP Pair Scorer

**Status:** ☑ Complete

**Gate:** FN-weighted loss and MLP scorer tests pass; training/probe path is
wired.

**Current evidence:**

- `src/losses/fn_weighted_infonce.py` and `src/models/pair_scorer.py` exist.
- `MLPPairScorer` is implemented.
- `scripts/train_fn.py` is implemented with training stability controls.
- `configs/experiment/full_mlp_fn_mlp.yaml` exists and runs 10K-step full config.
- `tests/test_fn_weighted_loss.py` and `tests/test_pair_scorer.py` exist.

**Open items:**

- Before full ablation, align `configs/loss/fn_weighted_mlp.yaml` with the
  intended `max_fn_weight=0.5` guard or document why it remains `1.0`.
- H2 is not scientifically decided until Stage 10 3-seed rows exist.

---

## Stage 7: KAN Pair Scorer

**Status:** ☑ Complete

**Gate:** KAN scorer implemented, tested, and parameter-matched enough for H3
ablation.

**Current evidence:**

- `KANPairScorer` is implemented in `src/models/pair_scorer.py`.
- `configs/model/kan_scorer.yaml` exists with parameter parity comments.
- `configs/experiment/full_mlp_fn_kan.yaml` exists.
- `tests/test_kan_pair_scorer.py` exists.

**Open items:**

- H3 is testable, but not concluded until full 3-seed ablation rows exist.

---

## Stage 7.5: Edge-Aware FN-Weighted InfoNCE

**Status:** ☑ Complete

**Gate:** Edge-aware loss/scorer/features are implemented and backward-compat
tests pass for lambda-zero behavior.

**Current evidence:**

- `src/losses/edge_features.py` and `src/losses/edge_aware_fn_loss.py` exist.
- `src/models/edge_aware_scorer.py` exists.
- `scripts/train_edge.py` is implemented.
- `configs/experiment/full_edge_off.yaml`, `full_edge_l005.yaml`, and
  `full_edge_align_l005.yaml` exist.
- `tests/test_edge_features.py` and `tests/test_edge_aware_fn_loss.py` exist.

**Open items:**

- H4 is not scientifically decided until Stage 10 full edge rows exist.

---

## Stage 8: Geometry Metrics

**Status:** ☑ Complete

**Gate:** Geometry tests pass and geometry CSV writing path exists.

**Current evidence:**

- `src/metrics/geometry.py` implements alignment, uniformity, effective rank,
  per-dimension standard deviation, and off-diagonal covariance norm.
- `src/metrics/embedding_viz.py` implements UMAP/PCA visualization output.
- `scripts/analyze_geometry.py` appends to `reports/tables/geometry.csv`.
- `tests/test_geometry.py` exists.

**Open items:**

- Full H1 table needs Stage 10 3-seed rows for MLP, KAN, and Res-KAN.

---

## Stage 9: CheXpert Full Pipeline

**Status:** ☐ Not Started

**Gate:** Deferred. Per `CLAUDE.md` Dataset Scope, CheXpert is not active for
training/probe/geometry yet.

**Current evidence:**

- Some CheXpert scaffolding/tests/configs exist in the repo, but active dataset
  routing intentionally rejects CheXpert for current training/probe/geometry
  paths.
- No CheXpert headline rows should exist in `reports/tables/probe_results.csv`.

**Open items:**

- When Stage 9 begins, integrate CheXpert as a separate active dataset branch.
- Re-run dataset leakage audit for patient-level CheXpert splits.
- Do not silently route CheXpert configs through ChestMNIST code.

---

## Stage 10: Full Ablation Runner + Paper Tables

**Status:** ☐ In Progress

**Gate:** `reports/tables/ablation_master.csv` has all 9 cells x 3 seeds, no
FAILED rows, and all four paper tables are generated.

**Current evidence:**

- `scripts/ablate.py` exists.
- `configs/experiment/ablation.yaml` has 9 cells and seeds `[42, 1337, 2024]`.
- 9 full training configs exist:
  - `full_mlp_infonce.yaml`
  - `full_kan_infonce.yaml`
  - `full_reskan_infonce.yaml`
  - `full_mlp_fn_mlp.yaml`
  - `full_mlp_fn_kan.yaml`
  - `full_reskan_fn_kan.yaml`
  - `full_edge_off.yaml`
  - `full_edge_l005.yaml`
  - `full_edge_align_l005.yaml`
- `scripts/make_paper_tables.py` exists and has guard clauses plus per-class
  rare-disease table support.
- `reports/tables/ablation_master.csv` currently has only a header row.

**Open items:**

- Run full ablation: 9 cells x 3 seeds = 27 completed rows.
- Verify zero `FAILED` rows in `reports/tables/ablation_master.csv`.
- Generate and review:
  - `reports/tables/table_h1.md`
  - `reports/tables/table_h2.md`
  - `reports/tables/table_h3.md`
  - `reports/tables/table_h4.md`
- Only after those rows exist, evaluate H1-H4.

---

## Current Next Actions

1. Fix or explicitly justify `configs/loss/fn_weighted_mlp.yaml:max_fn_weight`
   before full FN ablations.
2. Run the 27-row ChestMNIST ablation matrix.
3. Run `scripts/make_paper_tables.py` after ablation rows exist.
4. Review generated tables and record H1-H4 outcomes.
5. Keep CheXpert deferred until Stage 9 is deliberately activated.

---

## Final Status

**Overall Progress:**

```text
Completed Stages: 0, 1, 2, 3, 4, 5, 6, 7, 7.5, 8
In Progress: 10
Not Started / Deferred: 9

H1 Result: not decided - requires Stage 10 full ablation tables
H2 Result: not decided - requires Stage 10 full ablation tables
H3 Result: not decided - requires Stage 10 full ablation tables
H4 Result: not decided - requires Stage 10 full ablation tables

Thesis Status: In Progress
```

**Last Updated:** May 2026
