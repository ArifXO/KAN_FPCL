# TODO — cxr-kan-contrastive

All items unchecked. Work in order of dependencies.

---

## Stage 1 — Data Pipeline

- [ ] Implement `ChestMNISTDataset` wrapper around MedMNIST ChestMNIST
- [ ] Implement patient-level train/val/test split (R4)
- [ ] Assert split disjointness programmatically (R5)
- [ ] Implement data augmentation pipeline (random crop, horizontal flip, color jitter)
- [ ] Implement contrastive pair sampler (returns two augmented views per image)
- [ ] Write unit tests for contrastive mask: positive pairs (test_mask_positives)
- [ ] Write unit tests for contrastive mask: negative pairs (test_mask_negatives)
- [ ] Write unit tests for contrastive mask: false-negative exclusion (test_mask_fn_exclusion)
- [ ] Add `configs/data/chestmnist.yaml`

## Stage 2 — Baseline Models (must precede KAN work per R2)

- [ ] Implement MLP encoder (`src/models/mlp_encoder.py`)
- [ ] Implement projection head (shared architecture for MLP and KAN)
- [ ] Write smoke test for MLP forward pass with random input
- [ ] Count and log MLP parameter count
- [ ] Add `configs/model/mlp_encoder.yaml`

## Stage 3 — Loss Functions (must precede experiment wiring per R2)

- [ ] Implement NT-Xent (SimCLR) loss returning `dict[str, Tensor]` (R7)
- [ ] Implement Supervised Contrastive loss returning `dict[str, Tensor]` (R7)
- [ ] Implement label-based false-negative exclusion mask
- [ ] Implement embedding-similarity false-negative exclusion mask
- [ ] Write unit tests for NT-Xent loss (positive/negative/FN cases) (R3)
- [ ] Write unit tests for SupCon loss (positive/negative/FN cases) (R3)
- [ ] Add `configs/loss/ntxent.yaml` and `configs/loss/supcon.yaml`

## Stage 4 — KAN Encoder

- [ ] Implement KAN layer (spline-based univariate functions + linear combination)
- [ ] Implement KAN encoder (`src/models/kan/kan_encoder.py`)
- [ ] Match KAN parameter count to MLP baseline (R1)
- [ ] Write smoke test for KAN forward pass with random input
- [ ] Add `configs/model/kan_encoder.yaml`

## Stage 5 — Training Infrastructure

- [ ] Implement training loop (`scripts/train.py`) with Hydra config (R6)
- [ ] Implement artifact saving: config YAML, git hash, metrics JSON, param count, runtime (R8)
- [ ] Implement linear probe evaluation script (`scripts/eval_linear_probe.py`)
- [ ] Implement k-NN evaluation script (`scripts/eval_knn.py`)
- [ ] Add `configs/experiment/baseline_mlp_ntxent.yaml`
- [ ] Add `configs/experiment/baseline_mlp_supcon.yaml`
- [ ] Add `configs/experiment/kan_ntxent.yaml`
- [ ] Add `configs/experiment/kan_supcon.yaml`

## Stage 6 — Metrics and Evaluation

- [ ] Implement AUC-ROC per class (ChestMNIST is multi-label)
- [ ] Implement mean AUC across all 14 pathology labels
- [ ] Implement k-NN accuracy metric
- [ ] Implement linear probe accuracy metric
- [ ] Add label-efficiency sweep (10%, 25%, 50%, 100% labeled) for H2

## Stage 7 — Interpretability (H3)

- [ ] Implement spline visualization utility for KAN layers
- [ ] Generate per-neuron activation plots for trained KAN encoder
- [ ] Annotate top-5 splines with candidate radiological feature descriptions

## Stage 8 — Reporting

- [ ] Produce Table 1: AUC comparison MLP vs KAN across loss variants
- [ ] Produce Table 2: Label-efficiency results (H2)
- [ ] Produce Figure 1: t-SNE of MLP vs KAN representations
- [ ] Produce Figure 2: Spline activation visualizations (H3)
- [ ] Produce Figure 3: FN exclusion strategy ablation (H4)
- [ ] Write paper draft sections: Method, Experiments, Results
