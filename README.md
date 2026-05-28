# KAN-FPCL: Functional Pathway Contrastive Learning for Chest X-Ray Representation

**Thesis | May 2026 | Python 3.10+**

This repository implements the **KAN-FPCL thesis**, which investigates whether **Kolmogorov-Arnold Networks (KAN)** and **false-negative-aware contrastive learning** improve chest X-ray (CXR) representation quality over standard baselines.

---

## **The Research Problem**

Chest X-ray contrastive learning faces a critical **false-negative problem**: In standard frameworks (SimCLR, MoCo), every sample not paired with an anchor is treated as a negative example. This assumption breaks down in radiography because:

1. **Different patients frequently share the same pathology.** An image from Patient A with pneumonia and an image from Patient B with pneumonia are treated as negatives, even though they should be close in latent space.

2. **Labels are often uncertain and report-derived.** Radiologist disagreement and report ambiguity make binary positive/negative classification brittle.

3. **Important findings occupy small regions.** Global pooling washes out localized pathology signals. Two images that differ globally but share disease signatures get pushed apart incorrectly.

**Hypothesis:** Using KAN projection heads with learnable univariate edge functions and a false-negative-aware loss can recover better representations by respecting the true geometry of multi-label chest X-rays.

**Novel contribution:** We extract KAN's internal edge activations φ[B, O, I] (per-edge function values) and use them as an additional training signal — a mechanism impossible in standard MLPs. This "functional pathway" signal helps detect false negatives that are invisible in embedding space alone.

---

## **Four Hypotheses**

| **#** | **Claim** | **Tested Via** | **Success Metric** |
|---|---|---|---|
| **H1** | KAN projectors learn smoother, more disentangled latent geometries than parameter-matched MLPs. | Geometry metrics (alignment, uniformity, effective rank). | KAN ≥ MLP on ≥2/3 metrics; alignment difference ≥5%. |
| **H2** | False-negative-aware contrastive masking improves recall-sensitive metrics (AUROC, rare-disease detection, mAP). | FN-weighted InfoNCE vs. standard InfoNCE with MLP scorer. | Macro-AUROC gain ≥1% absolute; rare-disease AUROC +2%. |
| **H3** | A KAN-based pair scorer outperforms an MLP scorer at matched parameter count. | FN-weighted loss with KAN scorer vs. MLP scorer. | KAN scorer AUROC ≥ MLP scorer AUROC; param parity ±15%. |
| **H4** | KAN's internal edge activations improve false-negative detection and downstream metrics beyond z-only pair scoring. | Edge-aware FN loss with edge-fingerprint similarity and edge-alignment auxiliary loss. | Edge-aware (λ=0.05) AUROC ≥ z-only (λ=0); ≥1% absolute gain. |

**Success Definition:** All four hypotheses supported with ≥3 seeds (ChestMNIST + CheXpert).

**Negative Result Path:** If H1–H3 fail, the result is still publishable if ablations isolate which component (KAN vs. FN loss) does not help and why. If H4 fails, the thesis reduces to H2+H3 (FN loss alone, no edge signals).

---

## **System Architecture**

```
ChestMNIST / CheXpert (patient-level splits, augmentations)
         ↓
    ResNet-18 Encoder → [B, 512]
         ↓
  [MLP | KAN | Residual KAN] Projector Head → z[B, D]
  (when return_edges=True: also → φ[B, O, I])
         ↓
  [MLP | KAN | EdgeAware] Pair Scorer → p_FN[B, B]
         ↓
  [InfoNCE | FN-Weighted | EdgeAware-FN] Loss
         ↓
  Train: contrastive signal, edge-pathway signal (if H4)
         ↓
  Frozen Encoder: Linear Probe + kNN Evaluation → AUROC, mAP
         ↓
  Geometry Metrics: alignment, uniformity, effective rank
         ↓
  Ablation Grid (Stages) × Seeds [42, 1337, 2024]
         ↓
  probe_results.csv → Paper Tables (H1–H4)
```

---

## **Quick Start**

### **Installation**

```bash
# Clone repo
git clone <repo-url> && cd cxr-kan-contrastive

# Create venv
python3.11 -m venv venv && source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
# torch, torchvision, hydra-core, omegaconf, medmnist, einops, 
# scikit-learn, pandas, numpy, tqdm, pytest, pytest-cov, pytorch-lightning (optional)
```

### **Run Smoke Test**

```bash
pytest tests/test_smoke.py -v
```

Expected: All imports succeed, directories exist, CLAUDE.md and AGENTS.md readable.

### **Train Stage 2 Baseline (MLP + InfoNCE)**

```bash
python scripts/train.py \
  configs/experiment/smoke_mlp.yaml \
  seed=42 \
  model.hidden_dim=64 \
  batch_size=32 \
  num_epochs=10
```

Outputs: `checkpoints/run_<uuid>/` with model, config, metrics, git hash.

### **Probe Stage 2 Checkpoint**

```bash
python scripts/probe.py \
  checkpoint=checkpoints/run_<uuid>/model.pt \
  dataset=chestmnist \
  seed=42
```

Appends row to `probe_results.csv`:
```
run_id,encoder,head,loss,scorer,dataset,seed,params_total,macro_auroc_linear,macro_auroc_knn,mAP,runtime_sec
```

### **Run Full Ablation (All Stages, 3 Seeds)**

```bash
python scripts/ablate.py --multirun seed=42,1337,2024
```

Outputs: `ablation_master.csv` with all cells × seeds.

### **Generate Paper Tables**

```bash
python scripts/make_paper_tables.py
```

Reads from `runs/results/` (ephemeral, git-ignored). Generates:
- `runs/tables/table_h1.md` — KAN vs MLP geometry metrics
- `runs/tables/table_h2.md` — FN-weighted vs InfoNCE AUROC
- `runs/tables/table_h3.md` — KAN vs MLP scorer AUROC
- `runs/tables/table_h4.md` — Edge-aware vs z-only AUROC (novel)

Copy curated tables to `reports/tables/` before committing thesis-ready outputs.

---

## **Project Structure**

```
cxr-kan-contrastive/
├── CLAUDE.md                          # Scientific rules (R1–R10)
├── AGENTS.md                          # Subagent specifications
├── README.md                          # This file
├── TODO.md                            # Stage checklist
├── pyproject.toml
├── .gitignore
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── chestmnist.py              # ChestMNIST dataset wrapper
│   │   ├── chexpert.py                # CheXpert dataset wrapper (Stage 9)
│   │   ├── splits.py                  # Patient-level split logic (R4, R5)
│   │   └── augmentations.py           # TwoViewTransform (SimCLR-style)
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoder.py                 # ResNet-18 backbone
│   │   ├── mlp_head.py                # 2-layer MLP projector
│   │   ├── pair_scorer.py             # MLP/KAN/EdgeAware pair scorer
│   │   │
│   │   └── kan/
│   │       ├── __init__.py
│   │       ├── fastkan.py             # FastKAN layer + projector (Stage 4)
│   │       └── residual_warp.py       # Residual KAN warp (Stage 5)
│   │
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── masks.py                   # Positive-pair mask builder (R3)
│   │   ├── infonce.py                 # Standard InfoNCE (Stage 2)
│   │   ├── fn_weighted_infonce.py     # FN-weighted loss (Stage 6)
│   │   ├── edge_features.py           # Edge fingerprint projection (Stage 7.5)
│   │   └── edge_aware_fn_loss.py      # Edge-aware FN loss (Stage 7.5)
│   │
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── linear_probe.py            # sklearn LogisticRegression probe
│   │   ├── knn.py                     # kNN classifier
│   │   ├── auc.py                     # Multi-label AUROC computer (R3)
│   │   ├── geometry.py                # Alignment, uniformity, effective rank (Stage 8)
│   │   └── embedding_viz.py           # UMAP/PCA visualizer
│   │
│   └── utils/
│       ├── __init__.py
│       ├── param_count.py             # Count model parameters
│       ├── checkpoint.py              # Save/load artifacts (R8)
│       └── reproducibility.py         # set_seed() function
│
├── scripts/
│   ├── train.py                       # Hydra training entry-point (all stages)
│   ├── probe.py                       # Frozen-encoder downstream eval (Stage 3+)
│   ├── analyze_geometry.py            # Geometry metrics on checkpoint
│   ├── ablate.py                      # Ablation runner (Stage 10)
│   └── make_paper_tables.py           # Generate H1–H4 tables
│
├── configs/
│   ├── data/
│   │   ├── chestmnist.yaml
│   │   └── chexpert.yaml
│   │
│   ├── model/
│   │   ├── mlp_head.yaml
│   │   ├── kan_head.yaml
│   │   ├── residual_fastkan_warp.yaml
│   │   └── encoder.yaml
│   │
│   ├── loss/
│   │   ├── infonce.yaml
│   │   ├── fn_weighted_mlp.yaml
│   │   ├── edge_aware_fn_mlp.yaml
│   │   └── edge_aware_fn_kan.yaml
│   │
│   └── experiment/
│       ├── smoke_mlp.yaml             # Quick smoke test
│       ├── smoke_kan.yaml
│       ├── smoke_edge_aware.yaml
│       ├── baseline_mlp.yaml
│       ├── baseline_kan.yaml
│       └── probe.yaml
│
├── tests/
│   ├── test_smoke.py                  # Import + file existence
│   ├── test_data_splits.py            # Patient disjointness (R4, R5)
│   ├── test_data_pipeline.py          # Dataset shapes, labels
│   ├── test_infonce.py                # Diagonal mask, positives, negatives (R3)
│   ├── test_mlp_head.py               # Shape, L2-norm, param count (R1)
│   ├── test_fn_weighted_loss.py       # FN recovery, NaN handling (R3)
│   ├── test_pair_scorer.py            # Shape, bounds, gradient
│   ├── test_fastkan.py                # FastKAN layer, return_edges (Stage 4)
│   ├── test_residual_kan.py           # Identity-at-init, alpha clamp (Stage 5)
│   ├── test_edge_aware_fn_loss.py     # Backward compat, edge features (Stage 7.5)
│   ├── test_geometry.py               # Alignment, uniformity, effective rank
│   ├── test_metrics.py                # Linear probe, kNN, AUROC on synthetic
│   └── test_chexpert.py               # CheXpert loader, uncertainty policy
│
├── runs/                               # Ephemeral artifacts (git-ignored)
│   ├── results/                       # Cumulative CSVs from probe/ablate
│   │   ├── probe_results.csv
│   │   ├── ablation_master.csv
│   │   └── geometry.csv
│   ├── tables/                        # Auto-generated markdown tables
│   │   └── table_h{1,2,3,4}.md
│   └── figures/                       # Auto-generated plots
│
├── reports/                           # Curated thesis outputs (git-committed)
│   ├── figures/                       # Hand-selected plots for thesis
│   └── tables/                        # Final paper tables (copied from runs/tables/)
│
├── .claude/
│   ├── agents/
│   │   ├── loss-auditor.md
│   │   ├── dataset-leakage-checker.md
│   │   ├── experiment-auditor.md
│   │   ├── pytorch-debugger.md
│   │   └── code-reviewer.md
│   │
│   ├── commands/
│   │   └── review-stage.py
│   │
│   └── skills/
│       ├── contrastive-loss-engineer.md
│       ├── cxr-dataset-pipeline.md
│       └── experiment-config-hydra.md
│
└── .git/                               # Version control
```

---

## **Implementation Stages (0–10)**

See the `KAN_FPCL_Reimplementation_Playbook.docx` for full prompts. Quick summary:

| **Stage** | **What** | **Gate** | **Subagent** |
|---|---|---|---|
| **0** | Repo bootstrap, CLAUDE.md, AGENTS.md, TODO.md | pytest test_smoke.py | code-reviewer |
| **1** | ChestMNIST splits, TwoView augmentation | tests/test_data_*.py green | dataset-leakage-checker |
| **2** | MLP + InfoNCE baseline (critical!) | AUROC in probe_results.csv | loss-auditor, pytorch-debugger |
| **3** | Linear probe + kNN evaluation (GATE) | Baseline row committed | experiment-auditor |
| **4** | FastKAN projector, return_edges support | Geometry metrics | loss-auditor |
| **5** | Residual KAN warp, identity-at-init | Geometry metrics improve | loss-auditor |
| **6** | FN-weighted loss + MLP scorer | H2 test gate | loss-auditor, pytorch-debugger |
| **7** | KAN pair scorer (swap MLP in Stage 6) | H3 test gate | loss-auditor |
| **7.5** | Edge-aware FN loss (KAN-FPCL novel) | H4 test gate | loss-auditor, pytorch-debugger |
| **8** | Geometry metrics (alignment, etc.) | Metrics complete | code-reviewer |
| **9** | CheXpert full pipeline | CheXpert baseline committed | dataset-leakage-checker |
| **10** | Ablation runner + paper tables | All tables generated | experiment-auditor |

**Critical Gate: Stage 3.** No later stage may proceed until AUROC numbers are produced. This isolates whether improvements come from better loss/architecture or broken evaluation.

---

## **Evaluation Pipeline**

### **Linear Probe**
Frozen encoder → sklearn LogisticRegression on train embeddings → eval on val/test.
**Metrics:** macro-AUROC, micro-AUROC, per-class AUROC, mAP (mean average precision).

### **kNN Classifier**
Frozen encoder → cosine-nearest-neighbors with k=20 → weighted vote.
**Metrics:** Same as above.

### **Geometry Metrics** (Stage 8)
On frozen eval embeddings:
- **Alignment:** MSE between positive-pair embeddings.
- **Uniformity:** Expected distance between random samples on unit sphere (should be uniform).
- **Effective Rank:** SVD entropy → exp(entropy).
- **Per-dimension std:** Should be similar across dims (disentangled).
- **Off-diagonal covariance norm:** Low = no spurious correlations.

### **Ablation Grid** (Stage 10)

| Cell ID | Head | Loss | Scorer | λ_edge | λ_align | Hypothesis |
|---|---|---|---|---|---|---|
| mlp_infonce | MLP | InfoNCE | — | — | — | H1 baseline |
| kan_infonce | FastKAN | InfoNCE | — | — | — | H1 |
| reskan_infonce | Res-KAN | InfoNCE | — | — | — | H1 |
| mlp_fn_mlp | MLP | FN-weighted | MLP | — | — | H2 |
| mlp_fn_kan | MLP | FN-weighted | KAN | — | — | H3 |
| kan_fn_kan | FastKAN | FN-weighted | KAN | — | — | H3+H1 |
| zonly_fn | FastKAN | EdgeAware-FN | MLP (z-only) | 0.0 | 0.0 | H4 baseline |
| edge_scorer_no_aux | FastKAN | EdgeAware-FN | EdgeAware-MLP | 0.0 | 0.0 | H4 control |
| edge_contrastive | FastKAN | EdgeAware-FN | EdgeAware-MLP | 0.05 | 0.0 | H4 treatment A |
| edge_align | FastKAN | EdgeAware-FN | EdgeAware-MLP | 0.0 | 0.05 | H4 treatment B |
| edge_contrastive_kan | FastKAN | EdgeAware-FN | EdgeAware-KAN | 0.05 | 0.0 | H3 cross-check |

**3 seeds (default):** [42, 1337, 2024] → 33 rows total
**5 seeds (full thesis):** [42, 1337, 2024, 7, 9001] → 55 rows total

---

## **Scientific Rules (TL;DR)**

1. **R1:** Every KAN result pairs with ±15% parameter-matched MLP.
2. **R2:** Baseline losses pass tests before combined stages.
3. **R3:** All masks tested (pos/neg/false-negative cases).
4. **R4:** Patient-level splits, not row-level.
5. **R5:** Disjoint train/val/test patient sets.
6. **R6:** Config-driven experiments (Hydra, no magic numbers).
7. **R7:** Losses return dicts with named components.
8. **R8:** Every run saves config, checkpoint, metrics, git hash.
9. **R9:** Descriptive errors, no silent fallbacks.
10. **R10:** Modules ≤200 lines; split if larger.

See `CLAUDE.md` for full rules.

---

## **Subagent Audit Commands**

```bash
# After implementing a loss
/run-subagent loss-auditor

# After implementing a dataset loader
/run-subagent dataset-leakage-checker

# Before training
/run-subagent pytorch-debugger

# Before commit
/run-subagent code-reviewer

# Before marking stage complete
/run-subagent experiment-auditor
```

See `AGENTS.md` for specifications.

---

## **Key Files for Each Hypothesis**

| **Hypothesis** | **Key Files** | **Success Condition** |
|---|---|---|
| **H1** | src/models/kan/, tests/test_geometry.py, scripts/analyze_geometry.py | KAN alignment +5%, uniformity, rank match or beat MLP |
| **H2** | src/losses/fn_weighted_infonce.py, tests/test_fn_weighted_loss.py | AUROC gain ≥1% absolute over InfoNCE |
| **H3** | src/models/pair_scorer.py (KAN variant), tests/test_kan_pair_scorer.py | KAN scorer AUROC ≥ MLP scorer AUROC, params ±15% |
| **H4** | src/losses/edge_aware_fn_loss.py, src/losses/edge_features.py, Stage 7.5 tests | Edge-aware (λ=0.05) ≥ z-only (λ=0), ≥1% gain |

---

## **Common Pitfalls**

1. **Skipping Stage 3.** No later result is verifiable without AUROC.
2. **Not testing FN masks.** False-negative logic bugs are silent.
3. **Parameter mismatch.** Can't claim KAN is better if KAN has 2× params.
4. **Single seed claims.** Always report mean ± std over ≥3 seeds.
5. **log(1 − p_FN) NaN.** Use `clamp_min(1e-10)` before `.log()`.
6. **Patient leakage.** Same patient in train and val inflates AUROC.
7. **Labels in pair scorer.** Scorer must be label-free (unless oracle ablation).
8. **ChestMNIST as headline.** Headlines from CheXpert only.

See `CLAUDE.md` section "Critical Pitfalls" for details and fixes.

---

## **Citation**

If you use code or results from this project:

```bibtex
@thesis{kan_fpcl_2026,
  title={KAN-FPCL: Functional Pathway Contrastive Learning for False-Negative-Aware 
         Multi-Label Chest X-Ray Representation},
  author={[Author]},
  year={2026},
  school={[University]},
  url={https://github.com/[repo-url]}
}
```

---

## **Contact & Issues**

For questions about implementation, rules, or subagent behavior:
- Review `CLAUDE.md` (scientific rules)
- Review `AGENTS.md` (subagent checklists)
- Check `TODO.md` for stage status
- Run `/run-subagent <name>` to diagnose

---

## **License**

[To be specified in LICENSE file]

---

**Last Updated:** May 2026  
**Current Stage:** 10 (Ablation — configs ready, full runs not yet executed)  
**Status:** Code-ready for all hypotheses; awaiting `scripts/ablate.py` execution
