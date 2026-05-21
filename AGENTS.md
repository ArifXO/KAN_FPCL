# AGENTS.md — Subagent Specifications for KAN-FPCL Thesis

This document defines 5 subagents that audit code for compliance with CLAUDE.md rules (R1–R10) and thesis hypotheses (H1–H4). Invoke these after implementing each stage.

---

## **1. loss-auditor Subagent**

**Purpose:** Audit contrastive losses and pair scorers for correctness and compliance with R1, R3, R7, R9.

**Activation:** After implementing or modifying `src/losses/*.py` or `src/models/pair_scorer.py`.

**Checklist:**

- [ ] **R1 Param Parity:** If two variants exist (MLP vs KAN, or MLP vs residual), report parameter counts. Fail if not within ±15%.
- [ ] **R3 Mask Tests:** Read `tests/test_*_loss.py`. Verify unit tests exist for:
  - Positive-only batch (all pairs are positives).
  - Negative-only batch (all pairs are negatives).
  - False-negative case (true positives manually marked as negatives).
  - All tests pass.
- [ ] **R7 Dict Returns:** Inspect loss.forward() return type. Must return `dict[str, Tensor]`, not scalar. Verify keys:
  - `"loss"` (total)
  - `"temperature"` or `"tau"` (hyperparams)
  - `"pos_sim_mean"`, `"neg_sim_mean"` (debug)
  - Any stage-specific keys (e.g., `"p_fn_mean"`, `"edge_contrastive_loss"`).
- [ ] **R9 Error Handling:** Search for bare `except:`, `try: ... except: pass`, or silent return None. Fail if found.
  - Verify ValueError raised on invalid inputs (temperature ≤ 0, p_fn outside [0, 1], etc.).
  - Errors must include descriptive context (not generic).
- [ ] **Numerical Stability:**
  - For FN-weighted loss: check `clamp_min(1e-10)` before `.log()`.
  - For edge fingerprint: confirm compression from O×I to 256 dimensions.
  - No `torch.where()` without clamping the true branch.
- [ ] **Gradient Flow:** Verify all learnable params have gradients after backward. Special attention to:
  - KAN edge weights.
  - Pair scorer params.
  - Any learned projection matrices.

**Output Format:**
```
LOSS-AUDITOR REPORT
Stage: <N>
Files Checked: <list>

PASS: R1 Param Parity ✓
  - MLP variant: 1089 params
  - KAN variant: 1193 params
  - Difference: +9.5% (within ±15%) ✓

PASS: R3 Unit Tests ✓
  - test_infonce.py::test_diagonal_excluded: PASS
  - test_infonce.py::test_pos_only_batch: PASS
  - test_infonce.py::test_neg_only_batch: PASS
  - test_infonce.py::test_fn_marked_negatives: PASS (FN-specific)

PASS: R7 Dict Return ✓
  - Keys: loss, pos_sim_mean, neg_sim_mean, temperature ✓
  - All values are Tensor ✓

PASS: R9 Error Handling ✓
  - No bare except ✓
  - ValueError on temperature ≤ 0 ✓
  - Descriptive context in error messages ✓

PASS: Numerical Stability ✓
  - clamp_min(1e-10) before .log() in FN loss ✓
  - Edge fingerprint compressed to 256-d ✓

PASS: Gradient Flow ✓
  - encoder.conv layers: grad ✓
  - mlp_head: grad ✓
  - pair_scorer: grad ✓

Overall: PASS (0 FAILs)
```

**Fail Conditions:**
- Parameter count mismatch > ±15% → FAIL
- Missing unit test (pos/neg/FN case) → FAIL
- Dict return missing required keys → FAIL
- Bare except or silent failures → FAIL
- NaN/inf in gradient → FAIL

---

## **2. dataset-leakage-checker Subagent**

**Purpose:** Audit dataset loaders for compliance with R4 (patient-level splits) and R5 (disjoint split sets).

**Activation:** After implementing or modifying `src/data/*.py`.

**Checklist:**

- [ ] **R4 Patient-Level Split:** 
  - Inspect `src/data/splits.py::patient_level_split()`.
  - Confirm function receives patient IDs, not row indices.
  - For ChestMNIST: patient ID = image ID (unique per sample).
  - For CheXpert: patient ID extracted from filename prefix or explicit column.
  - Fail if using random row-level split (e.g., `train_test_split(df, test_size=0.2)`).

- [ ] **R5 Disjointness:**
  - Verify test in `tests/test_data_splits.py`:
    ```python
    def test_patient_level_disjointness():
        train_ids, val_ids, test_ids = patient_level_split(...)
        assert len(train_ids & val_ids) == 0
        assert len(val_ids & test_ids) == 0
        assert len(train_ids & test_ids) == 0
    ```
  - Run on synthetic data and real dataset sample.
  - Fail if any overlap detected.

- [ ] **R5 Visualization:**
  - Report breakdown: "ChestMNIST train: 10K images from 5K patients; val: 1.5K images from 750 patients; test: 1K from 500 patients."
  - Verify no patient appears in multiple splits.

- [ ] **CheXpert-Specific:**
  - Check `uncertainty_policy` config (must be explicit: 'ignore', 'positive', 'negative', 'lsr').
  - Fail if hardcoded policy or silent default.
  - Verify uncertain labels handled consistently.

- [ ] **Leakage Scenarios (Test Coverage):**
  - Same patient different views → train/val split → caught by disjointness test.
  - Data augmentation leakage (train aug params visible to val) → not this subagent's scope, but flag if found.

**Output Format:**
```
DATASET-LEAKAGE-CHECKER REPORT
Stage: <N>
Files Checked: <list>

PASS: R4 Patient-Level Split ✓
  - ChestMNIST splits by image ID (unique per sample) ✓
  - CheXpert splits by subject ID prefix ✓
  - patient_level_split() function signature verified ✓

PASS: R5 Disjointness ✓
  - test_data_splits.py::test_patient_disjointness: PASS
  - Synthetic data (n_patients=100): train/val/test disjoint ✓
  - ChestMNIST real sample (n=5000): train/val/test disjoint ✓
  - No patient in >1 split ✓

PASS: Split Breakdown ✓
  - ChestMNIST: train 10K/5K patients, val 1.5K/750, test 1K/500
  - CheXpert: train 45K/11.2K patients, val 10K/2.5K, test 12K/3K

PASS: CheXpert Uncertainty Handling ✓
  - uncertainty_policy in config (value: 'ignore') ✓
  - No hardcoded policy fallback ✓
  - ValueError raised for invalid policies ✓

Overall: PASS (0 FAILs)
```

**Fail Conditions:**
- Patient-level split not verified → FAIL
- Test overlap detected (train & val share patients) → FAIL
- Hardcoded or silent uncertainty policy → FAIL
- Breakdown missing or incorrect → FAIL

---

## **3. experiment-auditor Subagent**

**Purpose:** Audit experiment scripts and configs for compliance with R6 (Hydra) and R8 (artifact saving).

**Activation:** Before marking a stage complete; before any commit of training/evaluation code.

**Checklist:**

- [ ] **R6 Hydra Config-Driven:**
  - Verify `scripts/train.py` or eval script decorated with `@hydra.main(config_path=..., config_name=...)`.
  - Check no hardcoded hyperparams (learning_rate, batch_size, etc.). All must be in config.
  - Verify `cfg.model.hidden_dim`, not magic numbers like `hidden_dim = 64`.
  - Hydra multirun: `python train.py --multirun seed=42,1337,2024`.

- [ ] **R8 Artifact Saving:**
  - After training, verify code saves:
    - [ ] Checkpoint: `checkpoints/run_{uuid}/model.pt`
    - [ ] Config: `checkpoints/run_{uuid}/config.yaml` (via `OmegaConf.to_yaml()`)
    - [ ] Metrics: `checkpoints/run_{uuid}/metrics.json` (train_loss, train_time, etc.)
    - [ ] Param count: `checkpoints/run_{uuid}/param_count.txt`
    - [ ] Git info: `checkpoints/run_{uuid}/git_info.txt` (commit hash + branch)
  - Verify `run_uuid` is generated (timestamp or UUID), not hardcoded by user.

- [ ] **Probe Script (Stage 3+):**
  - `scripts/probe.py` reads checkpoint, extracts frozen embeddings, runs probe + kNN.
  - Appends row to `probe_results.csv` with columns:
    - `run_id, encoder, head, loss, scorer, dataset, seed, params_total, macro_auroc_linear, macro_auroc_knn, mAP, runtime_sec`
  - Verify no duplicate rows (same run_id, seed, dataset).

- [ ] **Ablation Runner (Stage 10):**
  - `scripts/ablate.py` uses Hydra --multirun to sweep ablation matrix.
  - For each cell × seed: train → probe → analyze_geometry.
  - On error: mark FAILED in CSV with exception, continue (don't halt).
  - Columns: `cell_id, head, loss, scorer, lambda_edge, lambda_edge_align, dataset, seed, params_total, macro_auroc_linear, macro_auroc_knn, mAP, alignment, uniformity, effective_rank, runtime_sec`

- [ ] **Hyperparameter Curriculum:**
  - If stage involves curriculum (e.g., λ_edge ramping from 0 to 0.05), verify:
    - Curriculum defined in config (not hardcoded).
    - Logged at every epoch.
    - Test case for λ_edge=0 → equivalence to previous stage.

- [ ] **Reproducibility:**
  - `set_seed(cfg.seed)` called before any data/model ops.
  - Git hash + branch saved before training.
  - Config hash (for detecting config changes) computed.

**Output Format:**
```
EXPERIMENT-AUDITOR REPORT
Stage: <N>
Script: <scripts/train.py or scripts/probe.py>

PASS: R6 Hydra Config-Driven ✓
  - Decorator: @hydra.main(config_path="configs", config_name="smoke_mlp") ✓
  - No hardcoded hyperparams ✓
  - Usage: python train.py seed=42 model.hidden_dim=128 ✓

PASS: R8 Artifact Saving ✓
  - Checkpoint saved: checkpoints/run_uuid_20260521_153044/model.pt ✓
  - Config saved: config.yaml (OmegaConf.to_yaml) ✓
  - Metrics saved: metrics.json (keys: train_loss, train_time) ✓
  - Param count saved: param_count.txt (encoder: 23.5M, head: 156K, total: 23.7M) ✓
  - Git info saved: git_info.txt (commit abc1234 on branch main) ✓

PASS: Probe Script (Stage 3+) ✓
  - Reads checkpoint ✓
  - Extracts frozen embeddings ✓
  - Computes linear probe + kNN AUROC ✓
  - Appends to probe_results.csv with all required columns ✓
  - No duplicate rows ✓

PASS: Reproducibility ✓
  - set_seed(cfg.seed) called at line 42 ✓
  - Git hash saved before training ✓
  - Deterministic: no random file system ops ✓

Overall: PASS (0 FAILs)
```

**Fail Conditions:**
- No Hydra decorator or wrong config path → FAIL
- Hardcoded learning_rate, batch_size, etc. → FAIL
- Missing artifact (checkpoint, config, metrics, git hash) → FAIL
- Probe CSV columns incomplete or wrong order → FAIL
- run_uuid hardcoded or user-input → FAIL

---

## **4. pytorch-debugger Subagent**

**Purpose:** Debug tensor shape mismatches, NaN propagation, and gradient flow issues after loss implementation.

**Activation:** After loss.py or projector.py changes; before first training run.

**Procedure:**

1. **Create synthetic batch:** 
   ```python
   B, D = 4, 64
   z = torch.randn(2*B, D)  # Contrastive: doubled batch
   z_norm = F.normalize(z, dim=-1)
   targets = torch.arange(B).repeat(2)  # Positive pairs
   ```

2. **Run forward pass:**
   ```python
   loss_dict = loss_fn(z_norm)
   print(f"Loss keys: {loss_dict.keys()}")
   print(f"Loss value: {loss_dict['loss'].item()}")
   ```

3. **Check finite:**
   ```python
   assert torch.isfinite(loss_dict['loss']), "NaN in loss"
   ```

4. **Run backward:**
   ```python
   loss_dict['loss'].backward()
   for name, param in model.named_parameters():
       if param.grad is None:
           print(f"WARNING: {name} has no gradient")
       else:
           assert torch.isfinite(param.grad).all(), f"NaN in {name}.grad"
   ```

5. **Report:**
   - All params have finite gradients → PASS
   - Any param with NaN grad → FAIL (print param name and grad value)
   - Any param with zero grad (but not in frozen layers) → WARNING

**Output Format:**
```
PYTORCH-DEBUGGER REPORT
Module: src/losses/fn_weighted_infonce.py

Synthetic Batch: [8, 64] (B=4, D=64)

PASS: Forward Pass ✓
  - Loss keys: ['loss', 'pos_sim_mean', 'neg_sim_mean', 'p_fn_mean', 'temperature']
  - Loss value: 0.5432 (finite) ✓

PASS: Backward Pass ✓
  - encoder.conv1.weight: grad finite ✓
  - mlp_head.linear1.weight: grad finite ✓
  - pair_scorer.fc1.weight: grad finite ✓
  - All 127 params have gradients ✓

PASS: NaN Check ✓
  - No NaN in any grad ✓
  - Max grad norm: 0.1234 (reasonable) ✓

Overall: PASS (0 FAILs)
```

**Fail Conditions:**
- NaN in loss → FAIL
- NaN in any grad → FAIL
- Param with zero grad (not frozen) → FLAG (warn, not fail unless systematic)

---

## **5. code-reviewer Subagent**

**Purpose:** General code quality check for R10 (module line limits) and PEP 8 compliance.

**Activation:** Before any merge or stage gate.

**Checklist:**

- [ ] **R10 Module Line Limit:**
  - Report line counts for all files in `src/{models,losses,metrics,data}/`.
  - Fail if any file > 200 lines (non-test).
  - If split is needed, suggest breakpoints.

- [ ] **PEP 8 / Style:**
  - Variable names: `snake_case`, not `camelCase`.
  - No unused imports.
  - Docstrings on public functions.
  - Type hints on function signatures.

- [ ] **Imports:**
  - No circular imports.
  - No wildcard imports (`from module import *`).
  - Standard library, third-party, local imports in that order.

- [ ] **Dead Code:**
  - Unused variables flagged.
  - Commented-out code removed.

- [ ] **Comments:**
  - Functions > 5 lines have docstrings.
  - Numerical stability concerns documented (e.g., "clamp_min(1e-10) to avoid log(0)").

**Output Format:**
```
CODE-REVIEWER REPORT
Files Changed: <list>

PASS: R10 Module Line Limits ✓
  - src/losses/infonce.py: 85 lines ✓
  - src/losses/fn_weighted_infonce.py: 142 lines ✓
  - src/models/pair_scorer.py: 198 lines ✓

PASS: PEP 8 ✓
  - snake_case variables ✓
  - No unused imports ✓
  - Docstrings on public functions ✓

PASS: Imports ✓
  - No circular imports ✓
  - No wildcard imports ✓
  - Correct order (stdlib, third-party, local) ✓

PASS: Dead Code ✓
  - No unused variables ✓
  - No commented-out code ✓

PASS: Comments ✓
  - All functions > 5 lines have docstrings ✓
  - Numerical stability documented ✓

Overall: PASS (0 FAILs)
```

**Fail Conditions:**
- Any file > 200 lines → FAIL
- Circular imports → FAIL
- Wildcard imports → FAIL
- Missing docstrings on public functions → FLAG (warn)

---

## **Subagent Invocation Workflow**

### **Stage 2 (InfoNCE Baseline)**
1. Implement `src/losses/infonce.py` + tests.
2. `/run-subagent loss-auditor` → must PASS before proceeding.
3. `/run-subagent pytorch-debugger` on synthetic batch → must PASS before first train run.

### **Stage 3 (Linear Probe)**
1. Implement `scripts/probe.py`.
2. `/run-subagent experiment-auditor` → must PASS.
3. Train Stage 2 checkpoint, run probe, commit row to `probe_results.csv`.

### **Stage 6 (FN-Weighted Loss)**
1. Implement `src/losses/fn_weighted_infonce.py` + `src/models/pair_scorer.py`.
2. `/run-subagent loss-auditor` → must PASS (especially R3 FN unit test).
3. `/run-subagent pytorch-debugger` → must PASS.
4. Train, probe, append row.

### **Stage 9 (CheXpert)**
1. Implement `src/data/chexpert.py`.
2. `/run-subagent dataset-leakage-checker` → must PASS.
3. `/run-subagent code-reviewer` → must PASS R10.
4. Train MLP baseline on CheXpert, probe, append row.

### **Stage 10 (Ablation + Final Review)**
1. Implement `scripts/ablate.py` + `scripts/make_paper_tables.py`.
2. `/run-subagent experiment-auditor` on ablate.py → must PASS.
3. `/run-subagent code-reviewer` on all changed files → must PASS.
4. Run full ablation suite across 3 seeds (5 if full).
5. Generate paper tables from `ablation_master.csv`.

---

## **Subagent Decision Tree**

```
Stage Complete?
├─ Loss/Scorer Changed?
│  └─ YES → /run-subagent loss-auditor → FAIL? Stop. FIX. → PASS → Continue.
├─ Dataset Changed?
│  └─ YES → /run-subagent dataset-leakage-checker → FAIL? Stop. FIX. → PASS → Continue.
├─ Train/Eval Script Changed?
│  └─ YES → /run-subagent experiment-auditor → FAIL? Stop. FIX. → PASS → Continue.
├─ First Train of New Loss/Projector?
│  └─ YES → /run-subagent pytorch-debugger → FAIL? Stop. DEBUG. → PASS → Continue.
└─ About to Commit?
   └─ YES → /run-subagent code-reviewer → FAIL? Stop. FIX. → PASS → Commit.
```

---

## **Metadata for Each Subagent Run**

Log each invocation with:
- Date/time
- Stage number
- Files checked
- PASS/FAIL status
- If FAIL, issue list + suggested fixes

Commit log to `reports/subagent_audits.log`.

