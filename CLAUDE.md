# CLAUDE.md — KAN-FPCL Thesis Scientific Rules & Agent Guidelines

This document defines 10 inviolable scientific rules and agent configuration for Claude Code work on the KAN-FPCL thesis. Every module, loss, and dataset handler must comply.

---

## **10 Scientific Rules (R1–R10)**

### **R1: Every KAN Result Pairs with Parameter-Matched MLP Baseline**
- When implementing a KAN projector, MLP projector, KAN scorer, or MLP scorer, parameter counts must be within ±15%.
- Print parameter counts in config files as comments.
- Do NOT compare different parameter budgets without explicit "parameter ablation" framing.
- **Why:** A KAN win with 2× the parameters is not a win for KAN — it's a win for capacity. Hypothesis H1 requires architectural parity.

### **R2: Do NOT Implement Combined Model Before Baseline Losses + Tests Pass**
- Stage 2 (InfoNCE + MLP) must have all tests green and probe results committed before Stage 4 (FastKAN) begins.
- Stage 6 (FN-weighted loss + MLP scorer) must have all tests green before Stage 7 (KAN scorer) begins.
- **Why:** If downstream stage fails, you cannot isolate whether the failure is in your code or in a broken dependency.

### **R3: All Contrastive Masks Must Have Unit Tests (Positive/Negative/FN Cases)**
- Every loss function producing a mask or weighting the loss must test:
  - **Positive-only batch:** all pairs are positives → loss near zero.
  - **Negative-only batch:** all pairs are negatives → loss finite and > 0.
  - **False-negative case:** deliberately mark a true positive as negative in the mask → loss increases compared to unmarked case.
- Test on synthetic batch [B=4, D=8], not just on random tensors.
- **Why:** Contrastive losses silently converge to zero with broken masks. Unit tests catch this before experiment weeks.

### **R4: Dataset Splits Must Be Patient-Level Where Patient IDs Exist**
- In ChestMNIST: unique image ID is patient-level.
- In CheXpert: patient ID is derived from the DICOM filename prefix or explicit column.
- **Do NOT** use random row-level splits. This leaks the same patient across train and test, inflating AUROC by 5–15%.
- **Why:** The thesis makes claims about representation quality. If the same patient appears in train and val, the claim is void.

### **R5: No Data Leakage. Train/Val/Test Patient Sets Must Be Disjoint**
- After patient-level split, verify disjointness:
  ```python
  assert len(train_ids & val_ids) == 0
  assert len(val_ids & test_ids) == 0
  assert len(train_ids & test_ids) == 0
  ```
- Raise `ValueError` if any overlap detected.
- Document split code with patient counts (e.g., "Train: 10K images from 2.5K patients").
- **Why:** Overlapping splits corrupt every downstream metric and make negative results unpublishable.

### **R6: Experiments Config-Driven (Hydra). No Hardcoded Hyperparams**
- Every run must be configurable via YAML: learning rate, batch size, model dims, loss lambdas, augmentation strength, number of epochs.
- Hardcoded magic numbers are forbidden. Use `cfg.model.hidden_dim`, not `hidden_dim = 64`.
- Config files live in `configs/{data,model,loss,experiment}/` and are committed.
- **Why:** Replicability requires showing exactly what hyperparams you used. Hydra + committed YAMLs enable that.

### **R7: Every Loss Returns `dict[str, Tensor]` With Named Components**
- Do NOT return a scalar loss. Return:
  ```python
  {
    "loss": scalar_tensor,              # Total loss
    "info_nce_component": scalar,       # Named parts
    "fn_weighting_component": scalar,   # ...
    "temperature": scalar_or_float,     # Hyperparams for logging
    "pos_sim_mean": scalar,             # Debug metrics
    "neg_sim_mean": scalar,
    ... (more keys for diagnosis)
  }
  ```
- Tests must verify all keys are present and named correctly.
- **Why:** When loss doesn't converge, you need to know which component is at fault. Dict returns provide that visibility.

### **R8: Every Run Saves Artifact Set (Config YAML, Git Hash, Metrics JSON, Param Count, Runtime)**
- After training, save:
  ```
  checkpoints/run_{run_id}/
    config.yaml               # Full Hydra resolved config
    model.pt                  # Checkpoint
    metrics.json              # {"train_loss": [...], "train_time": 42.5}
    param_count.txt           # "encoder: 23.5M, head: 156K, total: 23.7M"
    git_info.txt              # Commit hash + branch
  ```
- Scripts must compute `run_id` from timestamp or UUID, not user input.
- **Why:** When reproducing a result months later, you need to know the exact config and code version that produced it.

### **R9: No Silent Fallbacks. Raise Descriptive Errors. No Bare `except: pass`**
- Every ValueError, RuntimeError, or custom exception must include context:
  ```python
  if not torch.isfinite(loss).all():
      raise RuntimeError(
          f"NaN in loss. temperature={temperature}, "
          f"max_p_fn={p_fn.max():.4f}, "
          f"expected finite. Check pair scorer bounds."
      )
  ```
- Do NOT catch and ignore errors. Let them propagate; tests and CI will catch them.
- **Why:** Silent failures corrupt results silently. Descriptive errors speed debugging by 10×.

### **R10: Modules ≤200 Lines. Split If Larger**
- Each file in `src/{models,losses,metrics,data}/` should be ≤200 lines of non-test code.
- If a file grows beyond 200 lines, split it into submodules (`model_a.py`, `model_b.py`) and import in `__init__.py`.
- Comments and docstrings count toward the line limit.
- **Why:** Smaller modules are easier to unit-test, audit, and reason about. The loss-auditor subagent can only check files ≤200 lines carefully.

---

## **Subagents & Their Activation Rules**

Use the following subagents to audit code after each stage. Invoke with `/run-subagent <name>`.

| **Subagent** | **Checks** | **Invoke After** | **Trigger Rule** |
|---|---|---|---|
| **loss-auditor** | R1, R2, R3, R7, R9. Tests all masks, verifies dict keys, checks no hardcoded temps. | Stages 2, 6, 7, 7.5 | Any file in `src/losses/` or `src/models/pair_scorer.py` changed. |
| **dataset-leakage-checker** | R4, R5. Verifies patient-level splits, disjointness, no patient overlap across splits. | Stages 1, 9 | Any file in `src/data/` changed. |
| **experiment-auditor** | R6, R8. Checks Hydra configs, verifies artifact saving, git hash recording. | Before marking a stage complete | Before committing any run. |
| **pytorch-debugger** | Tensor shape mismatches, NaN propagation, gradient flow on all params. | Stages 2, 6, 7, 7.5 | After loss implementation, before first probe run. |
| **code-reviewer** | R10 (line counts), imports, dead code, PEP 8. | General review | Before any PR or stage gate. |

---

## **Skills: Activate Before Implementation**

| **Skill** | **When to Activate** |
|---|---|
| **contrastive-loss-engineer** | Stages 2, 6, 7, 7.5. Guides masked loss design, FN weighting, edge-aware mechanisms. |
| **cxr-dataset-pipeline** | Stages 1, 9. Ensures patient splits, uncertainty handling, augmentation correctness. |
| **experiment-config-hydra** | Stage 0 (scaffolding). Guides YAML structure, nested configs, multirun syntax. |

---

## **Commands: Short-Circuit Audits**

- **/review-stage `<N>`** — runs experiment-auditor on Stage N, reports 0 FAILs or lists issues.
- **/check-loss** — runs loss-auditor on current `src/losses/` changes.
- **/check-data** — runs dataset-leakage-checker on current `src/data/` changes.

---

## **Numerical Stability Pitfalls (Critical)**

### **1. log(1 − p_FN) for p_FN Close to 1**
When p_FN → 1, log(1 − p_FN) → −∞. Use:
```python
weight = (1 - p_fn).clamp_min(1e-10)
loss_component = weight.log()
```
Both branches of `torch.where()` are **always evaluated**, so even the false branch will overflow if not clamped.

### **2. p_FN Outside [0, 1]**
Scorer outputs must be strictly bounded. Validate:
```python
assert (p_fn >= 0).all() and (p_fn <= 1).all(), \
    f"p_fn has values outside [0,1]: min={p_fn.min()}, max={p_fn.max()}"
```

### **3. L2-Normalization Stability**
When z has norm near zero, `F.normalize()` can produce NaN. Check:
```python
norm = z.norm(dim=-1, keepdim=True).clamp_min(1e-12)
z_norm = z / norm
```

### **4. Edge Fingerprint Compression**
For high-dimensional edge tensors (e.g., d_in × d_out × num_centers), compressing to 256 dimensions is mandatory to avoid memory explosion.

---

## **Hypothesis Gates (When Each Claim is Testable)**

| **Gate** | **Passes When** | **Blocks** |
|---|---|---|
| **H1 (KAN > MLP geometry)** | Geometry metrics green (alignment, uniformity, effective_rank). KAN ≥ MLP on ≥2/3 metrics. | Stage 5 merge |
| **H2 (FN loss improves recall)** | FN-weighted AUROC > InfoNCE AUROC across 3 seeds. Rare-disease AUROC improvement ≥2% absolute. | Stage 7 merge |
| **H3 (KAN scorer > MLP scorer)** | KAN scorer AUROC > MLP scorer AUROC within H2 framework. Param parity verified. | Stage 7.5 setup |
| **H4 (Edge signals improve FN detection)** | Edge-aware loss + edge-align (λ>0) gives ≥1% AUROC improvement over z-only scorer. | Final ablation |

---

## **Checklist Before Commit**

- [ ] All tests in `tests/` pass: `pytest tests/ -v` green.
- [ ] Relevant subagent invoked and 0 FAILs.
- [ ] Config saved to `configs/...` and committed.
- [ ] Artifacts (checkpoints, metrics, git hash) saved.
- [ ] Rule violations listed in comment if intentional (with justification).
- [ ] New TODO items added to TODO.md for next stage.

---

## **When to Escalate to Opus 4.7**

Use Opus (high-effort Claude Code) when:
- Implementing or modifying any loss function (R3, R7 compliance is subtle).
- Implementing dataset split logic (R4, R5 compliance requires careful patient ID tracking).
- Debugging NaN or non-convergence (pytorch-debugger needed).

Use Sonnet 4.6 (medium) when:
- Implementing evaluation modules (linear probe, kNN, AUROC) — well-specified contract, less room for subtle bugs.
- Adding configs or utility functions.

Use Haiku 4.5 (low) when:
- File/variable renaming, TODO updates, simple type annotations.

---

## **Git Commit Messages**

Format: `[Stage<N>] <what> — <why one sentence>`

```
[Stage2] InfoNCE loss + diagonal mask — R3: unit tests on pos/neg/FN cases
[Stage6] FN-weighted loss + MLP scorer — R1: param parity ±15% verified
[Stage7.5] Edge-aware FN loss — H4: edge fingerprint signals for FN detection
```

---

## **Metadata for Papers**

Every thesis-relevant claim must cite:
- Stage number (0–10)
- Commitment hash (git log --oneline | head -1)
- Seeds used (default [42, 1337, 2024], full [42, 1337, 2024, 7, 9001])
- Mean ± std over seeds from `probe_results.csv`
