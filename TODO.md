# TODO.md — KAN-FPCL Implementation Checklist

This file tracks progress through 11 stages (0–10) of the KAN-FPCL thesis implementation. Mark items complete only when the specified gate condition is met.

---

## **Stage 0: Repository Bootstrap**

**Gate:** `pytest tests/test_smoke.py -v` passes. All files exist.

- [ ] Create directory structure: `src/`, `configs/`, `tests/`, `scripts/`, `reports/`, `.claude/`
- [ ] Create CLAUDE.md with 10 scientific rules (R1–R10)
- [ ] Create AGENTS.md with 5 subagent specifications
- [ ] Create README.md with thesis overview and hypotheses
- [ ] Create TODO.md (this file)
- [ ] Create pyproject.toml with dependencies:
  - [ ] torch, torchvision
  - [ ] hydra-core, omegaconf
  - [ ] medmnist, einops
  - [ ] scikit-learn, pandas, numpy, tqdm
  - [ ] pytest, pytest-cov
  - [ ] (optional) pytorch-lightning, umap-learn
- [ ] Create .gitignore (venv/, checkpoints/, __pycache__, *.pyc, .DS_Store)
- [ ] Create subagent files in `.claude/agents/`:
  - [ ] loss-auditor.md
  - [ ] dataset-leakage-checker.md
  - [ ] experiment-auditor.md
  - [ ] pytorch-debugger.md
  - [ ] code-reviewer.md
- [ ] Create skill files in `.claude/skills/`:
  - [ ] contrastive-loss-engineer.md
  - [ ] cxr-dataset-pipeline.md
  - [ ] experiment-config-hydra.md
- [ ] Create command file: `.claude/commands/review-stage.py`
- [ ] Create stub __init__.py files in all src/ subdirectories
- [ ] Create tests/test_smoke.py (imports, directory checks)
- [ ] git init && git add -A && git commit -m "[Stage0] repo bootstrap"

**Subagent:** code-reviewer (final check)

**Notes:**
- [ ] Verify CLAUDE.md is committed (not ignored)
- [ ] Verify pyproject.toml has all required packages
- [ ] Verify all subagent and skill files are readable YAML/Markdown

**Status:** ☐ Not Started | ☐ In Progress | ☑ **Complete** | ☐ Blocked

---

## **Stage 1: ChestMNIST Smoke Data Pipeline**

**Gate:** `pytest tests/test_data_*.py -v` passes. No leakage. Patient disjointness verified.

- [ ] Implement `src/data/chestmnist.py`:
  - [ ] ChestMNISTDataset class wrapping medmnist.ChestMNIST
  - [ ] __getitem__ returns (image, label_vector, patient_id)
  - [ ] Verify label dtype and shape (14-dim multi-label)
- [ ] Implement `src/data/splits.py`:
  - [ ] patient_level_split(ids, ratios=[0.7, 0.15, 0.15], seed=42) → train_ids, val_ids, test_ids
  - [ ] Raise ValueError if any patient in >1 split
  - [ ] Unit test on synthetic + real data
- [ ] Implement `src/data/augmentations.py`:
  - [ ] TwoViewTransform (SimCLR-style, no vertical flip for CXR)
  - [ ] Inherits from torchvision.transforms.Compose
- [ ] Implement `src/data/__init__.py`:
  - [ ] get_dataloader(cfg) → dict[str, DataLoader]
  - [ ] Returns {train, val, test} dataloaders
- [ ] Create `configs/data/chestmnist.yaml`:
  - [ ] batch_size, num_workers, pin_memory
  - [ ] augmentation params
- [ ] Implement tests/test_data_splits.py:
  - [ ] test_disjointness() — verify no patient overlap
  - [ ] test_disjointness_real_sample() — on ChestMNIST sample
- [ ] Implement tests/test_data_pipeline.py:
  - [ ] test_batch_shapes() — image [B,1,28,28], label [B,14]
  - [ ] test_two_views() — TwoViewTransform produces 2 augmented views
  - [ ] test_label_dtype() — uint8 or float32 as expected
- [ ] Activate: `/run-subagent dataset-leakage-checker` (must PASS)
- [ ] Commit: git commit -m "[Stage1] ChestMNIST pipeline + disjoint splits"

**Subagents:**
- [ ] dataset-leakage-checker (R4, R5)
- [ ] code-reviewer (R10)

**Notes:**
- [ ] Patient ID must be unique per image (ChestMNIST uses image index)
- [ ] No row-level train/test split (forbidden)
- [ ] Report breakdown: "Train: X images from Y patients", etc.

**Status:** ☐ Not Started | ☐ In Progress | ☐ Complete | ☐ Blocked

---

## **Stage 2: MLP Encoder + InfoNCE Baseline (CRITICAL GATE)**

**Gate:** Tests pass. Loss auditor passes. Probe produces AUROC in probe_results.csv.

- [ ] Implement `src/models/encoder.py`:
  - [ ] ResNet-18 backbone (pretrained: bool config)
  - [ ] forward(x) → [B, 512]
- [ ] Implement `src/models/mlp_head.py`:
  - [ ] MLPProjector(input_dim=512, hidden_dim, output_dim=128, hidden_activation='relu')
  - [ ] L2 normalization at output (optional config)
  - [ ] Expose parameter_count() → int
- [ ] Implement `src/losses/masks.py`:
  - [ ] build_positive_mask(B) → [2B, 2B] bool mask (diagonal excluded)
  - [ ] Unit test: positive-pair mask is correct
- [ ] Implement `src/losses/infonce.py`:
  - [ ] InfoNCELoss(temperature=0.1, normalize_embeddings=True)
  - [ ] forward(z[2B, D]) → dict{loss, pos_sim_mean, neg_sim_mean, temperature}
  - [ ] Verify diagonal excluded (R3)
  - [ ] Clamp temperature > 0 (R9)
- [ ] Implement `src/utils/param_count.py`:
  - [ ] count_parameters(model) → dict with layer-wise counts
- [ ] Implement `src/utils/checkpoint.py`:
  - [ ] save_checkpoint(model, config, metrics, path)
  - [ ] Saves model.pt, config.yaml, metrics.json, param_count.txt, git_info.txt (R8)
- [ ] Implement `src/utils/reproducibility.py`:
  - [ ] set_seed(seed) — sets torch, numpy, random seeds
- [ ] Create configs/model/encoder.yaml, configs/model/mlp_head.yaml, configs/loss/infonce.yaml
- [ ] Implement scripts/train.py:
  - [ ] @hydra.main decorator
  - [ ] Hydra-driven hyperparams (no hardcoding)
  - [ ] Saves artifacts (R8)
  - [ ] Logs every N epochs
- [ ] Implement tests/test_infonce.py:
  - [ ] test_diagonal_excluded() — verify mask excludes self-pairs
  - [ ] test_positive_only_batch() — loss → 0
  - [ ] test_negative_only_batch() — loss finite and > 0
  - [ ] test_fn_false_negatives() — marked FN → loss increases
  - [ ] test_dict_keys() — verify all keys present
  - [ ] test_gradient_flow() — all params have gradients
  - [ ] test_temperature_validation() — ValueError on T ≤ 0
- [ ] Implement tests/test_mlp_head.py:
  - [ ] test_output_shape() — [B, output_dim]
  - [ ] test_l2_norm() — output L2-norm ≈ 1
  - [ ] test_param_count() — matches manual count
- [ ] Activate: `/run-subagent loss-auditor` (must PASS)
- [ ] Activate: `/run-subagent pytorch-debugger` (must PASS)
- [ ] Run train: `python scripts/train.py configs/experiment/smoke_mlp.yaml seed=42 num_epochs=5`
  - [ ] Verify loss decreases
  - [ ] Verify artifacts saved
- [ ] Commit: git commit -m "[Stage2] InfoNCE baseline + MLP head — R3, R7, R9 compliance"

**Subagents:**
- [ ] loss-auditor (R1, R3, R7, R9)
- [ ] pytorch-debugger (NaN, gradient flow)
- [ ] code-reviewer (R10)

**Paper Gate:**
- [ ] Baseline AUROC measured but NOT YET used for claims (wait for Stage 3 probe)

**Notes:**
- [ ] This is the CRITICAL BASELINE. Every later result compares to this.
- [ ] InfoNCE loss has a subtle bug if diagonal mask is wrong — unit tests catch it.
- [ ] Do NOT optimize temperature during training yet (fixed for now).

**Status:** ☐ Not Started | ☐ In Progress | ☐ Complete | ☐ Blocked

---

## **Stage 3: Linear Probe + kNN Evaluation (HYPOTHESIS GATE)**

**Gate:** probe_results.csv has Stage 2 baseline row(s). AUROC numbers for all later claims.

- [ ] Implement `src/metrics/linear_probe.py`:
  - [ ] linear_probe(train_emb[N_tr, D], train_labels[N_tr, K], val_emb, val_labels, max_iter, C)
  - [ ] sklearn LogisticRegression, one-vs-rest (multilabel)
  - [ ] Returns dict{macro_auroc, micro_auroc, per_class_auroc, mAP}
- [ ] Implement `src/metrics/knn.py`:
  - [ ] knn_eval(train_emb, train_labels, val_emb, val_labels, k=20)
  - [ ] Cosine distance, weighted vote
  - [ ] Returns dict{macro_auroc, micro_auroc, per_class_auroc, mAP}
- [ ] Implement `src/metrics/auc.py`:
  - [ ] multilabel_auc(scores[N, K], labels[N, K]) → dict with per-class and macro AUROC
  - [ ] Raise ValueError if any class has 0 positives (R9)
  - [ ] No silent NaN
- [ ] Implement scripts/probe.py:
  - [ ] @hydra.main decorator
  - [ ] Loads checkpoint
  - [ ] Extracts frozen embeddings (no gradient)
  - [ ] Runs linear_probe() + knn_eval()
  - [ ] Computes geometry metrics (Stage 8 ready)
  - [ ] Appends row to probe_results.csv
  - [ ] CSV columns: run_id, encoder, head, loss, scorer, dataset, seed, params_total, macro_auroc_linear, macro_auroc_knn, mAP, runtime_sec
- [ ] Create configs/experiment/probe.yaml
- [ ] Implement tests/test_metrics.py:
  - [ ] test_random_embeddings() — AUROC ≈ 0.5
  - [ ] test_perfect_clustering() — AUROC ≈ 1.0
  - [ ] test_zero_positives() — ValueError
  - [ ] test_dict_keys() — all required keys present
  - [ ] test_knn_identical_vectors() — k=1 → AUROC = 1.0
- [ ] Run probe on Stage 2 checkpoint:
  - [ ] `python scripts/probe.py checkpoint=<stage2-checkpoint> seed=42`
  - [ ] Verify row appended to probe_results.csv
  - [ ] Inspect AUROC (rough estimate for sanity check)
- [ ] Commit: git commit -m "[Stage3] Linear probe + kNN — H1/H2/H3/H4 measurement harness"

**Subagents:**
- [ ] experiment-auditor (R6, R8)
- [ ] code-reviewer (R10)

**Paper Gate:**
- [ ] Stage 3 gate passed = Stage 2 baseline AUROC is now official
- [ ] All later claims must compare against this baseline
- [ ] No stage > 3 may proceed until this row is in probe_results.csv

**Notes:**
- [ ] This is the HYPOTHESIS GATE. Without AUROC numbers, no claim is verifiable.
- [ ] Linear probe and kNN together cover both "simple" and "complex" downstream tasks.
- [ ] Report mean ± std over 3+ seeds for all claims.

**Status:** ☐ Not Started | ☐ In Progress | ☐ Complete | ☐ Blocked

---

## **Stage 4: FastKAN Projector**

**Gate:** Tests pass. Geometry metrics computed. Probe row committed.

- [ ] Implement `src/models/kan/fastkan.py`:
  - [ ] FastKANLayer(input_dim, output_dim, num_centers=8, grid_min=-2.0, grid_max=2.0)
  - [ ] forward(x[B, in]) → [B, out]
  - [ ] **Critical:** forward(x, return_edges=False) → Tensor | Tuple[Tensor, Tensor]
  - [ ] When return_edges=True: return (output, phi[B, out, in])
  - [ ] phi[b, o, i] = per-edge activation before summing
  - [ ] L2 normalization in FastKANProjector
  - [ ] Expose num_edges property
- [ ] Implement `src/models/kan/__init__.py` (exports)
- [ ] Create configs/model/kan_head.yaml:
  - [ ] Parameter count within ±15% of MLP baseline (Stage 2)
  - [ ] Document: "MLP baseline: 1089 params, KAN variant: 1193 params (+9.5%)"
- [ ] Implement tests/test_fastkan.py:
  - [ ] test_output_shape() — [B, output_dim]
  - [ ] test_rbf_centers_fixed() — centres are buffers (non-trainable)
  - [ ] test_l2_norm() — output norm ≈ 1
  - [ ] test_gradient_flow() — all trainable params have gradients
  - [ ] test_return_edges_tuple() — return_edges=True → (output, phi)
  - [ ] test_phi_shape() — phi is [B, out, in]
  - [ ] test_parameter_count_parity() — within ±15% of MLP
  - [ ] test_rbf_bandwidth_clamp() — bandwidth >= 1e-6 (stability)
- [ ] Activate: `/run-subagent loss-auditor` on pair_scorer.py (if used in loss)
- [ ] Run smoke train on KAN: `python scripts/train.py configs/experiment/smoke_kan.yaml seed=42`
- [ ] Run probe: append KAN baseline row to probe_results.csv
- [ ] Test H1 (geometry metrics): run analyze_geometry.py on Stage 4 checkpoint
- [ ] Commit: git commit -m "[Stage4] FastKAN projector + return_edges support — H1 geometry test"

**Subagents:**
- [ ] loss-auditor (if pair scorer uses KAN)
- [ ] pytorch-debugger (NaN in phi, gradient flow)
- [ ] code-reviewer (R10, line limits)

**H1 Gate:**
- [ ] Geometry metrics computed for both MLP (Stage 2) and KAN (Stage 4)
- [ ] Alignment, uniformity, effective rank compared
- [ ] If KAN ≥ MLP on ≥2/3 metrics, H1 shows promise

**Notes:**
- [ ] FastKAN RBF basis: centers non-trainable, bandwidth learnable
- [ ] return_edges is critical for Stage 7.5 (edge-aware loss)
- [ ] Parameter parity is mandatory (R1)

**Status:** ☐ Not Started | ☐ In Progress | ☐ Complete | ☐ Blocked

---

## **Stage 5: Residual FastKAN Warp**

**Gate:** Tests pass (4 invariants). Probe row committed.

- [ ] Implement `src/models/kan/residual_warp.py`:
  - [ ] ResidualFastKANWarp(input_dim, hidden_dim, alpha_init=0.0, learnable_alpha=True, clamp_alpha=True, clamp_max=0.2)
  - [ ] forward(z[B, d]) → [B, d] L2-normalized
  - [ ] z̃ = normalise(z + α · KAN(LayerNorm(z)))
  - [ ] Invariant 1: alpha_init=0, learnable=False ⇒ warp(z) == F.normalize(z) EXACTLY
  - [ ] Invariant 2: gradients flow through alpha and KAN params
  - [ ] Invariant 3: alpha clamped to [0, clamp_max] when clamp_alpha=True
  - [ ] Invariant 4: output always L2-normalised
  - [ ] @property alpha (clamped value for logging)
- [ ] Create configs/model/residual_fastkan_warp.yaml
- [ ] Implement tests/test_residual_kan.py:
  - [ ] test_identity_at_init() — alpha=0, learnable=False ⇒ warp identical to F.normalize
  - [ ] test_gradient_flow() — gradients to alpha, KAN W
  - [ ] test_alpha_clamp() — alpha clamped to [0, 0.2]
  - [ ] test_output_norm() — output always L2-normalized
- [ ] Run smoke train: `python scripts/train.py configs/experiment/smoke_residual_kan.yaml seed=42`
- [ ] Monitor alpha value during training (should grow from 0 if beneficial)
- [ ] Run probe: append row for residual KAN to probe_results.csv
- [ ] Compare H1 metrics: Res-KAN vs KAN vs MLP geometry
- [ ] Commit: git commit -m "[Stage5] Residual FastKAN warp — identity-at-init invariants verified"

**Subagents:**
- [ ] pytorch-debugger (alpha clamping, norm computation)
- [ ] code-reviewer (R10)

**Notes:**
- [ ] Identity-at-init is crucial: allows interpretation of warp as learned correction
- [ ] If alpha stays near zero, warp is unnecessary (clean negative result)
- [ ] LayerNorm stabilizes KAN input

**Status:** ☐ Not Started | ☐ In Progress | ☐ Complete | ☐ Blocked

---

## **Stage 6: FN-Weighted InfoNCE + MLP Pair Scorer (H2 GATE)**

**Gate:** Tests pass. Loss auditor passes. Probe row committed. H2 gate tested.

- [ ] Implement `src/losses/fn_weighted_infonce.py`:
  - [ ] FNWeightedInfoNCELoss(temperature=0.1, normalize_embeddings=True, max_fn_weight=0.95)
  - [ ] forward(z[2B, D], p_fn[B, B]) → dict{loss, pos_sim_mean, neg_sim_mean, p_fn_mean, p_fn_max, downweighted_fraction, temperature}
  - [ ] loss = InfoNCE(z) with downweighting: weight = (1 − p_fn).clamp_min(1e-10)
  - [ ] Numerical stability: clamp before .log() (R9)
  - [ ] Validate: temperature > 0, p_fn in [0, 1], no NaN
- [ ] Implement `src/models/pair_scorer.py`:
  - [ ] MLPPairScorer(input_dim=512, hidden_dim=32, num_layers=2, activation='relu')
  - [ ] forward(z_view1[B, D]) → [B, B] in [0, 1]
  - [ ] Pair features: [cos_sim, l2_dist, element_wise_prod, ...]
  - [ ] Expose parameter_count() → int
- [ ] Create configs/loss/fn_weighted_mlp.yaml
- [ ] Implement tests/test_fn_weighted_loss.py (R3):
  - [ ] test_p_fn_zero_equals_infonce() — p_fn=0 everywhere ⇒ loss ≈ InfoNCE (allclose 1e-5)
  - [ ] test_p_fn_one_near_zero() — p_fn=1 (true negatives) ⇒ loss → 0
  - [ ] test_dict_keys() — all required keys present
  - [ ] test_monotonicity() — higher p_fn ⇒ lower loss
  - [ ] test_numerical_stability() — no NaN on edge cases
  - [ ] test_gradient_flow() — all params have gradients
  - [ ] test_temperature_validation() — ValueError on T ≤ 0
  - [ ] test_p_fn_bounds() — ValueError if p_fn < 0 or > 1
- [ ] Implement tests/test_pair_scorer.py:
  - [ ] test_output_shape() — [B, B]
  - [ ] test_output_bounds() — all values in [0, 1]
  - [ ] test_gradient_flow() — all params have gradients
  - [ ] test_parameter_count() — parity with MLP head
- [ ] Implement scripts/train.py updates (Stage 2 already done, but wire pair scorer):
  - [ ] Loss forward receives z and p_fn = pair_scorer(z)
  - [ ] No labels used in scorer (R3 implicit)
- [ ] Activate: `/run-subagent loss-auditor` (must PASS, especially FN unit tests)
- [ ] Activate: `/run-subagent pytorch-debugger` (must PASS)
- [ ] Run train: `python scripts/train.py configs/experiment/fn_weighted_mlp.yaml seed=42 num_epochs=10`
- [ ] Run probe on 3 seeds: [42, 1337, 2024]
- [ ] **H2 Gate Check:**
  - [ ] Compute macro-AUROC mean ± std (FN-weighted MLP vs InfoNCE MLP)
  - [ ] If FN-weighted AUROC > InfoNCE AUROC by ≥1% absolute → H2 shows promise
  - [ ] If not, investigate: is p_FN collapsing? Is pair scorer broken?
- [ ] Commit: git commit -m "[Stage6] FN-weighted loss + MLP scorer — H2 gate test"

**Subagents:**
- [ ] loss-auditor (R1, R3, R7, R9 — especially FN unit tests)
- [ ] pytorch-debugger (log(0) NaN, p_fn bounds)
- [ ] code-reviewer (R10)

**H2 Gate:**
- [ ] FN-weighted AUROC > InfoNCE AUROC (ChestMNIST + CheXpert, mean over 3 seeds)
- [ ] Rare-disease AUROC improvement ≥2% absolute
- [ ] If gate fails, investigate pair scorer (is it learning? Is p_FN reasonable?)

**Notes:**
- [ ] Pair scorer must be label-free (no labels passed during training)
- [ ] p_FN should be ∈ [0, 1]; pair scorer should have sigmoid output
- [ ] If p_FN collapses to all 0s or all 1s, the loss becomes InfoNCE (bad) or ~0 (bad)

**Status:** ☐ Not Started | ☐ In Progress | ☐ Complete | ☐ Blocked

---

## **Stage 7: KAN Pair Scorer (H3 GATE)**

**Gate:** Tests pass. Loss auditor passes. Probe row committed. H3 gate tested.

- [ ] Extend `src/models/pair_scorer.py` with KANPairScorer:
  - [ ] KANPairScorer(input_dim=512, hidden_dim=4, num_layers=2, num_centers=8)
  - [ ] forward(z_view1[B, D]) → [B, B] in [0, 1]
  - [ ] Uses FastKANLayer internally
  - [ ] Parameter count within ±15% of MLPPairScorer(hidden_dim=32) (R1)
  - [ ] Output: sigmoid (or tanh clamped to [0, 1])
- [ ] Create configs/loss/fn_weighted_kan.yaml (same loss, different scorer config)
- [ ] Implement tests/test_kan_pair_scorer.py:
  - [ ] test_output_shape() — [B, B]
  - [ ] test_output_bounds() — all values in [0, 1]
  - [ ] test_gradient_flow() — all params have gradients
  - [ ] test_parameter_parity() — params within ±15% of MLPPairScorer
  - [ ] test_interchangeability() — can swap KANPairScorer into FNWeightedInfoNCELoss without error
- [ ] Activate: `/run-subagent loss-auditor` (must PASS)
- [ ] Run train: `python scripts/train.py configs/experiment/fn_weighted_kan.yaml seed=42`
- [ ] Run probe on 3 seeds: [42, 1337, 2024]
- [ ] **H3 Gate Check:**
  - [ ] Compute macro-AUROC mean ± std (KAN scorer vs MLP scorer, both with FN-weighted loss)
  - [ ] If KAN scorer AUROC ≥ MLP scorer AUROC (or within margin of error) → H3 holds
  - [ ] If KAN << MLP, investigate: parameter mismatch? Scorer architecture?
- [ ] Commit: git commit -m "[Stage7] KAN pair scorer — H3 gate test (parity with MLP)"

**Subagents:**
- [ ] loss-auditor (R1 parity check)
- [ ] pytorch-debugger
- [ ] code-reviewer (R10)

**H3 Gate:**
- [ ] KAN scorer AUROC ≥ MLP scorer AUROC (mean over 3 seeds, same loss)
- [ ] Parameter parity verified (±15%)
- [ ] If gate passes, KAN is not worse at the scorer task

**Notes:**
- [ ] KAN scorer is not claimed to be **better** yet — H3 just checks parity
- [ ] Edge signals come in Stage 7.5 (H4), which unlocks the advantage

**Status:** ☐ Not Started | ☐ In Progress | ☐ Complete | ☐ Blocked

---

## **Stage 7.5: Edge-Aware FN-Weighted InfoNCE (KAN-FPCL, H4 GATE)**

**Gate:** Tests pass (backward-compat verified). Loss auditor passes. Probe rows committed. H4 gate tested.

- [ ] Implement `src/losses/edge_features.py`:
  - [ ] edge_fingerprint(phi[B, O, I]) → F[B, 256] L2-normalized
  - [ ] Flatten phi[B, O, I] → [B, O*I], project to 256-d via learned/fixed linear layer
  - [ ] L2-normalize result
  - [ ] edge_pair_similarity(F[B, 256]) → [B, B] cosine similarity matrix
- [ ] Extend `src/models/pair_scorer.py` with EdgeAwarePairScorer:
  - [ ] EdgeAwarePairScorer(input_dim, edge_dim, hidden_dim, num_layers, use_edge_features: bool, scorer_type: 'mlp'|'kan')
  - [ ] When use_edge_features=True:
    - [ ] Pair features now include: [cos_sim_z, ..., e_ik, delta_ik]
    - [ ] e_ik = edge_pair_similarity[i, k]
    - [ ] delta_ik = e_ik − cos_sim_z[i, k] (disagreement signal)
  - [ ] When use_edge_features=False: identical to Stage 6/7 scorer (backward compat)
  - [ ] forward(z_view1, edge_features: Tensor|None) → [B, B]
  - [ ] **CRITICAL CONTRACT:** No labels flow into the scorer (label-free)
- [ ] Implement `src/losses/edge_aware_fn_loss.py`:
  - [ ] EdgeAwareFNWeightedInfoNCELoss(temperature, lambda_edge=0.0, tau_edge=0.1, lambda_edge_align=0.0)
  - [ ] forward(z[2B, D], p_fn[B, B], edge_features[B, 256]|None) → dict
  - [ ] loss = fn_loss + lambda_edge * edge_contrastive_loss + lambda_edge_align * edge_align_loss
  - [ ] fn_loss: FNWeightedInfoNCELoss(z, p_fn) [compose, don't duplicate]
  - [ ] edge_contrastive_loss: InfoNCE(edge_features, tau_edge)
  - [ ] edge_align_loss: mean_i (1 − cos(F_i^view1, F_i^view2))
  - [ ] When lambda_edge=0 and lambda_edge_align=0: must equal Stage 7 loss exactly (allclose 1e-5)
  - [ ] Dict keys: loss, fn_loss, edge_contrastive_loss, edge_align_loss, lambda_edge, lambda_edge_align, temperature, tau_edge, ...
- [ ] Create configs/loss/edge_aware_fn_mlp.yaml, edge_aware_fn_kan.yaml, smoke_edge_aware.yaml
- [ ] Update scripts/train.py:
  - [ ] When loss config has edge_aware: true, call projector(h, return_edges=True) → (z, phi)
  - [ ] Wire phi → edge_fingerprint → edge_features
  - [ ] Pass edge_features to loss and scorer
  - [ ] When edge_aware: false, behavior identical to Stage 6/7
- [ ] Implement tests/test_edge_aware_fn_loss.py (R3):
  - [ ] **BACKWARD COMPAT (critical):**
    - [ ] lambda_edge=0, lambda_edge_align=0, use_edge_features=False ⇒ loss ≈ Stage 7 loss (allclose 1e-5)
    - [ ] lambda_edge=0, lambda_edge_align=0, edge_features provided ⇒ loss still ≈ Stage 7
  - [ ] test_edge_features_none_with_nonzero_lambda() → ValueError
  - [ ] test_use_edge_features_true_with_none_features() → ValueError
  - [ ] test_edge_fingerprint_zero_vector() → no NaN
  - [ ] test_monotonicity_edge_contrastive() — higher edge_pos_sim ⇒ lower edge_contrastive_loss
  - [ ] test_gradient_flow() — gradients to z, scorer params, KAN W
  - [ ] test_dict_keys() — all 13+ keys present
  - [ ] test_param_parity() — EdgeAware(kan) ±15% of EdgeAware(mlp)
- [ ] Implement tests/test_edge_features.py:
  - [ ] test_fingerprint_norm() — F is L2-normalized
  - [ ] test_similarity_symmetric() — similarity[i,j] ≈ similarity[j,i]
  - [ ] test_identical_features() — edge_pair_similarity(F, F) → ~1.0 diagonal
- [ ] Activate: `/run-subagent loss-auditor` (must PASS, backward-compat critical)
- [ ] Activate: `/run-subagent pytorch-debugger` (must PASS)
- [ ] Run train with lambda_edge=0.0 (should match Stage 7):
  - [ ] `python scripts/train.py configs/experiment/edge_aware_fn_kan.yaml lambda_edge=0.0 seed=42`
  - [ ] Verify loss curves match Stage 7
- [ ] Run train with lambda_edge=0.05:
  - [ ] `python scripts/train.py configs/experiment/edge_aware_fn_kan.yaml lambda_edge=0.05 seed=42`
  - [ ] Monitor edge_contrastive_loss and edge_align_loss
- [ ] Run probe on 3 seeds for BOTH lambda_edge=0.0 and lambda_edge=0.05
- [ ] **H4 Gate Check:**
  - [ ] Compute macro-AUROC mean ± std (lambda=0.05 vs lambda=0.0)
  - [ ] If lambda=0.05 AUROC ≥ lambda=0.0 AUROC by ≥1% absolute → H4 shows promise
  - [ ] Compare to Stage 7 (z-only): edge signals + edge-align should beat z-only
  - [ ] If gate fails, investigate: is edge_contrastive_loss helping? Is edge_align_loss a bottleneck?
- [ ] Commit: git commit -m "[Stage7.5] Edge-aware FN loss (KAN-FPCL) — H4 gate test (novel)"

**Subagents:**
- [ ] loss-auditor (R1, R3, R7, R9 — backward-compat test is critical)
- [ ] pytorch-debugger (edge fingerprint computation, gradient flow)
- [ ] code-reviewer (R10)

**H4 Gate:**
- [ ] Edge-aware loss (lambda=0.05) AUROC ≥ z-only loss (lambda=0.0) AUROC
- [ ] Gain ≥1% absolute (ChestMNIST + CheXpert, mean over 3 seeds)
- [ ] Backward-compat test passes: lambda=0 → Stage 7 loss exactly
- [ ] If gate fails, edge signals may be correlated with z by construction (requires deeper investigation)

**Notes:**
- [ ] **This is the most novel stage.** Edge-pathway signals are unique to KAN.
- [ ] Edge features must be label-free (no labels in scorer or loss during training).
- [ ] Label-aware pair scorer is reserved for oracle ablation in Stage 10 only.
- [ ] If H4 fails but H1–H3 succeed, thesis still has strong contributions.

**Status:** ☐ Not Started | ☐ In Progress | ☐ Complete | ☐ Blocked

---

## **Stage 8: Geometry Metrics**

**Gate:** Tests pass. Metrics computed for all variants (MLP, KAN, Res-KAN). Tables generated.

- [ ] Implement `src/metrics/geometry.py`:
  - [ ] alignment(z_a, z_b) → scalar (MSE between positive-pair embeddings)
  - [ ] uniformity(z, t=2.0) → scalar (expected distance on unit sphere)
  - [ ] effective_rank(z) → scalar (SVD entropy → exp)
  - [ ] per_dim_std(z) → [D] (per-dimension std)
  - [ ] off_diagonal_covariance_norm(z) → scalar
  - [ ] Use torch.pdist for efficiency (avoid [N, N] matrices)
- [ ] Implement `src/metrics/embedding_viz.py`:
  - [ ] save_umap(embeddings, labels, output_path, use_pca_fallback=True)
- [ ] Create tests/test_geometry.py:
  - [ ] test_alignment_identical() — identical vectors → 0
  - [ ] test_uniform_vs_clustered() — uniform > clustered
  - [ ] test_effective_rank_ones() — rank-1 → ~1
  - [ ] test_effective_rank_uniform() — uniform → ~D
  - [ ] test_per_dim_std_range() — reasonable values
- [ ] Implement scripts/analyze_geometry.py:
  - [ ] Loads checkpoint
  - [ ] Extracts frozen embeddings (train, val, test)
  - [ ] Computes all 5 geometry metrics
  - [ ] Appends row to reports/tables/geometry.csv
  - [ ] CSV columns: run_id, encoder, head, loss, seed, alignment, uniformity, effective_rank, per_dim_std_mean, offdiag_covariance, dataset
- [ ] Analyze geometry for all Stage 2–7.5 variants (3 seeds each)
- [ ] **H1 Hypothesis Check:**
  - [ ] Compare MLP (Stage 2) vs KAN (Stage 4) vs Res-KAN (Stage 5)
  - [ ] If KAN ≥ MLP on ≥2/3 metrics → H1 holds
  - [ ] Report: "Alignment: MLP=X, KAN=Y (Z% improvement), Uniformity: ..."
- [ ] Commit: git commit -m "[Stage8] Geometry metrics (alignment, uniformity, rank) — H1 support"

**Subagents:**
- [ ] code-reviewer (R10)

**Notes:**
- [ ] Geometry metrics are **explanatory** (why the representation is better), not **causal** claims.
- [ ] Report mean ± std over seeds.

**Status:** ☐ Not Started | ☐ In Progress | ☐ Complete | ☐ Blocked

---

## **Stage 9: CheXpert Full Pipeline (PRODUCTION DATA)**

**Gate:** Tests pass. Leakage checker passes. CheXpert baseline row in probe_results.csv.

- [ ] Download CheXpert dataset (if not already done)
- [ ] Implement `src/data/chexpert.py`:
  - [ ] CheXpertDataset(root, split, view, uncertainty_policy, image_size)
  - [ ] Parses CSV, extracts patient ID from Path prefix
  - [ ] uncertainty_policy: 'ignore' (drop), 'positive' (1.0), 'negative' (0.0), 'lsr' (0.5)
  - [ ] **Config-driven (R6):** ValueError if policy not specified or invalid (R9)
  - [ ] forward returns (image, label_vector, patient_id)
- [ ] Extend `src/data/splits.py` for CheXpert patient-ID splitting
  - [ ] Patient counts: ~65K patients, 224K images
  - [ ] Suggested split: train 45K/11.2K patients, val 10K/2.5K, test 12K/3K
- [ ] Create configs/data/chexpert.yaml:
  - [ ] uncertainty_policy, image_size, batch_size, num_workers
- [ ] Implement scripts/preprocess_chexpert.py:
  - [ ] Resize images, re-encode as JPEG
  - [ ] Idempotent (don't re-encode if already done)
- [ ] Implement tests/test_chexpert.py:
  - [ ] Synthetic CSV fixture with uncertainty labels
  - [ ] test_uncertainty_policy_ignore() — uncertain rows dropped
  - [ ] test_uncertainty_policy_positive() — uncertain → 1.0
  - [ ] test_disjoint_splits() — patient-level disjointness (R5)
- [ ] Activate: `/run-subagent dataset-leakage-checker` (must PASS)
- [ ] Train MLP baseline on CheXpert:
  - [ ] `python scripts/train.py configs/experiment/baseline_mlp.yaml dataset=chexpert seed=42 num_epochs=50`
- [ ] Run probe on 3 seeds
- [ ] **CheXpert Baseline Gate:**
  - [ ] AUROC measured (baseline for all CheXpert claims)
  - [ ] All later CheXpert results reported as mean ± std over seeds
- [ ] Commit: git commit -m "[Stage9] CheXpert pipeline (patient-level splits, uncertainty handling)"

**Subagents:**
- [ ] dataset-leakage-checker (R4, R5)
- [ ] code-reviewer (R10)

**Notes:**
- [ ] CheXpert uncertainty policy is critical: wrong policy silently corrupts training
- [ ] All headline numbers come from CheXpert (ChestMNIST for sanity checks only)

**Status:** ☐ Not Started | ☐ In Progress | ☐ Complete | ☐ Blocked

---

## **Stage 10: Full Ablation Runner + Paper Tables (FINAL)**

**Gate:** ablation_master.csv has all cells × 3 seeds. All 4 tables generated. No FAILs.

- [ ] Implement scripts/ablate.py:
  - [ ] Hydra --multirun over ablation matrix (9 cells × 3+ seeds)
  - [ ] For each cell: train → probe → analyze_geometry
  - [ ] On error: log to CSV and continue (don't halt)
  - [ ] Output: ablation_master.csv with 17 columns (cell_id, head, loss, scorer, lambda_edge, lambda_edge_align, dataset, seed, params_total, macro_auroc_linear, macro_auroc_knn, mAP, alignment, uniformity, effective_rank, runtime_sec)
- [ ] Implement scripts/make_paper_tables.py:
  - [ ] Read ablation_master.csv
  - [ ] Generate 4 tables:
    - [ ] **table_h1.md:** Geometry metrics (MLP vs KAN vs Res-KAN)
    - [ ] **table_h2.md:** AUROC improvements (InfoNCE vs FN-weighted)
    - [ ] **table_h3.md:** Scorer comparison (MLP vs KAN at matched params)
    - [ ] **table_h4.md:** Edge-aware ablation (λ=0.0 vs 0.05, edge-align on/off)
  - [ ] Each table: mean ± std, param counts, seed counts
  - [ ] Use pandas.to_markdown()
- [ ] Run full ablation:
  - [ ] Default: `python scripts/ablate.py --multirun seed=42,1337,2024`
  - [ ] Full: `python scripts/ablate.py --multirun seed=42,1337,2024,7,9001`
- [ ] Generate tables:
  - [ ] `python scripts/make_paper_tables.py`
- [ ] Verify ablation_master.csv:
  - [ ] 9 cells × 3 seeds = 27 rows (or 45 rows if full)
  - [ ] 0 FAILED entries (or investigate and fix)
  - [ ] AUROC ranges are reasonable (not all NaN, not all 0 or 1)
- [ ] Review generated tables:
  - [ ] table_h1.md: KAN geometry ≥ MLP on ≥2/3 metrics?
  - [ ] table_h2.md: FN-weighted AUROC > InfoNCE AUROC by ≥1%?
  - [ ] table_h3.md: KAN scorer AUROC ≥ MLP scorer AUROC?
  - [ ] table_h4.md: Edge-aware (λ=0.05) AUROC > z-only (λ=0.0) AUROC by ≥1%?
- [ ] Activate: `/run-subagent experiment-auditor` on scripts/ablate.py (must PASS)
- [ ] Activate: `/run-subagent code-reviewer` on all final scripts (must PASS)
- [ ] Generate negative-result discussion (if any hypotheses fail):
  - [ ] Analyze which stages contributed to failure
  - [ ] Propose follow-up experiments
  - [ ] Frame failure as scientific insight (when components don't help, what can we learn?)
- [ ] Commit: git commit -m "[Stage10] Full ablation (9 cells × 3+ seeds) — H1/H2/H3/H4 final results"

**Subagents:**
- [ ] experiment-auditor (Hydra multirun, error handling)
- [ ] code-reviewer (R10)

**Final Checklist Before Submission:**
- [ ] All 4 hypothesis tables generated and reviewed
- [ ] ablation_master.csv committed (with all rows)
- [ ] probe_results.csv committed (all probes from all stages)
- [ ] geometry.csv committed (all geometry metrics)
- [ ] No stage > 3 has unverified AUROC claims (all compared to baseline)
- [ ] All scientific rules (R1–R10) followed
- [ ] All subagent audits passed
- [ ] README.md, CLAUDE.md, AGENTS.md, TODO.md up-to-date

**Status:** ☐ Not Started | ☐ In Progress | ☐ Complete | ☐ Blocked

---

## **Cross-Cutting Tasks**

- [ ] Version Control:
  - [ ] `git init` (Stage 0)
  - [ ] Commit after each stage gate passes
  - [ ] Commit format: `[Stage<N>] <what> — <why>`
  - [ ] Final commit before submission: `[Final] All stages complete — submission ready`

- [ ] Logging & Reporting:
  - [ ] Every subagent run logged to reports/subagent_audits.log
  - [ ] AUROC results logged to probe_results.csv (cumulative)
  - [ ] Geometry metrics logged to geometry.csv (cumulative)
  - [ ] Ablation results logged to ablation_master.csv (final)

- [ ] Documentation:
  - [ ] CLAUDE.md: 10 rules, subagents, skills, commands
  - [ ] AGENTS.md: 5 subagent specs in detail
  - [ ] README.md: thesis overview, hypotheses, setup, results
  - [ ] TODO.md: this file (stage checklist)
  - [ ] All updated as stages complete

- [ ] Paper Writing (Outside Scope of This TODO, But Relevant):
  - [ ] H1 analysis: use geometry metrics + AUROC from tables
  - [ ] H2 analysis: use FN-weighted AUROC gains from table_h2.md
  - [ ] H3 analysis: use KAN vs MLP scorer AUROC from table_h3.md
  - [ ] H4 analysis: use edge-aware AUROC gains from table_h4.md
  - [ ] Negative results: reframe as scientific contributions (Section: "When Edge Signals Don't Help")

---

## **Final Status**

**Overall Progress:**
```
Completed Stages: [ 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 7.5 | 8 | 9 | 10 ]
Date Completed: _______________

H1 Result: ☐ Strong Positive | ☐ Moderate | ☐ Negative
H2 Result: ☐ Strong Positive | ☐ Moderate | ☐ Negative
H3 Result: ☐ Strong Positive | ☐ Moderate | ☐ Negative
H4 Result: ☐ Strong Positive | ☐ Moderate | ☐ Negative

Thesis Status: ☐ In Progress | ☐ Ready for Writing | ☐ Submitted
```

**Key Milestones:**
- [ ] Stage 3 gate (probe harness) ✓
- [ ] Stage 6/7 gate (H2/H3 tested) ✓
- [ ] Stage 7.5 gate (H4 tested) ✓
- [ ] Stage 10 gate (full ablation) ✓
- [ ] All tables generated ✓
- [ ] All rules (R1–R10) verified ✓
- [ ] All subagents passed ✓

---

**Last Updated:** May 2026  
**Maintainer:** [Author Name]  
**Contact:** [Email]

