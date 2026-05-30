# Baselines: Debiased CL and Multi-label SupCon Oracle (ChestMNIST)

Source: 2026-05-30 user task brief. Companion to `BUG_*.md`,
`reports/pfn_saturation_fix.md`, and `CLAUDE.md`.

## Why these baselines

The thesis claims that **false-negative-aware** contrastive learning (FN-weighted
InfoNCE, edge-aware KAN-FPCL) improves chest-X-ray representations because
multi-label CXR datasets routinely have hidden positives among InfoNCE's
nominal negatives. Before that claim is publishable we need two stronger
reference points than plain InfoNCE:

1. **Debiased Contrastive Learning (Chuang et al. 2020) — label-free.**
   Corrects the InfoNCE bias *without* using labels by subtracting an
   estimate of the positive-class mass from the negative pool. If our
   FN-aware methods don't beat Debiased CL we are not, in fact, adding
   useful supervision over a *purely statistical* bias correction.

2. **Multi-label SupCon Oracle — label-aware ceiling.**
   Generalises SupCon (Khosla et al. 2020) to multi-hot labels. Uses
   ChestMNIST training labels during contrastive pre-training as ground
   truth for positivity. This is **not** a self-supervised method — it
   is an *upper bound* on what FN-aware methods could in principle
   recover from labels. Read the gap between SupCon Oracle and
   KAN-FPCL as "how much label-aware signal we still missed".

Both are now in the headline comparison table next to InfoNCE,
FN-weighted (MLP), FN-weighted (KAN), and edge-aware KAN-FPCL.

## A. Debiased Contrastive Learning

* **File:** `src/losses/debiased_contrastive.py`
* **Class:** `DebiasedContrastiveLoss(temperature=0.5, tau_plus=0.1,
  normalize_embeddings=True, estimator_clip_min=1e-8)`
* **Forward:** `forward(z) -> dict[str, Tensor]` — **label-free** by contract;
  ValueError on non-finite / odd-batch `z`.
* **Estimator:** standard Chuang-et-al. correction

  ```
  pos      = exp(sim(anchor, positive) / temperature)
  neg_sum  = sum_j exp(sim(anchor, j) / temperature)   over j != self, != pos
  debiased = (neg_sum - tau_plus * N * pos) / (1 - tau_plus)
  debiased = clamp(debiased, min=estimator_clip_min)   # never NaN
  loss_i   = -log(pos / (pos + debiased))
  ```

  Averaged over all `2B` anchors.
* **Dict keys:** `loss, pos_sim_mean, neg_sim_mean, pos_exp_mean,
  neg_exp_mean, debiased_neg_mean, tau_plus, temperature`.
* **Tag:** `loss_name="debiased"`, `baseline_family="false_negative_aware_baseline"`,
  `oracle_labels_used=false`.

## B. Multi-label SupCon Oracle

* **File:** `src/losses/supcon_oracle.py`
* **Class:** `MultilabelSupConOracleLoss(temperature=0.1,
  normalize_embeddings=True, positive_mode="any_overlap", min_jaccard=0.0,
  include_self_view_positive=True, no_positive_policy="self_view_only")`
* **Forward:** `forward(z, labels) -> dict[str, Tensor]` — `labels` are the
  un-duplicated `[B, C]` multi-hot ChestMNIST labels; the loss expands them
  internally so they line up with `z = [v1; v2]`.
* **Positive modes:**
  - `"any_overlap"`: pairs sharing ≥1 label.
  - `"jaccard"`: pairs with `Jaccard(L_i, L_j) > min_jaccard`. Jaccard
    is the detached positive weight; self-view positives carry weight `1.0`.
* **No-positive policy:**
  - `"self_view_only"`: fall back to the augmented-view positive (R3 — never
    silently NaN).
  - `"drop_anchor"`: exclude the anchor; `dropped_anchor_fraction` logged.
* **Dict keys:** `loss, num_oracle_positives_{mean,min,max},
  positive_pair_fraction, dropped_anchor_fraction, pos_sim_mean,
  neg_sim_mean, temperature, positive_mode, min_jaccard`.
* **Tag:** `loss_name="supcon_oracle"`,
  `baseline_family="oracle_label_aware_baseline"`, `oracle_labels_used=true`.

The labels are **only** consumed by this loss. They are not routed through
the FN scorer, the KAN scorer, or the edge-aware KAN-FPCL path.

## C. Shared mask utilities

* **File:** `src/losses/multilabel_masks.py`
* **Functions:** `expand_labels_for_two_views`, `multilabel_overlap_matrix`,
  `multilabel_jaccard_matrix`, `build_self_view_positive_mask`,
  `build_supcon_oracle_positive_mask`.
* Diagonal is always False; all-zero label rows do not produce NaN
  (Jaccard 0/0 → 0); unit tests in `tests/test_multilabel_masks.py`.

## D. Training integration

* `scripts/train/train.py` now routes on `cfg.loss.name`:
  - `infonce` / `debiased` (or no `name`): `loss_fn(z)`.
  - `supcon_oracle`: `loss_fn(z, labels.to(device))`.
* `name` is popped from the loss config before `get_class(target)(**kwargs)`
  so it is never passed as a constructor kwarg.
* `train_fn.py` / `train_edge.py` are unchanged — FN-weighted and
  edge-aware paths continue to work exactly as before.
* Validation is inlined in `train.py` (rather than via
  `train_common.run_validation`) because the SupCon Oracle val pass needs
  per-batch labels.

## E. Configs

| File | Notes |
|------|-------|
| `configs/loss/debiased.yaml` | `name: debiased`, τ=0.5, τ⁺=0.1. |
| `configs/loss/supcon_oracle_any_overlap.yaml` | `name: supcon_oracle`, any-overlap. |
| `configs/loss/supcon_oracle_jaccard.yaml` | `name: supcon_oracle`, jaccard mode. |
| `configs/experiment/chestmnist_debiased.yaml` | Same encoder/projector/budget as `full_mlp_infonce`. |
| `configs/experiment/chestmnist_supcon_oracle_any_overlap.yaml` | Oracle, any-overlap. |
| `configs/experiment/chestmnist_supcon_oracle_jaccard.yaml` | Oracle, Jaccard. |
| `configs/experiment/smoke_{debiased,supcon_oracle_any_overlap,supcon_oracle_jaccard}.yaml` | 2-step CPU smokes. |

## F. Probe-results CSV

`scripts/analysis/probe.py` now emits, in addition to the existing columns:

* `loss_name` — `infonce` / `debiased` / `supcon_oracle` / `fn_weighted` / `edge_aware_fn`.
* `baseline_family` — `self_supervised_baseline` /
  `false_negative_aware_baseline` / `oracle_label_aware_baseline` /
  `false_negative_aware_method`.
* `oracle_labels_used` — string `"true"` only for `supcon_oracle`.
* `positive_mode` — `"any_overlap"` / `"jaccard"` for SupCon Oracle, else empty.
* `tau_plus` — populated for Debiased CL.
* `temperature` — recorded for every loss for cross-row comparison.

These fields are derived from the checkpoint's `config.yaml/loss` block,
so re-probing a saved run will populate them retroactively.

## G. Tests

* `tests/test_multilabel_masks.py` (16 tests) — shape, diagonal, overlap,
  Jaccard, self-view, all-zero-labels-no-NaN.
* `tests/test_debiased_contrastive.py` (13 tests) — label-free contract,
  estimator clamp under adversarial τ⁺, perfect-positive loss, gradient flow.
* `tests/test_supcon_oracle.py` (17 tests) — label-required, any-overlap,
  jaccard threshold, drop-anchor policy, dict keys.
* Existing tests untouched: `test_infonce.py`, `test_fn_weighted_loss.py`,
  `test_edge_aware_fn_loss.py` all still pass.

## H. Comparison protocol

Any KAN-FPCL claim should report at minimum:

1. **InfoNCE** — self-supervised reference.
2. **Debiased CL** — self-supervised + statistical FN correction.
3. **SupCon Oracle (any-overlap)** — label-aware ceiling, lax positives.
4. **SupCon Oracle (jaccard)** — label-aware ceiling, weighted positives.
5. **FN-weighted MLP scorer** — H2 evidence row.
6. **FN-weighted KAN scorer** — H3 evidence row.
7. **Edge-aware KAN-FPCL** — H4 evidence row.

If our method (5–7) does not beat **Debiased CL** (2), we have not
demonstrated that the FN scoring adds anything over a label-free
statistical correction. If it does not approach **SupCon Oracle** (3–4),
that gap quantifies the headroom still on the table.

## I. Acceptance

* `pytest tests/test_debiased_contrastive.py tests/test_supcon_oracle.py
  tests/test_multilabel_masks.py -v` — 46 pass.
* `pytest tests/test_infonce.py tests/test_fn_weighted_loss.py
  tests/test_edge_aware_fn_loss.py -v` — green.
* `pytest tests/ -q` — **289 passed**.
* CPU smoke runs (`smoke_debiased`, `smoke_supcon_oracle_any_overlap`,
  `smoke_supcon_oracle_jaccard`, `smoke_mlp run.device=cpu`) all complete
  with the R8 artifact bundle (config.yaml, metrics.json, model_best.pt,
  step_metrics.csv/json, git_info.txt, param_count.txt) under
  `runs/checkpoints/`.
* No NaNs, no silent fallback to InfoNCE.
