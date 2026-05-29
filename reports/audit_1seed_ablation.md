# Experiment-Auditor Report — 1-Seed Ablation (seed 42)

Source: `runs/results/ablation_1seed_master.csv` (11 rows, all `status=OK`)
Tables: `runs/tables/table_h{1,2,3,4}.md`
Date: 2026-05-28 · Scope: ChestMNIST only (CLAUDE.md §Dataset Scope)

> NOTE ON PATHS: the prompt referenced `reports/tables/` but the tables and the
> CSV actually live under `runs/tables/` and `runs/results/`. `reports/tables/`
> is empty. No table is missing — they are just in `runs/`.

---

## SECTION 1 — INVENTORY

### 1a. Full CSV (all 11 rows)

| cell_id | head | loss | scorer | λ_edge | λ_align | seed | params_total | params_scorer | auroc_lin | auroc_knn | mAP | alignment | uniformity | eff_rank | runtime_s | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp_infonce | mlp | infonce | none | 0.0 | 0.0 | 42 | 11,499,584 | 0 | 0.674855 | 0.627504 | 0.111656 | 0.224545 | -3.885061 | 116.92 | 1033.01 | OK |
| kan_infonce | fastkan | infonce | none | 0.0 | 0.0 | 42 | 11,498,747 | 0 | 0.671829 | 0.619449 | 0.102529 | 0.207447 | -3.768189 | 45.53 | 1085.65 | OK |
| reskan_infonce | residual_fastkan | infonce | none | 0.0 | 0.0 | 42 | 11,536,851 | 0 | 0.678254 | 0.628854 | 0.109956 | 0.226575 | -3.882958 | 116.30 | 989.91 | OK |
| mlp_fn_mlp | mlp | fn_weighted | mlp | 0.0 | 0.0 | 42 | 11,508,897 | 9,313 | 0.679846 | 0.633022 | 0.111383 | 0.223100 | -3.874785 | 113.03 | 989.47 | OK |
| mlp_fn_kan | mlp | fn_weighted | kan | 0.0 | 0.0 | 42 | 11,508,556 | 8,972 | 0.685827 | 0.633127 | 0.112795 | 0.221619 | -3.874969 | 113.31 | 1265.89 | OK |
| kan_fn_kan | fastkan | fn_weighted | kan | 0.0 | 0.0 | 42 | 11,548,046 | 8,972 | 0.677941 | 0.618835 | 0.102604 | 0.209468 | -3.775478 | 49.88 | 1322.12 | OK |
| edge_scorer_no_aux | fastkan | edge_aware_fn | edge_mlp | 0.0 | 0.0 | 42 | 11,543,539 | 4,465 | 0.684442 | 0.620976 | 0.107622 | 0.208730 | -3.777546 | 47.90 | 1282.19 | OK |
| edge_contrastive | fastkan | edge_aware_fn | edge_mlp | 0.05 | 0.0 | 42 | 11,543,539 | 4,465 | 0.680625 | 0.615904 | 0.106135 | 0.236816 | -3.846600 | 103.42 | 1279.95 | OK |
| edge_align | fastkan | edge_aware_fn | edge_mlp | 0.0 | 0.05 | 42 | 11,543,539 | 4,465 | 0.686699 | 0.625049 | 0.105411 | 0.212591 | -3.791282 | 67.79 | 1288.41 | OK |
| zonly_fn | fastkan | edge_aware_fn | mlp_zonly | 0.0 | 0.0 | 42 | 11,543,507 | 4,433 | 0.680823 | 0.618413 | 0.103165 | 0.206572 | -3.776012 | 48.41 | 1287.10 | OK |
| edge_contrastive_kan | fastkan | edge_aware_fn | edge_kan | 0.05 | 0.0 | 42 | 11,543,244 | 4,170 | 0.678132 | 0.619917 | 0.103571 | 0.240321 | -3.845403 | 103.25 | 1311.55 | OK |

### 1b. Cell coverage (11 expected from `configs/experiment/ablation.yaml`)

All 11 cells **PRESENT** with `status=OK`. None FAILED, none MISSING.

1. mlp_infonce — PRESENT · 2. kan_infonce — PRESENT · 3. reskan_infonce — PRESENT
4. mlp_fn_mlp — PRESENT · 5. mlp_fn_kan — PRESENT · 6. kan_fn_kan — PRESENT
7. edge_scorer_no_aux — PRESENT · 8. edge_contrastive — PRESENT · 9. edge_align — PRESENT
10. zonly_fn — PRESENT · 11. edge_contrastive_kan — PRESENT

### 1c. NULL/NA columns

- **No column present in the CSV is entirely NULL.** Every column above is populated for all 11 rows. `params_scorer=0` only for the three no-scorer H1 cells (expected, not null).
- **`per_class_auroc_linear_json`: PRESENT and populated for all 11 rows** ✓ (includes `hernia`, `emphysema`, `fibrosis`).
- **`rare_disease_auroc`: there is NO such column in the master CSV.** It is not produced by `ablate.py`; `make_paper_tables.py` synthesizes it. For H2 it is recomputed from the JSON (real values); for H4 it is left as NaN → shows `NA` (see 2f / Bug #2).
- **Test-split columns absent.** `probe.py` emits `macro_auroc_linear_test`, `mAP_test`, `per_class_auroc_linear_test_json`, etc., but the master CSV keeps only the **val** metrics (`macro_auroc_linear` = val). The reported headline AUROC is therefore validation AUROC (see Bug #3).

### 1d. table_h2.md / table_h3.md existence

Both **EXIST** at `runs/tables/table_h2.md` and `runs/tables/table_h3.md` (all four H-tables present). The H2 builder fires on cells `mlp_infonce`+`mlp_fn_mlp` (both present); the H3 builder fires on `mlp_fn_mlp`,`mlp_fn_kan`,`kan_fn_kan`,`edge_contrastive`,`edge_contrastive_kan` (all present). No cell absence blocked any table.

---

## SECTION 2 — CODE INVESTIGATION

Anomaly under test: `kan_infonce` effective_rank **45.53** vs `mlp_infonce` **116.92** /
`reskan_infonce` **116.30**, at near-identical param counts.

### 2a. `src/models/kan/fastkan.py` — norm / init / gradient flow
**EXPECTED BEHAVIOR (no code defect).** There is **no BatchNorm or LayerNorm** anywhere in `FastKANLayer`/`FastKANProjector`; `rbf_weight ~ N(0, 0.1)`, `bandwidth = grid_width/(centers-1) ≈ 0.571`, final L2-norm with `clamp_min(1e-12)`. The forward path has **no `.detach()` and no non-differentiable op** (`exp`, `einsum`, `sum`, `+ base`), so every layer receives gradient. The low rank is **not** caused by normalization, init, or dead gradients — it is caused by the **hidden bottleneck** (see 2b/2d).

### 2b. `src/models/mlp_head.py` vs KAN — does the difference explain the gap?
**EXPECTED BEHAVIOR — yes, the hidden width alone explains it.** `MLPHead` is `512 → 512 → 128` (BN, ReLU): the hidden width (512) is **above** the 128 output, so the 128-dim output is not bottlenecked → effective_rank ≈ 117. The param-matched `FastKANProjector` is `512 → 57 → 128` (`kan_head.yaml hidden_dim: 57`): the hidden width **57 is below the 128 output**, so the output factors through a 57-dim space and effective_rank is capped near 57. Observed **45.53 ≤ 57** is exactly consistent. BN-vs-no-BN is a second-order effect; the rank gap is the **hidden-bottleneck confound**.

### 2c. `scripts/analyze_geometry.py` — is `z` taken after the full head, in eval()?
**EXPECTED BEHAVIOR (extraction is correct).** `_load_model` calls `encoder.eval()` and `head.eval()` (lines 68–69); `_embed = F.normalize(head(encoder(images)))` (line 102) takes `z` **after the complete head, including the final KAN layer**. The extra `F.normalize` is idempotent with the projector's internal L2-norm (harmless). Crucially, **the same extraction yielded rank 116.92 for MLP**, so an N-cap / sampling artifact is ruled out — the 45.53 is a genuine property of the FastKAN embedding, not an extraction bug.

### 2d. `full_kan_infonce.yaml` vs `full_mlp_infonce.yaml` — differing output dim?
**No output-dim difference; the confound is hidden dim.** Both heads emit `output_dim: 128`. The only relevant difference is `hidden_dim`: **57 (FastKAN) vs 512 (MLP)**, forced by R1 param-matching (the RBF weight tensor `O×I×C` is param-expensive, so matching ~329K head params required shrinking the KAN hidden width below the output width). `reskan` confirms the mechanism: its KAN runs as a **residual** (`z + α·KAN`, `α_init=0`, clamped ≤0.2) around a full `512→512→128` MLP main path, so the 16-dim KAN bottleneck never constrains the main embedding → rank 116.30.

### 2e. `fn_weighted_infonce.py` / `edge_aware_fn_loss.py` — can p_FN collapse to ~1?
**POSSIBLE BUG by construction, but mitigated in config (not triggered here).** The FN-weighted loss is monotonically decreasing in `p_fn` (CLAUDE.md Pitfall #5), so a scorer is rewarded for driving `p_fn → max_fn_weight`. The loss correctly **clamps `(1-p_fn).clamp_min(1e-10)` before `.log()`**, validates `p_fn ∈ [0,1]`, and raises on NaN/non-finite (R9) — sigmoid direction is correct. Collapse is prevented by **config**, not by the loss: `fn_weighted_mlp.yaml`/`edge_aware.yaml` set `max_fn_weight: 0.5` (floor of 0.5 on negative weights) and the FN experiment configs add `lambda_pfn_reg: 0.01`, plus the checkpoint selector now scores on `val_total = loss + λ·p_fn.mean()`. The H2 row (auroc 0.6798, not random) confirms no catastrophic collapse occurred at seed 42. Residual risk remains if a future config sets `max_fn_weight=1.0` with `lambda_pfn_reg=0`.

### 2f. `scripts/probe.py` — per-class AUROC + rare-class names
**Probe is correct; the NA is a downstream table bug.** `probe.py` computes per-class AUROC and serializes `per_class_auroc_linear_json` (lines 259–262), and `CHESTMNIST_CLASS_NAMES` (lines 37–42) includes `emphysema`, `fibrosis`, `hernia` — **exactly matching** `RARE_CLASSES = ("hernia","emphysema","fibrosis")` in `make_paper_tables.py` (line 23). The rare-class data is present for every cell. The H4 `NA` comes from `make_paper_tables.py`: the H4 builder aggregates `rare_disease_auroc=("rare_disease_auroc", _auroc)` (line 421) against the **all-NaN synthesized column**, instead of computing it from the JSON the way H2 does via `_extract_rare_auroc` (lines 369–371). One-line fix: add `h4["rare_auroc"] = h4["per_class_auroc_linear_json"].apply(_extract_rare_auroc)` and aggregate that, mirroring H2.

---

## SECTION 3 — PRELIMINARY HYPOTHESIS SIGNALS (n=1, no error bars)

**H1 | PRELIMINARY SIGNAL: Negative (for KAN), and confounded.**
Key numbers — macro_auroc: mlp 0.6749 / kan 0.6718 / reskan 0.6783; alignment (lower=better): mlp 0.2245 / kan 0.2074 / reskan 0.2266; uniformity (more-neg=better): mlp −3.885 / kan −3.768 / reskan −3.883; effective_rank: mlp 116.92 / kan 45.53 / reskan 116.30. KAN beats MLP on only **1/3** geometry metrics (alignment), losing uniformity and effective_rank, and is below MLP on AUROC → fails the H1 gate (≥2/3).
R1 check: params_total mlp 11,499,584 / kan 11,498,747 (−0.007%) / reskan 11,536,851 (+0.32%); head-level 329,344 / 328,507 (−0.25%) / 366,611 (+11.3%) — **all within ±15% ✓**.
Primary concern: the effective_rank gap is a **structural confound** (FastKAN `hidden_dim=57` bottleneck below the 128 output), not a clean architecture effect — "KAN has lower rank" is currently inseparable from "KAN's hidden layer is narrower."
Thesis risk: **High.**

**H2 | PRELIMINARY SIGNAL: Neutral (negative on the headline rare-class metric).**
Key numbers — mlp_infonce vs mlp_fn_mlp: macro 0.6749 → 0.6798 (+0.0049); rare_disease 0.6604 → 0.6608 (**+0.0004**); mAP 0.1117 → 0.1114. The H2 gate needs **≥2% absolute rare-disease improvement** across 3 seeds; observed is ~0.04%.
Primary concern: the core H2 claim (FN-weighting lifts rare/recall) is essentially unsupported at 1 seed — rare-disease delta ≈ 0.
Thesis risk: **Medium-High.**

**H3 | PRELIMINARY SIGNAL: Weakly Positive but inconsistent.**
Key numbers — pure-FN context: mlp_fn_mlp 0.6798 vs mlp_fn_kan 0.6858 (**+0.0060**, KAN scorer wins), mAP 0.1114 → 0.1128; edge context cross-check: edge_contrastive (edge_mlp) 0.6806 vs edge_contrastive_kan (edge_kan) 0.6781 (**−0.0025**, KAN scorer loses). Param parity: MLP scorer 9,313 vs KAN scorer 8,972 (−3.7%) ✓; edge_mlp 4,465 vs edge_kan 4,170 (−6.6%) ✓.
Primary concern: the KAN-scorer advantage **flips sign** between the FN and edge settings → not robust at 1 seed.
Thesis risk: **Medium.**

**H4 | PRELIMINARY SIGNAL: Neutral / weakly-positive, below gate.**
Key numbers (table_h4) — zonly_fn 0.6808, edge_scorer_no_aux 0.6844, edge_contrastive 0.6806, edge_align 0.6867. Edge-feature effect (no_aux − zonly) = +0.0036; edge_align − zonly = **+0.0059 (<1% gate)**; edge_contrastive − zonly = **−0.0002**, and edge_contrastive − no_aux = −0.0038. rare_disease_auroc = **NA** (reporting bug, Bug #2).
Primary concern: as flagged, the λ_edge=0.05 contrastive term does **not** help (it sits below both z-only and the no-aux control) — at 1 seed this is indistinguishable from noise, but it trends the wrong way; only edge-**align** moves the right direction and even that misses the 1% gate.
Thesis risk: **Medium-High.**

---

## SECTION 4 — RANKED BUG LIST

1. **[HIGH / BLOCKING for H1]** — FastKAN `hidden_dim=57` is a bottleneck *below* the 128-dim output, capping effective_rank (~45.5) and confounding the H1 geometry comparison with a narrow-hidden artifact.
   → Evidence: `configs/model/kan_head.yaml:14` (`hidden_dim: 57`); CSV `effective_rank` 45.53 (kan) vs 116.92 (mlp); `src/models/kan/fastkan.py` (no rank-reducing norm — pure bottleneck effect).
   → Fix required before 3-seed run: **Yes** (3 seeds only add error bars to a confounded comparison).

2. **[MEDIUM]** — H4 table reports `rare_disease_auroc = NA` because `make_paper_tables.py` aggregates the all-NaN synthesized column instead of computing from `per_class_auroc_linear_json` (as H2 does).
   → Evidence: `scripts/make_paper_tables.py:421` (H4 agg) vs `:369-371` + `_extract_rare_auroc` (`:309-318`); CSV column `per_class_auroc_linear_json` is fully populated.
   → Fix required before 3-seed run: **No** (table-gen only; data is in the CSV and can be regenerated post-hoc — but fix before *reading* H4).

3. **[MEDIUM]** — Ablation master keeps only **val** metrics (`macro_auroc_linear` = val); the test-split columns `probe.py` emits are dropped. `model_best.pt` is selected on val, so val-evaluated headline numbers are optimistically biased.
   → Evidence: master CSV header lacks `macro_auroc_linear_test`; `scripts/probe.py:48-53,283-287` emit test columns.
   → Fix required before 3-seed run: **Yes, if test is the intended headline** (cheap: carry test columns through `ablate.py`).

4. **[LOW]** — `edge_contrastive` (λ_edge=0.05) underperforms both `zonly_fn` and `edge_scorer_no_aux`; the contrastive aux may be structurally unhelpful, but 1 seed cannot separate this from noise.
   → Evidence: `runs/tables/table_h4.md` (0.6806 vs 0.6808 vs 0.6844).
   → Fix required before 3-seed run: **No** (investigate once 3 seeds give error bars).

5. **[LOW]** — All tables print `± 0.0000` (n=1) which can be misread as zero variance / significance.
   → Evidence: every `runs/tables/table_h*.md` cell; `_fmt_metric` sets std=0 when `len==1` (`make_paper_tables.py:170`).
   → Fix required before 3-seed run: **No** (self-resolves at n=3; consider printing `n=1` caveat).

No BLOCKING-for-all-results bug found: the pipeline executed cleanly and produced plausible, non-random numbers. The HIGH item blocks the **H1 claim** specifically.

---

## SECTION 5 — BEFORE YOU RUN THE 3-SEED ABLATION

### Must-fix (else 3-seed results are uninterpretable or systematically wrong)

- **FIX 1 — Resolve the H1 hidden-bottleneck confound.** Decide and implement one of: (a) match the FastKAN head to MLP on a geometry-relevant axis (equal `hidden_dim`, or `hidden_dim ≥ output_dim`) and reframe any param difference as an explicit "parameter ablation"; or (b) keep `hidden_dim=57` but reframe H1 to state plainly that the effective_rank result is a capacity/bottleneck effect, not a pure KAN-vs-MLP effect. Without this, the 3-seed H1 geometry comparison stays confounded.
- **FIX 2 — Fix the headline split.** Pick val *or* test as the headline metric and ensure the ablation master carries the chosen split's AUROC columns. If headline = test (recommended, since `model_best` is val-selected), thread `macro_auroc_linear_test`/`mAP_test`/test JSON from `probe.py` through `ablate.py` so 3-seed numbers aren't val-selection-biased.

### Nice-to-have (not blocking the run)

- Fix `make_paper_tables.py` H4 `rare_disease_auroc` to read from `per_class_auroc_linear_json` (Bug #2) — regenerable any time.
- Suppress or annotate `± 0.0000` for n=1 outputs (Bug #5).
- After 3 seeds, re-examine whether `edge_contrastive` (λ_edge=0.05) is genuinely unhelpful vs noise (Bug #4); if real, consider tuning/curriculum on λ_edge.
