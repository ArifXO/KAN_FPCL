# BugFix.md — 1-Seed Ablation Audit Remediation

Source audit: `reports/audit_1seed_ablation.md` (experiment-auditor, seed 42, 2026-05-28).
Fixes applied: 2026-05-29. Scope: ChestMNIST only (CLAUDE.md §Dataset Scope).

This file records, per bug, **what was broken** and **what was fixed**. A fresh
review pass after the fixes is appended in the last section.

---

## Bug #1 — [HIGH / was blocking H1] FastKAN hidden-bottleneck confound

**What was broken.** `configs/model/kan_head.yaml` sets `hidden_dim: 57`, which is
*below* the 128-D output. The embedding is forced through a 57-D space, so
`effective_rank` is capped near 57 (observed 45.5 vs MLP's 116.9). "KAN has lower
rank" was therefore inseparable from "KAN's hidden layer is narrower than its
output" — a structural confound, not a clean architecture effect. The audit's
suggested fix (`hidden ≥ output`) collides head-on with R1 (±15% param matching):
the RBF weight tensor is `O×I×centers`, so a wide KAN head balloons to ~737K
params (+124% vs MLP's 329K). You cannot have both `hidden ≥ output` and R1 parity
for a pure KAN head.

**What was fixed.** Per the agreed "keep matched + add labeled wide variant"
decision, the R1-matched head is left untouched (it remains the parity control),
and a **labeled parameter-ablation** cell was added to disentangle the two effects:

- New `configs/model/kan_wide_head.yaml` — `hidden_dim: 128` (= output). Head =
  **737,538** params (verified by instantiation). The file header documents the
  per-layer param breakdown and the explicit R1 deviation (CLAUDE.md Rule 5): this
  cell is a *capacity* reference, never a head-to-head KAN-vs-MLP claim.
- New `configs/experiment/full_kan_wide_infonce.yaml` — InfoNCE, wide head.
- New `kan_wide_infonce` cell in `configs/experiment/ablation.yaml` (hypothesis h1,
  `head: fastkan_wide`), plus an updated label-verification header.
- `scripts/make_paper_tables.py` H1 slice + coverage gate now include
  `kan_wide_infonce` as an extra context row, annotated as a parameter ablation.

**How to read it after the run.** Compare `kan_wide_infonce` vs `kan_infonce`
(same architecture, only hidden width differs). If wide-KAN's effective_rank jumps
toward ~117, the original rank gap was the bottleneck, not KAN. The R1-matched
`kan_infonce` stays the only cell used for the parity H1 verdict.

---

## Bug #2 — [MEDIUM] H4 table reported `rare_disease_auroc = NA`

**What was broken.** `scripts/make_paper_tables.py` H4 builder aggregated the
synthesized `rare_disease_auroc` column, which is all-NaN in the master CSV, so the
table printed `NA` even though the rare-class data was fully present in
`per_class_auroc_linear_json`. H2 already did this correctly via `_extract_rare_auroc`.

**What was fixed.** The H4 builder now computes `rare_auroc` from
`per_class_auroc_linear_json` (mean over hernia/emphysema/fibrosis), mirroring H2.
Regenerated `runs/tables/table_h4.md` now shows real values (e.g. zonly_fn 0.6868,
edge_scorer_no_aux 0.6538, edge_align 0.6771).

---

## Bug #3 — [MEDIUM] Master CSV kept only val metrics (val-selection bias)

**What was broken.** `scripts/ablate.py` `_COLUMNS` carried only
`macro_auroc_linear` (val). `model_best.pt` is selected on val, so val-evaluated
headline numbers are optimistically biased. `probe.py` already emits the test-split
columns, but `ablate.py` dropped them on the way into the master CSV.

**What was fixed.** `ablate.py` now carries `macro_auroc_linear_test`,
`macro_auroc_knn_test`, `mAP_test`, and `per_class_auroc_linear_test_json` through
`_COLUMNS` and the `_run_cell` `row.update`. Future runs (incl. the re-run) will
record both splits. The val-vs-test headline *choice* is deliberately deferred —
the data is now present so the thesis can pick test (recommended) without a re-run.

---

## Bug #4 — [LOW] `edge_contrastive` (λ_edge=0.05) underperforms controls

**What was broken / status.** Not a code defect — an empirical signal: the
λ_edge=0.05 contrastive aux sits below both `zonly_fn` and the `edge_scorer_no_aux`
control at 1 seed, trending the wrong way. Cannot be separated from noise at n=1.

**What was fixed.** No code change (would be premature). Flagged for re-examination
once the re-run gives multiple seeds / error bars. The newly-fixed H4 rare-AUROC
column (Bug #2) actually sharpens the concern: edge features *reduce* rare-class
AUROC at seed 42 (0.6538 with edge features vs 0.6868 z-only).

---

## Bug #5 — [LOW] Tables printed `± 0.0000` for n=1

**What was broken.** `_fmt_metric` set std=0 for single-seed cells, printing
`± 0.0000`, which reads as "zero variance / significant" rather than "no error bars".

**What was fixed.** `_fmt_metric` now prints `<mean> (n=1)` for single-seed values.
Self-resolves to real `± std` once ≥2 seeds are present.

---

## Verification performed

- `python scripts/make_paper_tables.py --input runs/results/ablation_1seed_master.csv`
  regenerated all four tables cleanly; H4 rare-AUROC populated, n=1 tagged, H1 gate
  correctly flags `kan_wide_infonce` as not-yet-run.
- Wide-KAN head instantiates to **737,538** params (matches the documented comment).
- `pytest tests/ -q` → **218 passed, 1 failed**. The single failure
  (`test_required_directories_exist`) is unrelated and pre-existing: it asserts the
  git-ignored ephemeral dir `runs/figures` exists, which is absent in this checkout.
  `tests/test_fastkan.py` passes with 100% coverage of `src/models/kan/fastkan.py`.

---

## Post-Fix Review (re-run pass)

Re-run by the `review-agent` over the changed files (2026-05-29). Verdict: **all
five fixes verified correct; no new problems introduced; both audit must-fix items
(FIX 1 H1 confound, FIX 2 headline split) addressed.**

**Verified OK**
- Bug #3: the four new test-column keys in `ablate.py` (`_COLUMNS` + `_run_cell`
  `row.update`) **exactly match** the keys `probe.py` writes (`_CSV_COLUMNS` and its
  `row` dict). No typos → no silent-blank risk.
- Bug #2: H4 now reuses `_extract_rare_auroc` from `per_class_auroc_linear_json`,
  mirroring H2 exactly; H1/H2/H3 builders behaviourally untouched.
- Bug #5: `_fmt_metric` n=1 path returns `(n=1)`; n≥2 and `_fmt_params` paths
  unaffected.
- Bug #1: independently recomputed wide head = 737,538 (L1 589,953 + L2 147,585)
  and narrow head = 328,507 — both match the config comments to the digit.
  `FastKANProjector` imposes no `hidden ≤ input` constraint, so the wide config
  instantiates. R1 deviation documented per Rule 5 in all three touched configs.
- R9/R10: no bare excepts added; edited files live in `scripts/` (R10 targets
  `src/`).

**LOW (operational, not code defects)**
1. `configs/model/kan_wide_head.yaml` and `configs/experiment/full_kan_wide_infonce.yaml`
   are still **untracked**, while the `ablation.yaml` that references them is modified.
   They MUST be staged in the same commit, or the `kan_wide_infonce` cell will fail
   at train time with a missing-config error.
2. `reports/audit_1seed_ablation.md` still describes the pre-fix state (stale text,
   not a code issue) — left as the historical record; this file supersedes it.

**Pre-existing, unrelated:** `tests/test_smoke.py::test_required_directories_exist`
fails because the git-ignored ephemeral dir `runs/figures` is absent in this
checkout. Not caused by these fixes.
