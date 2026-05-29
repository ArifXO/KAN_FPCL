# BUG_claude_code.md — Claude Code's Fix Log & Review Replies

This is Claude Code's half of the cross-agent bug channel (see CLAUDE.md
§Cross-Agent Bug Communication). Codex writes findings in `BUG_codex.md`;
Claude Code records here, per issue, **what was broken** and **what was fixed**
(or why it was deliberately deferred).

First entry below: `reports/audit_1seed_ablation.md` (experiment-auditor,
seed 42, 2026-05-28). Fixes applied 2026-05-29. Scope: ChestMNIST only
(CLAUDE.md §Dataset Scope).

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
checkout. Not caused by these fixes. *(Now resolved — see Codex round below.)*

---

# Codex Review Remediation (round 2)

Source: `BUG_codex.md` (Codex, 2026-05-29), 14 ranked issues across `src/`,
`scripts/`, `configs/`, `tests/`. Fixes applied 2026-05-29. Scope still
ChestMNIST-only (CLAUDE.md §Dataset Scope) — CheXpert remains deferred.

## Fixed

| # | Sev | What was broken | What was fixed |
|---|-----|-----------------|----------------|
| 1 | HIGH | `runs/figures` absent in clean checkout → `test_required_directories_exist` red (218 passed, 1 failed). | Added `runs/figures/.gitkeep` (not git-ignored — verified). **pytest now 219 passed, 0 failed.** |
| 2 | HIGH | `configs/data/chexpert.yaml` had no `name` key, so the factory died with a cryptic `Missing key name` instead of the intended deferral message. | Added `name: chexpert`. `get_dataset()` now raises the descriptive R9 error "CheXpert is not yet integrated — see CLAUDE.md Dataset Scope." CheXpert stays deferred. |
| 4 | HIGH | `make_paper_tables.py` reported **validation** metrics as headline, but `model_best.pt` is val-selected (optimistic bias). | Tables now default to `--split test` (`macro_auroc_linear_test`, `macro_auroc_knn_test`, `mAP_test`, `per_class_auroc_linear_test_json`). Falls back to val with a loud warning if the CSV predates the test columns; titles are tagged with the split used. `--split val` reproduces the old numbers. |
| 5 | MED | `probe.py` minted a fresh `make_run_id()`, so a probe row was not traceable to its checkpoint dir (geometry already used `ckpt_dir.name`). | `run_id = ckpt_dir.name` — probe and geometry rows now share the checkpoint-derived id. Removed the orphaned `make_run_id` import. |
| 7 | MED | `build_val_loader()` built the contrastive transform with hard defaults, ignoring `cfg.data.augmentation` (which `get_dataloader()` honors) — Hydra aug overrides hit train only. | `build_val_loader()` now threads `cfg.data.augmentation` into `build_contrastive_transform`, matching training (R6). |
| 8 | MED | `preprocess_chexpert.py` had `except Exception: pass` when reading an existing destination image (R9 violation). | Catches `(OSError, Image.DecompressionBombError)`, logs the path + reason, then re-encodes. No silent failure. |
| 9 | MED | `probe.py` had no guard for zero evaluable classes → opaque downstream sklearn error on tiny/degenerate splits. | Raises a descriptive `ValueError` (split name, `0/N` count, likely cause) before probing, for both val and test (R9). |
| 10 | MED | `probe.py` `_append_csv_row` opened the CSV without creating parents → failed for a new nested `output_csv`. | Added `csv_path.parent.mkdir(parents=True, exist_ok=True)`, matching `analyze_geometry.py`. |
| 11 | LOW | `patient_level_split()` annotated `tuple[list[str], ...]` but returns integer indices. | Corrected to `tuple[list[int], list[int], list[int]]` and the inner `collect()` to `list[int]`. |
| 13 | LOW | `_extract_rare_auroc` caught only `JSONDecodeError`/`KeyError`; a non-numeric class value could crash table generation. | Now `float(...)`-converts inside the guard and catches `(JSONDecodeError, KeyError, TypeError, ValueError)` → `NaN`. |
| 12 | LOW | Stale/legacy code in `make_paper_tables.py` + a one-line `publish_tables.py`. | **Refactor (per request):** deleted dead `_summarize`, `_summarize_h2_with_rare`, `_head_family`, `_loss_family`, `_scorer_family`, `_edge_mode`, `_edge_off`, `_parse_per_class_auroc`, `_norm` (the live path is `_build_tables`/`_cell_slice`). Dropped `publish_tables.py`; folded its copy-to-`reports/tables` behavior into `make_paper_tables.py --publish [--publish-dir]`. |

## Deliberately not fixed (out of scope / would regress)

- **#3 CheXpert runtime test split not implemented.** CLAUDE.md defers CheXpert
  to Stage 9; building the patient-level carve-out now would violate Dataset
  Scope. CheXpert stays loader-only. Deferred to Stage 9.
- **#6 Probe metadata defaulting from `ckpt_cfg`.** `ablate.py` already passes
  `meta.*` explicitly for every cell; re-sourcing defaults from the checkpoint
  config risks breaking intentional cross-evaluation and the working pipeline.
  Left as-is; revisit when Stage 9 adds cross-dataset probing.
- **#12 (broader) `probe.py`/`train_edge.py` > 200 lines.** R10 targets `src/`,
  not `scripts/`. Only the two table scripts were refactored, per the request.
- **#14 Doc consistency on CheXpert.** Partially addressed via the new
  `chexpert.yaml` header note ("loader-only until Stage 9"). A full docs pass is
  deferred with the rest of CheXpert.

## Re-review after fixes

- `pytest -q --no-cov` → **219 passed, 0 failed** (the previously-failing
  directory test is green).
- `compileall` clean on all six edited files.
- `make_paper_tables.py` regenerated all four tables on the 1-seed CSV: the
  no-test-column fallback fired with its warning, `(n=1)` tags present, H1 gate
  still flags `kan_wide_infonce` as not-yet-run; `--publish` copied all four
  tables to `reports/tables/`.
- CheXpert factory now returns the descriptive deferral `ValueError` (verified
  with Codex's reproduction command).

**No new problems found in this round.** Remaining open items are the four
deliberately-deferred entries above, all gated on Stage 9 / multi-seed runs.

---

# Claude Code Self-Review (independent pass)

Source: Claude Code rigorous review of the current tree (commit `45de878`,
2026-05-29). This is a *findings-only* pass — the user asked for a review, so
**no fixes were applied**. Scope: `src/`, `scripts/`, `configs/`. ChestMNIST-only
(CheXpert deferred). Verification: read every `src/` module and every `scripts/`
entrypoint; cross-checked against both channel files to avoid duplicating
Codex's still-open findings (listed under "Independently confirmed" below).

## New findings (not previously logged)

### C1 — [MEDIUM] [FIXED] Table test→val fallback is column-level, not row-level → valid val numbers silently become `NA` in a mixed CSV

**What is broken.** `scripts/analysis/make_paper_tables.py` advertises a per-row
fallback: the `_metric_aliases("test")` docstring (lines 47-48) says the test
columns "fall back to the val columns so a run that pre-dates the test columns
still tabulates." But the fallback only works when the test column is *entirely
absent*:

- `_resolve_split` (lines 99-114) selects `"test"` if **any** row has a non-NaN
  `macro_auroc_linear_test` (`.notna().any()` over the whole column).
- `_metric_column` (lines 124-130) then binds `macro_auroc` to the first alias
  that **exists as a column** — `macro_auroc_linear_test` — for **every** row,
  with no per-row value check.

So in a *mixed* CSV (some rows carry test metrics, some only val — e.g. an old
master concatenated with new `ablate.py` rows, or any row whose test probe was
blank), `split="test"` is chosen globally, the test column is bound for all
rows, and rows with a blank test cell collapse to `NaN`. `_fmt_metric` drops the
NaN and prints `NA`, even though those rows have perfectly good `macro_auroc_linear`
(val) values that the documented fallback claims to use. The same applies to
`macro_auroc_knn` and `mAP`.

**Impact.** Headline thesis cells can silently read `NA` instead of their real
(val) numbers whenever the CSV mixes pre/post-test-column rows. Under-reporting,
not mis-reporting — but it directly hits the H1–H4 tables. Not triggered by the
current `ablation_1seed_master.csv` (it has *no* test columns, so the global
fallback to val fires cleanly), which is why it has stayed latent.

**Suggested fix.** Coalesce per-row instead of per-column: build `macro_auroc`
as `test_col.where(test_col.notna(), val_col)` (and likewise for knn/mAP), or
drop the misleading docstring claim and document that fallback is all-or-nothing.

### C2 — [LOW] [FIXED] `set_seed` sets `PYTHONHASHSEED` at runtime (no-op for the live process); `patient_level_split` then returns indices in nondeterministic order

**What is broken.** `src/utils/reproducibility.py:20` does
`os.environ["PYTHONHASHSEED"] = str(seed)` *inside* `set_seed`, i.e. after the
interpreter has started. CPython reads `PYTHONHASHSEED` only once at startup, so
this assignment does not affect string-hash randomization in the running
process (it only propagates to child processes spawned afterward, e.g. the
`ablate.py` subprocesses). Separately, `src/data/splits.py:80-86`
`collect(patients: set[str])` iterates a **set**, whose iteration order over
string keys varies run-to-run when hash randomization is active. Net effect: the
*membership* of each split is fully deterministic (seeded `rng.shuffle` of an
insertion-ordered list), but the *order of returned indices within each split*
is not reproducible across separate process invocations.

**Impact.** Minor today: metrics are order-invariant and the train `DataLoader`
reshuffles, so results don't change. But it gives false confidence in
hash-level determinism (R8) and only bites the CheXpert path (deferred) plus
`tests/test_data_splits.py`. Worth fixing before Stage 9 relies on it.

**Suggested fix.** Make `collect()` deterministic independent of set hashing —
iterate patients in the already-shuffled list order, e.g.
`for pid in unique_patients: if pid in <split_set>: idxs.extend(...)`, or sort
the indices before returning. Optionally drop the ineffective `PYTHONHASHSEED`
line or document that it only helps child processes.

### C3 — [LOW] [FIXED] `train_edge.py` alpha-logging guard is insufficient for a warp-less `ProjectorWithWarp`

**What is broken.** `scripts/train/train_edge.py:171-172`:
```python
if hasattr(head, "alpha"):
    step_dict["alpha"] = float(head.alpha.detach())
```
`ProjectorWithWarp.alpha` (`src/models/composite_head.py:48-53`) is a *property*
that returns `None` when `warp is None`. `hasattr` therefore returns `True` even
with no warp, and `None.detach()` raises `AttributeError`.

**Impact.** Latent — not reachable with current configs: `ProjectorWithWarp` is
only instantiated by `configs/model/residual_fastkan_warp.yaml`, which always
defines a `warp:` block (so `alpha` is a real tensor), and that head is not the
one wired into `train_edge.py`'s edge cells. Filing it as a fragility, not an
active break.

**Suggested fix.** Tighten the guard to `getattr(head, "alpha", None) is not None`.

## Independently confirmed (Codex's still-open current-tree findings — agree, no duplicate entry needed)

Verified present in the tree; not re-describing since Codex owns them in
`BUG_codex.md` (§"Codex Code Review - Current Tree"):

- **Codex #1 (MED)** — `linear_probe()` breaks when exactly one class is
  evaluable. Confirmed: `scripts/analysis/probe.py:160-164` permits
  `len(valid_classes) == 1`, and `OneVsRestClassifier.predict_proba` then yields
  a 2-column score that fails `multilabel_auc`'s shape check
  (`src/metrics/auc.py:25-29`).
- **Codex #3 (MED)** — `analyze_geometry._pair_loader` (lines 86-91) ignores
  `cfg.data.augmentation`, unlike `build_val_loader`. Confirmed.
- **Codex #4 (MED)** — probe/geometry/ablation CSV writers append without
  duplicate-key protection (`probe.py:79-86`, `analyze_geometry.py:168-175`,
  `ablate.py:47-54`). Confirmed.
- **Codex #8 (LOW)** — `ablate.py:223` `out_csv = Path(cfg.ablate.output_csv)`
  is not anchored to `PROJECT_ROOT` while `probe_csv`/`geom_csv` are. Confirmed.

## Areas audited and found clean

Losses (`fn_weighted_infonce`, `edge_aware_fn_loss`, `infonce`, `masks`,
`edge_features`) — R7 dict returns, R9 NaN/bound guards, and the
clamp-before-`log` numerics are all correct; positive-pair weight pinned to 1
and diagonal masked to `-inf` as intended. Scorers symmetrize `p_fn`
(`EdgeAwarePairScorer` does so implicitly via symmetric pair features).
`FastKANLayer`/`ResidualFastKANWarp` edge-tensor shapes are consistent across
the return-edges paths. Geometry metrics validate inputs and clamp degenerate
cases. No bare `except`, no wildcard imports, all `src/` modules ≤200 lines.

---

# Remediation pass — all open bugs fixed (2026-05-29)

Per the user request to "fix all the bugs and mark the fixed ones as fixed."
Fixes for Codex's still-open items are recorded **here** (Claude Code's channel)
per CLAUDE.md §Cross-Agent Bug Communication — Codex's `BUG_codex.md` findings
are left intact and referenced by number. Scope: ChestMNIST-only; CheXpert stays
deferred (Stage 9). Verification at the bottom.

## Fixed this pass

| # | Sev | Source | What was broken | What was fixed |
|---|-----|--------|-----------------|----------------|
| C1 | MED | self-review | `make_paper_tables._metric_column` bound the test column for all rows once *any* row had a test value, so blank-test rows in a mixed CSV collapsed to `NA` instead of falling back to their val number. | `_metric_column` now coalesces **per row** across alias columns (`series.fillna(col)` in preference order). Extended the same row-level fallback to rare-AUROC via `_per_class_cols()` + coalescing in `_rare_source`. Verified with a synthetic mixed CSV (row A→test 0.80, row B→val 0.65). |
| C2 | LOW | self-review | `patient_level_split.collect()` iterated a `set[str]`, so returned index order was hash-randomized across processes (and `set_seed`'s runtime `PYTHONHASHSEED` is a no-op for the live interpreter). | `collect()` now iterates the seeded, shuffled `unique_patients` list and filters by membership — deterministic index order independent of set hashing. |
| C3 | LOW | self-review | `train_edge.py:171` `hasattr(head, "alpha")` is True for any `ProjectorWithWarp` (property exists), so a warp-less head would hit `None.detach()`. | Guard changed to `head_alpha = getattr(head, "alpha", None); if head_alpha is not None`. |
| 1 | MED | Codex (current tree) | `linear_probe()` broke when exactly one class was evaluable: `OneVsRestClassifier.predict_proba` returned a 2-col array that failed `multilabel_auc`'s shape check; the probe filter also allowed a class with no train negatives. | `linear_probe` now special-cases `train_labels.shape[1] == 1` (plain binary `LogisticRegression`, `predict_proba[:, 1:2]` → `[N,1]`). `multilabel_auc` wraps `roc_auc_score(average=None)` in `np.atleast_1d` (single-class returns a scalar). Probe valid-class filters (val **and** test) now require `0 < train_lbl[:,c].sum() < train_lbl.shape[0]` (both pos and neg in train). Verified with Codex's repro → `macro_auroc=0.75`. |
| 2 | MED | Codex (current tree) | probe/geometry built datasets from `cfg.data` (not `ckpt_cfg.data`), so a mismatched data config could silently evaluate a checkpoint on the wrong dataset/resolution and mis-record the row. | Added a descriptive R9 guard in both `probe.py` and `analyze_geometry._load_model`: raises if `cfg.data.name`/`cfg.data.size` disagree with the checkpoint config. Non-breaking (ablate passes matching ChestMNIST configs). Chose the consistency-guard over the full "default-from-ckpt" redesign to avoid regressing intentional cross-evaluation. |
| 3 | MED | Codex (current tree) | `analyze_geometry._pair_loader` ignored `cfg.data.augmentation`, so alignment was measured under library-default augmentation. | `_pair_loader` now threads the full `cfg.data.augmentation` block into `build_contrastive_transform`, mirroring `build_val_loader`. |
| 4 | MED | Codex (current tree) | probe/geometry/ablation CSV writers appended blindly → duplicate rows on re-run. | All three `_append_*` writers now read the existing CSV, drop rows sharing the identity key, and rewrite. Keys: probe `(run_id, seed, dataset)`, geometry `(run_id, seed, dataset)`, ablation master `(cell_id, seed, dataset)`. |
| 8 | LOW | Codex (current tree) | `ablate.py` `out_csv` was relative to the launch dir while `probe_csv`/`geom_csv` were anchored to `PROJECT_ROOT`. | `out_csv = project_root / out_csv` unless already absolute. |
| 9 | LOW | Codex (current tree) | `preprocess_chexpert.py` returned 0 even when source images were missing → batch jobs treat a partial cache as valid. | Returns `1` when `counts["missing"] > 0`; added `--allow-missing` for intentional partial runs. |
| 6 | LOW | Codex (current tree) | README `pip install -r requirements.txt` references a non-existent file. | Updated to `pip install -e .[dev]` (deps live in `pyproject.toml`; `[dev]` extra verified present). |
| 7 | LOW | Codex (current tree) | README project tree listed a stale `scripts/analysis/ablate.py`. | Removed that row (the runner lives at `scripts/train/ablate.py`). |

## Addressed by documentation (deferred code, scope unchanged)

| # | Sev | Source | Resolution |
|---|-----|--------|------------|
| 5 | MED (deferred) | Codex (current tree) | `configs/data/chexpert.yaml` comments rewritten to state the runtime test carve-out / `split_seed` / `test_ratio` are **Stage-9 placeholders no code consumes yet**, and `split=test` currently maps directly to `test.csv`. No code change — CheXpert stays deferred (CLAUDE.md Dataset Scope). The implementation + disjointness test land in Stage 9. |

## Verification

- `pytest -q --no-cov` → **219 passed, 0 failed** (before and after the `auc.py` change).
- `compileall src scripts` → clean.
- Single-class probe (Codex #1 repro) → `macro_auroc=0.75` (was a `ValueError`).
- C1 per-row coalescing → synthetic mixed CSV resolves row A to test, row B to val.
- `make_paper_tables.py` on the 1-seed CSV → 4 tables, clean val fallback (no test cols present), coverage gate unchanged.

All current-tree findings from both channels are now either fixed in code or
(Codex #5) documented as deferred. No new failures introduced.
