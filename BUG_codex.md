# Bug Mix - Rigorous Codebase Review

Author: Codex
Date: 2026-05-29

Scope reviewed: `src/`, `scripts/`, `configs/`, `tests/`, `README.md`, `CLAUDE.md`, `AGENTS.md`, and existing run/report artifacts.

Verification run:

- `.\.venv\Scripts\python.exe -m pytest -q --no-cov`
  - Result: 218 passed, 1 failed.
  - Failing test: `tests/test_smoke.py::test_required_directories_exist`.
- `.\.venv\Scripts\python.exe -m compileall -q src scripts tests`
  - Result: passed.
- `get_dataset(OmegaConf.load("configs/data/chexpert.yaml"), "train", None)`
  - Result: fails with `ConfigAttributeError: Missing key name`.

## High Severity

### 1. Test suite is currently red because `runs/figures` is missing

Evidence:

- `tests/test_smoke.py:22` defines `test_required_directories_exist`.
- `tests/test_smoke.py:36` requires `runs/figures`.
- Full pytest result: `1 failed, 218 passed`.

Impact:

CI/stage-gate validation fails in a clean checkout even though most code tests pass.

Suggested fix:

Add `runs/figures/.gitkeep` or remove `runs/figures` from the required-directory smoke test if the directory is no longer part of the project contract.

### 2. CheXpert config cannot run through the shared data factory

Evidence:

- `configs/data/chexpert.yaml` has `_target_: src.data.CheXpertDataset`, but no `name` key.
- `src/data/__init__.py:32` does `name = data_cfg.name.lower()`.
- `src/data/__init__.py:42` explicitly rejects CheXpert with "not yet integrated".

Reproduction:

```powershell
.\.venv\Scripts\python.exe -c "from omegaconf import OmegaConf; from src.data import get_dataset; cfg=OmegaConf.load('configs/data/chexpert.yaml'); get_dataset(cfg, 'train', None)"
```

Observed result:

`omegaconf.errors.ConfigAttributeError: Missing key name`

Impact:

Stage 9 CheXpert training/probe/geometry cannot execute through the same pipeline used by the experiment scripts.

Suggested fix:

Add `name: chexpert`, export/import `CheXpertDataset` in `src/data/__init__.py`, and implement a `get_dataset()` branch that passes root/view/uncertainty/image-size settings.

### 3. CheXpert test split behavior is documented but not implemented

Evidence:

- `configs/data/chexpert.yaml:13-15` says the test set is built from disjoint train patients using `patient_level_split`.
- `configs/data/chexpert.yaml:41-46` defines `split_seed` and `test_ratio`.
- `src/data/chexpert.py:142` uses `csv_name = "valid.csv" if split == "val" else f"{split}.csv"`.
- No code uses `split_seed`, `test_ratio`, or `patient_level_split` inside the CheXpert runtime path.

Impact:

`split="test"` attempts to load `test.csv`, which may not have labels in the official CheXpert layout. The promised patient-level train/test carve-out does not exist.

Suggested fix:

Implement CheXpert split construction in the data factory or a dedicated CheXpert data module: load train manifest, split train/test by patient ID, use official `valid.csv` for val, and verify no patient appears in more than one split.

### 4. Paper tables still report validation metrics as headline metrics

Evidence:

- `scripts/make_paper_tables.py:33-35` maps `macro_auroc` to `macro_auroc_linear`, not `macro_auroc_linear_test`.
- H2/H4 rare AUROC extraction uses `per_class_auroc_linear_json`, not `per_class_auroc_linear_test_json`.
- `scripts/ablate.py` now carries test columns, but the table builder does not consume them.

Impact:

`model_best.pt` is selected on validation loss, then tables report validation AUROC/mAP. This can overstate results and conflicts with the need for unbiased thesis headline numbers.

Suggested fix:

Make the table generator default to test metrics (`macro_auroc_linear_test`, `macro_auroc_knn_test`, `mAP_test`, `per_class_auroc_linear_test_json`) or clearly label generated tables as validation-only.

## Medium Severity

### 5. `probe.py` generates a new `run_id` instead of using the checkpoint run ID

Evidence:

- `scripts/probe.py:252` sets `run_id = make_run_id()`.
- `scripts/analyze_geometry.py:196` uses `ckpt_dir.name`.
- `scripts/probe.py:79` appends rows without checking for existing rows.

Impact:

Probe rows are not directly traceable to the checkpoint directory. Re-running probe on the same checkpoint creates a new ID instead of detecting a duplicate `(checkpoint, seed, dataset)` result.

Suggested fix:

Use `ckpt_dir.name` or the checkpoint's saved metrics/config run ID as the probe `run_id`, and reject or replace duplicate rows for the same checkpoint/seed/dataset.

### 6. Probe data and metadata can silently disagree with the checkpoint

Evidence:

- `scripts/probe.py:138-144` builds datasets from `cfg.data`, not `ckpt_cfg.data`.
- `scripts/probe.py:270-274` writes `cfg.meta.*`, not metadata derived from the checkpoint config.

Impact:

A checkpoint trained on one dataset/model label can be probed and recorded under another label if CLI overrides are wrong or omitted. This is a reproducibility and reporting risk.

Suggested fix:

Default probe data/model metadata from `ckpt_cfg`, then require explicit overrides only when intentionally cross-evaluating. Validate that `cfg.meta` matches the loaded checkpoint.

### 7. Validation augmentation ignores configured augmentation values

Evidence:

- `src/data/__init__.py:63-75` reads augmentation settings from config for training.
- `scripts/train_common.py:30` calls `build_contrastive_transform(size=cfg.data.size, mean=mean, std=std)` without passing crop/rotation/jitter/flip settings.

Impact:

Hydra overrides to augmentation affect training but not contrastive validation. This violates the config-driven experiment rule and makes validation curves harder to interpret.

Suggested fix:

Centralize transform construction or pass `cfg.data.augmentation` into `build_val_loader()` with the same parameters used by `get_dataloader()`.

### 8. CheXpert preprocessing has a silent broad exception

Evidence:

- `scripts/preprocess_chexpert.py:60-64` catches `Exception` and immediately `pass`es when opening an existing destination image.

Impact:

Corrupt or unreadable destination images are silently overwritten. That may be acceptable operationally, but it violates the no-silent-failures rule and hides data-quality problems.

Suggested fix:

Catch specific PIL exceptions, log the destination path and reason, and count the event separately, e.g. `rewritten_corrupt`.

### 9. Probe has no explicit guard for zero valid classes

Evidence:

- `scripts/probe.py:159-164` computes `valid_classes`.
- `scripts/probe.py:225-239` slices labels using that list and then calls sklearn probe/kNN.

Impact:

Tiny smoke subsets or highly imbalanced CheXpert splits can produce zero valid classes, leading to a downstream sklearn/metric error instead of a clear project-level error.

Suggested fix:

Raise a descriptive `ValueError` when `valid_classes` or `valid_classes_test` is empty, including positive/negative counts per class.

### 10. Probe CSV writer does not create output parent directories

Evidence:

- `scripts/probe.py:79-85` opens the CSV path directly.
- `scripts/analyze_geometry.py:168` creates the parent directory first, so behavior is inconsistent.

Impact:

`probe.output_csv=some/new/path/results.csv` fails if the parent directory does not exist.

Suggested fix:

Call `csv_path.parent.mkdir(parents=True, exist_ok=True)` before opening the file.

## Low Severity / Maintainability

### 11. `patient_level_split()` return type annotation is wrong

Evidence:

- `src/data/splits.py:37-41` annotates `tuple[list[str], list[str], list[str]]`.
- `src/data/splits.py:80-86` returns sample indices, not patient ID strings.
- `tests/test_data_splits.py` also treats outputs as integer indices.

Impact:

Runtime behavior is correct, but type hints and docs are misleading and can cause mistakes in downstream code.

Suggested fix:

Change the return type to `tuple[list[int], list[int], list[int]]` and update the inner `collect()` annotation.

### 12. Several scripts exceed 200 lines and duplicate training/evaluation patterns

Evidence:

- `scripts/make_paper_tables.py`: 450 lines.
- `scripts/probe.py`: 265 lines.
- `scripts/train_edge.py`: 236 lines.

Impact:

The R10 line limit applies to `src/`, not scripts, so this is not a direct rule failure. It is still a maintenance risk: checkpoint loading, CSV appending, metadata handling, and training loops are duplicated.

Suggested fix:

Move shared checkpoint/probe CSV helpers and train-loop utilities into small modules under `src/utils/` or `scripts/common/`, then keep entrypoints thin.

### 13. `make_paper_tables.py` rare-AUROC parser is not fully defensive

Evidence:

- `scripts/make_paper_tables.py:313-322` catches `json.JSONDecodeError` and `KeyError`, but not `TypeError` or `ValueError` from malformed non-numeric class values.

Impact:

A malformed per-class JSON value can crash table generation instead of becoming `NA`.

Suggested fix:

Convert values inside a try block and catch `(json.JSONDecodeError, TypeError, ValueError, KeyError)`.

### 14. Documentation is inconsistent about CheXpert readiness

Evidence:

- `CLAUDE.md` says CheXpert is not integrated yet.
- `README.md` describes ChestMNIST/CheXpert success criteria and Stage 9.
- `configs/data/chexpert.yaml` has implementation-facing comments for a runtime split that is not implemented.

Impact:

It is easy to misread CheXpert as runnable in the normal pipeline when it is currently only partially scaffolded.

Suggested fix:

Mark CheXpert as "loader-only / not factory-integrated" everywhere until Stage 9 is actually wired into `get_dataset()`, training, probe, and geometry.

## Checks That Look Good

- All `src/{models,losses,metrics,data}` modules are under 200 lines.
- No wildcard imports were found in `src/` or `scripts/`.
- Loss modules return dicts with required diagnostic keys.
- FN-weighted loss clamps weights before `log()`.
- Edge fingerprint compression emits fixed 256-dimensional fingerprints.
- Loss/scorer tests cover gradients, invalid inputs, masking behavior, and edge-aware pathways.
- Hydra composition tests passed during the full test run.

## Recommended Fix Order

1. Add `runs/figures/.gitkeep` or update the smoke test so pytest is green.
2. Decide whether CheXpert is still deferred. If yes, make docs/configs explicitly say "not runnable"; if no, wire it into `get_dataset()` and implement the promised patient-level test carve-out.
3. Switch table generation to test metrics or rename current tables as validation tables.
4. Fix probe traceability: checkpoint run ID, duplicate protection, parent directory creation, and metadata validation.
5. Remove silent exception handling from CheXpert preprocessing.
6. Align validation augmentation with Hydra config.

---

# Codex Code Review - Current Tree

Author: Codex
Date: 2026-05-29

Scope reviewed: `src/`, `scripts/`, `configs/`, `tests/`, `README.md`,
`CLAUDE.md`, `AGENTS.md`, `BUG_codex.md`, `BUG_claude_code.md`, and current
run/report artifacts.

Verification run:

- `.\.venv\Scripts\python.exe -m pytest -q --no-cov`
  - Result: `219 passed in 11.88s`.
- `.\.venv\Scripts\python.exe -m compileall -q src scripts tests`
  - Result: passed.
- `.\.venv\Scripts\python.exe -m scripts.train.ablate --help`
  - Result: passed; config uses the new `scripts.train.*` and
    `scripts.analysis.*` module paths.
- `.\.venv\Scripts\python.exe -m scripts.analysis.probe --help`
  - Result: passed.
- `.\.venv\Scripts\python.exe -m scripts.analysis.analyze_geometry --help`
  - Result: passed.
- `.\.venv\Scripts\python.exe -m scripts.analysis.make_paper_tables --help`
  - Result: passed.
- `.\.venv\Scripts\python.exe -m scripts.analysis.make_paper_tables --input runs\results\ablation_1seed_master.csv --output-dir runs\tables`
  - Result: passed; warned correctly that the old CSV predates test metrics and
    fell back to validation metrics.
- CheXpert factory deferral check:
  - `get_dataset(OmegaConf.load("configs/data/chexpert.yaml"), "train", None)`
  - Result: raises the intended descriptive `ValueError`:
    `CheXpert is not yet integrated - see CLAUDE.md Dataset Scope.`

## Open Findings

### 1. Linear probe breaks when exactly one class remains evaluable

Severity: Medium

Evidence:

- `scripts/analysis/probe.py:160-164` keeps a class when val has positives and
  negatives and train has at least one positive.
- `scripts/analysis/probe.py:190-194` applies the same rule for test.
- That filter allows `len(valid_classes) == 1`.
- `src/metrics/linear_probe.py:44-49` always wraps `LogisticRegression` in
  `OneVsRestClassifier` and passes `clf.predict_proba(x_val)` directly to
  `multilabel_auc`.

Reproduction:

```powershell
.\.venv\Scripts\python.exe -c "import numpy as np; from src.metrics.linear_probe import linear_probe; train_emb=np.random.RandomState(0).randn(6,4).astype('float32'); val_emb=np.random.RandomState(1).randn(4,4).astype('float32'); train_labels=np.array([[0],[1],[0],[1],[0],[1]], dtype=np.int32); val_labels=np.array([[0],[1],[0],[1]], dtype=np.int32); print(linear_probe(train_emb, train_labels, val_emb, val_labels))"
```

Observed result:

`ValueError: scores and labels must have the same shape, got (4, 2) vs (4, 1)`

Impact:

Probe can fail on tiny smoke subsets or filtered CheXpert-style splits even
when one class is legitimately evaluable. The current zero-evaluable-class
guard does not catch this because one valid class is enough to pass it.

Suggested fix:

Handle the single-class case explicitly in `linear_probe()` by fitting a plain
binary `LogisticRegression` and using `predict_proba(...)[:, 1:2]`. Also update
the probe valid-class filters to require both positive and negative examples in
the training labels: `0 < train_lbl[:, c].sum() < train_lbl.shape[0]`.

### 2. Probe and geometry can evaluate a checkpoint with the wrong data config

Severity: Medium

Evidence:

- `scripts/analysis/probe.py:106-108` loads encoder/head architecture from the
  checkpoint config.
- `scripts/analysis/probe.py:139-152` builds train/val/test datasets from
  `cfg.data`, not `ckpt_cfg.data`.
- `scripts/analysis/probe.py:286-292` writes row metadata from `cfg.meta`, not
  the checkpoint config.
- `scripts/analysis/analyze_geometry.py:58-60` loads model architecture from
  the checkpoint config.
- `scripts/analysis/analyze_geometry.py:74-97` builds datasets from `cfg.data`.
- `scripts/analysis/analyze_geometry.py:196-204` writes metadata from `cfg.meta`.

Impact:

A checkpoint trained with one data size, dataset label, or metadata label can be
probed/analyzed under another if CLI overrides are omitted or wrong. Ablation
runs pass metadata explicitly, but standalone evaluation remains easy to
mis-record.

Suggested fix:

Default evaluation data and metadata from `ckpt_cfg`. Permit overrides only via
explicit flags such as `probe.allow_data_override=true`, and emit a warning or
raise when `cfg.data.name`, `cfg.data.size`, or `cfg.meta.*` disagrees with the
checkpoint config.

### 3. Geometry alignment ignores Hydra augmentation overrides

Severity: Medium

Evidence:

- `scripts/analysis/analyze_geometry.py:87-90` builds the two-view transform as
  `build_contrastive_transform(size=cfg.data.size, mean=mean, std=std)`.
- Unlike `scripts/train/train_common.py`, it does not pass
  `cfg.data.augmentation` values for crop, rotation, color jitter, or flip.

Impact:

Training and validation now honor augmentation overrides, but geometry alignment
does not. If an experiment changes augmentation strength, the geometry report is
computed under default augmentation rather than the actual experiment setting.

Suggested fix:

Mirror `build_val_loader()` and pass `cfg.data.augmentation` through
`build_contrastive_transform()` in `_pair_loader()`.

### 4. Result CSV writers still allow duplicate rows

Severity: Medium

Evidence:

- `AGENTS.md:188-189` requires probe results to avoid duplicate
  `(run_id, seed, dataset)` rows.
- `scripts/analysis/probe.py:79-86` appends rows without checking existing rows.
- `scripts/analysis/analyze_geometry.py:168-175` also appends blindly.
- `scripts/train/ablate.py:49-54` appends master rows blindly.
- `scripts/analysis/make_paper_tables.py:142-149` detects duplicate
  `(cell_id, dataset, seed)` rows later and keeps the last, which confirms
  duplicates are expected downstream rather than prevented at write time.

Impact:

Re-running probe, geometry, or ablation for the same run can silently create
duplicate source rows. Table generation masks this by keeping the last row, but
the raw CSVs remain ambiguous and violate the experiment-auditor contract.

Suggested fix:

Before appending, read the target CSV if it exists and reject or replace rows
with the same identity key:

- Probe: `(run_id, seed, dataset)`.
- Geometry: `(run_id, seed, dataset)`.
- Ablation master: `(cell_id, seed, dataset)`.

### 5. CheXpert config still promises a runtime test carve-out that does not exist

Severity: Medium, deferred by current dataset scope

Evidence:

- `CLAUDE.md:98` says CheXpert is not integrated for training yet.
- `src/data/__init__.py:42` correctly rejects CheXpert through the shared factory.
- `configs/data/chexpert.yaml:13-15` says the official val set is used and a
  runtime test carve-out is built from disjoint train patients.
- `configs/data/chexpert.yaml:48-52` exposes `split_seed` and `test_ratio`.
- `src/data/chexpert.py:142-143` still maps `split="test"` directly to
  `test.csv`; no code consumes `split_seed`, `test_ratio`, or
  `patient_level_split()`.

Impact:

This is not breaking the current ChestMNIST pipeline because CheXpert is
explicitly deferred. It is still a Stage 9 trap: the comments describe a
patient-level split guarantee that the loader does not implement.

Suggested fix:

Until Stage 9, change the config comments to say `split=test` is not implemented
through the factory. When Stage 9 starts, implement the train/test patient
carve-out in a data module or factory branch and add an integration test for
train/val/test patient disjointness.

### 6. README installation command references a missing file

Severity: Low

Evidence:

- `README.md:79` says `pip install -r requirements.txt`.
- `requirements.txt` does not exist in the repo.
- `pyproject.toml` is present and contains the dependency list.

Impact:

Fresh setup instructions fail before tests or training can start.

Suggested fix:

Either add `requirements.txt` or update the README to use the existing package
metadata, for example `pip install -e .[dev]` or `pip install -e .`.

### 7. README project structure still lists `scripts/analysis/ablate.py`

Severity: Low

Evidence:

- `README.md:198` correctly lists `scripts/train/ablate.py`.
- `README.md:204` still lists `scripts/analysis/ablate.py`.
- Current tree has `scripts/train/ablate.py`; there is no
  `scripts/analysis/ablate.py`.

Impact:

The current command examples are correct, but the project tree is stale after
the requested `ablate.py` move.

Suggested fix:

Remove the `scripts/analysis/ablate.py` row from the README structure block.

### 8. Ablation master CSV path is not anchored to the project root

Severity: Low

Evidence:

- `scripts/train/ablate.py:222-225` anchors `probe_csv` and `geom_csv` to
  `PROJECT_ROOT`, but leaves `out_csv = Path(cfg.ablate.output_csv)` relative
  to the current process directory.

Impact:

If `scripts.train.ablate` is launched from a directory other than the project
root, the intermediate probe/geometry CSVs go under the repo but the master CSV
can be written elsewhere. This makes run artifacts harder to find.

Suggested fix:

Resolve `out_csv` like the others:
`out_csv = project_root / cfg.ablate.output_csv` unless it is already absolute.

### 9. CheXpert preprocessing reports missing images but exits successfully

Severity: Low

Evidence:

- `scripts/preprocess/preprocess_chexpert.py:53-55` returns `"missing"` when a
  source image is absent.
- `scripts/preprocess/preprocess_chexpert.py:118-126` prints a warning if any
  paths are missing.
- `scripts/preprocess/preprocess_chexpert.py:127` still returns `0`.

Impact:

An automated preprocessing job can succeed at the process level while producing
an incomplete image tree. The warning is visible to a human, but CI or batch
scripts may treat the cache as valid.

Suggested fix:

Return a non-zero exit code when `counts["missing"] > 0`, or add an explicit
`--allow-missing` flag for intentionally partial debug runs.

## Checks That Look Good

- Full test suite is green: `219 passed`.
- `compileall` passes for `src`, `scripts`, and `tests`.
- Script entry points after the reorganization are importable:
  `scripts.train.ablate`, `scripts.analysis.probe`,
  `scripts.analysis.analyze_geometry`, and `scripts.analysis.make_paper_tables`.
- No stale `scripts.rest` references were found.
- All `src/{models,losses,metrics,data}` Python modules are under the R10
  200-line limit.
- No wildcard imports were found in `src/`, `scripts/`, or `tests/`.
- Previously reported issues now fixed in the current tree include:
  `runs/figures` missing, CheXpert config missing `name`, table generation
  defaulting to validation metrics, probe run IDs not matching checkpoint dirs,
  probe CSV parent creation, zero-evaluable-class probe guard, validation
  augmentation in training, broad silent CheXpert preprocessing exception,
  `patient_level_split()` return annotation, and rare-AUROC parser robustness.

## Recommended Fix Order

1. Fix `linear_probe()` for single-class inputs and strengthen the probe
   valid-class filter for train all-positive/all-negative classes.
2. Make probe/geometry default to checkpoint data and metadata, with explicit
   override guards.
3. Add duplicate-row protection to probe, geometry, and ablation CSV writers.
4. Thread augmentation overrides into geometry alignment.
5. Clarify the deferred CheXpert test-split comments or implement the Stage 9
   patient-level carve-out when CheXpert is activated.
6. Clean up README setup and project-structure drift.
