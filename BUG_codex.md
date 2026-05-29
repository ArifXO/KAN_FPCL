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
