# KAN-FPCL

**KAN-FPCL: Functional Pathway Contrastive Learning
with KAN Projectors for False-Negative-Aware
Multi-Label Chest X-ray Representation Learning
**

CSE400 Senior Research Project — Khan Iftekhar

---

## Overview

Chest X-ray contrastive learning faces a uniquely difficult false-negative problem. In standard frameworks like SimCLR or MoCo, every sample not paired with the anchor is treated as a negative. This assumption breaks in chest radiography because (a) different patients frequently share the same pathology, (b) labels are often uncertain and report-derived, and (c) important findings occupy small image regions that are washed out by global pooling. When two images share a disease but are treated as negatives, the loss pushes apart representations that should be close, corrupting downstream performance.
The thesis investigates whether Kolmogorov-Arnold Network (KAN) projection heads and a false-negative-aware contrastive objective can address this. More importantly, it introduces a novel direction: using the internal edge-function responses unique to KAN as an additional training signal that is impossible to extract from a standard MLP
.

---

## Hypotheses

### H1 — Representation Quality
>KAN projectors learn smoother latent geometries than parameter-matched MLPs on multi-label medical images.

### H2 — Label Efficiency
> False-negative-aware contrastive masking improves recall-sensitive downstream metrics.

### H3 — Interpretability via Spline Visualization
> A KAN-based pair scorer for false-negative probability outperforms an MLP pair scorer at matched parameter count.

### H4 — Contrastive Mask Sensitivity
> Edge-level signals derived from KAN projection heads improve false-negative estimation and downstream metrics over z-only pair scoring.

---

## Project Structure

```
cxr-kan-contrastive/
├── src/
│   ├── data/          # Dataset wrappers, splits, augmentations
│   ├── models/
│   │   └── kan/       # KAN backbone and MLP baseline
│   ├── losses/        # Contrastive and classification losses
│   ├── metrics/       # Evaluation metrics
│   └── utils/         # Logging, checkpointing, artifacts
├── configs/
│   ├── data/          # Dataset and augmentation configs
│   ├── model/         # Architecture configs
│   ├── loss/          # Loss function configs
│   └── experiment/    # Full experiment configs (Hydra)
├── tests/             # Unit and integration tests
├── scripts/           # Training and evaluation entry points
└── reports/           # Figures and tables for the paper
```

---

## Setup

```bash
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v --cov=src
```

## Running an Experiment

```bash
python scripts/train.py experiment=baseline_mlp_supcon
```

---

## Scientific Rules

See [CLAUDE.md](CLAUDE.md) for the 10 scientific rules governing all development (R1–R10).

## Agent Responsibilities

See [AGENTS.md](AGENTS.md) for subagent specializations and hand-off protocol.

## Open Tasks

See [TODO.md](TODO.md) for the current task list.
