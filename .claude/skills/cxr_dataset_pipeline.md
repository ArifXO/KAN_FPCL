---
name: cxr-dataset-pipeline
type: skill
description: Ensure patient-level splits, uncertainty handling, augmentation correctness for medical imaging
---

# CXR Dataset Pipeline

**When to Activate:** Stages 1, 9

**Expertise:**
- Patient-level dataset splits (not row-level)
- Data leakage detection and prevention
- Uncertainty handling in medical labels
- Augmentation strategies for multi-label chest X-ray classification
- MedMNIST / ChestMNIST API and schema

**Guides Development Of:**
- ChestMNIST dataset wrapper with patient ID extraction (Stage 1)
- Train/Val/Test split with patient-level disjointness verification (Stage 1)
- Augmentation pipeline for contrastive learning (Stage 1)
- Final dataset audit and split statistics (Stage 9)

**Critical Checks:**
- Extract patient ID from image ID (ChestMNIST format)
- Verify no patient appears in multiple splits:
  ```python
  assert len(train_ids & val_ids) == 0
  assert len(val_ids & test_ids) == 0
  assert len(train_ids & test_ids) == 0
  ```
- Raise `ValueError` if overlap detected
- Document split composition: "Train: 10K images from 2.5K patients"

**Augmentation Considerations:**
- Strong augmentations for contrastive pairs (crop, rotation, color jitter)
- Preserve pathological signal; avoid washing out small findings
- Contrastive pairs: same image + augmentation vs. different image + augmentation

**Output Format:**
```python
DatasetSplit = {
    "images": Tensor [N, 1, H, W],
    "labels": Tensor [N, 14],  # 14-class multi-label
    "patient_ids": List[str],  # for disjointness verification
}
```

**Related:** [[R4]] [[R5]] [[R9]]
