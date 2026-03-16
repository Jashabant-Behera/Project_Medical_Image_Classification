# Dataset Overview

This document outlines the structure and content of the dataset used for the **Medical Image Classification** project. This dataset contains Chest X-Ray images designed for a binary classification task to determine whether a lung scan is "NORMAL" or indicates "PNEUMONIA".

## Directory Structure

```text
Dataset/
├── Dataset.zip                      # The original downloaded dataset archive (~2.4 GB)
└── chest_xray/                      # Extracted dataset folder
    │
    ├── __MACOSX/                    # (Hidden macOS formatting files, can be safely ignored or deleted)
    ├── chest_xray/                  # (A nested duplicate folder created during extraction, common with zipped files)
    │
    ├── train/                       # 📂 Training Dataset (Used to train the ML model)
    │   ├── NORMAL/                  # ├─ Contains 1,341 normal chest X-ray images
    │   └── PNEUMONIA/               # └─ Contains 3,875 pneumonia chest X-ray images
    │
    ├── test/                        # 📂 Test Dataset (Used for final evaluation)
    │   ├── NORMAL/                  # ├─ Contains 234 normal chest X-ray images
    │   └── PNEUMONIA/               # └─ Contains 390 pneumonia chest X-ray images
    │
    └── val/                         # 📂 Validation Dataset (Used for checking metrics during training epochs)
        ├── NORMAL/                  # ├─ Contains 8 normal chest X-ray images
        └── PNEUMONIA/               # └─ Contains 8 pneumonia chest X-ray images
```

## Key Considerations for the Machine Learning Pipeline

1. **Classes**: As shown, the target variable covers exactly **2 distinct categories**: `NORMAL` vs `PNEUMONIA`.
2. **Class Imbalance**: Evaluating the `train` folder, the data leans heavily towards the `PNEUMONIA` class (3,875 examples) against `NORMAL` (1,341 examples). The training pipeline constructed in `ml/training/augmentations.py` and `ml/training/train.py` must account for this disparity, utilizing techniques such as Weighted Loss Functions (e.g., class weighting in Cross Entropy) or undersampling/oversampling to ensure unbiased results.
3. **Small Validation Set**: The default validation split provided within the `val` folder is exceptionally small, comprising only 16 images total. For a larger sample size validating model robustness at each epoch, consider adjusting the data loading logic in `dataset.py` to carve out a random percentage (e.g. 10%–20%) from the training split.