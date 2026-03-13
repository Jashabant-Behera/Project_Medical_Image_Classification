# Phase 1: Dataset Acquisition & Exploratory Data Analysis

## Objective
The primary goals of this phase were to retrieve the [Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset, reorganize it into a structured directory tree, run visualizations to detect data characteristics, and generate standardized `train.csv`, `val.csv`, and `test.csv` manifest mappings connecting filepaths to labels (`NORMAL` vs `PNEUMONIA`).

## Files Created
1. `scripts/create_csv.py` - Scans the `ml/data/raw/` subdirectories recursively to map label strings into numeric binaries.
2. `scripts/run_eda.py` - Parses dataset distribution graphs and sample montages.
3. `ml/notebooks/01_eda.ipynb` - The interactive Jupyter notebook implementation of the Exploratory Data Analysis.
4. `ml/reports/class_distribution.png` - Visual representation of the inherent class imbalance (3,875 Pneumonia scans vs 1,341 Normal scans in Training).
5. `ml/reports/sample_images.png` - Grid visualizations of various chest x-rays arrayed by classification.

## Implementation Details
We discovered that the initial Kaggle Validation (`val/`) set was critically small (comprising only 16 images total), while the image sizes varied radically (Heights `min: 672, max: 2663`, Widths `min: 912, max: 2916`).

This led to the subsequent Phase 2 architectural requirement: **Deep Learning Resizing Operations into [224x224]** batches dynamically via Torchvision.
