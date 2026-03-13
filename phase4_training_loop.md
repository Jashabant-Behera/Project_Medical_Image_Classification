# Phase 4: Full Training Loop, Evaluation & Metrics

## Objective
Establish autonomous scripting pipelines orchestrating forward passes, gradient descents, automated evaluations across validation splits seamlessly, maintaining metric states continuously tracking PyTorch `CrossEntropyLoss` metrics through dynamically instantiated `ReduceLROnPlateau` scheduling loops spanning up to `20 Epochs` naturally. Include testing suite generators using classification metrics from SciKit-Learn dynamically storing states to Checkpoint outputs.

## Files Created
1. `ml/training/train.py` - Core execution algorithm orchestrating dataset iteration wrappers (tqdm), learning rate scaling (1e-4 down), dynamic batch accumulations, validation looping, patience mechanisms (`patience=5`), and best `.pth` saving outputs.
2. `ml/training/evaluate.py` - Standard testing load routines evaluating local test metadata sequentially into Python DataFrames scoring predictions blindly into ROC-AUC plotting matrices (`roc_curve()`) and native Precision/Recall/F1 mappings dynamically mapping into `seaborn.heatmap()` confusion visualizers dynamically logging outputs (`ml/reports/metrics.json`).
3. `ml/saved_models/best_model_densenet121.pth`
4. `scripts/run_training.sh` - Standardized CI/CD execution shell for cloud deployments.

## Metrics
- **Optimizer**: `Adam`
- **Criterion**: `CrossEntropyLoss`
- **Scheduler**: `ReduceLROnPlateau(factor=0.5, patience=3)`
- **Core Monitoring**: Validation Area Under Receiver Operating Characteristic Curve (`ROC-AUC`).
