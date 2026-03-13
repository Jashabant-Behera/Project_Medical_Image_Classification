#!/bin/bash
# scripts/run_training.sh
# Reproducible training script with all hyperparameters documented
 
set -e   # Exit on any error
 
echo '=== Chest X-Ray Classifier Training ==='
echo 'Model: DenseNet121 | Epochs: 20 | LR: 1e-4 | Batch: 32'

# Wait dynamically for bash to locate either bin or Scripts (cross-compat)
if [ -d "venv/bin" ]; then
    source venv/bin/activate
else
    source venv/Scripts/activate
fi
 
python ml/training/train.py \
  --model densenet121 \
  --epochs 20 \
  --lr 0.0001 \
  --batch_size 32 \
  --patience 5 \
  --data_dir ml/data \
  --save_dir ml/saved_models
 
echo 'Training complete. Running evaluation...'
python ml/training/evaluate.py
echo 'Done. Check ml/reports/ for metrics and plots.'
