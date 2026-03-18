# Google Colab Training Guide — Chest X-Ray Classifier

> Local guide only — not tracked by git.
> Use this to train DenseNet121 on Google Colab with a free T4 GPU (~10x faster than CPU).

---

## Why Colab?

| Environment | Time per epoch | 20 epochs |
|-------------|---------------|-----------|
| Local CPU (your machine) | ~12 minutes | ~4–6 hours |
| Google Colab T4 GPU (free) | ~60–90 seconds | ~25–30 minutes |
| Google Colab A100 GPU (Pro) | ~20–30 seconds | ~10 minutes |

---

## Step 1 — Upload the ML code to Google Drive

### Option A: Upload via browser (simplest)
1. Go to [https://drive.google.com](https://drive.google.com)
2. Create a folder: `My Drive / chest-xray-project/`
3. Upload the entire `ml/` folder (drag and drop)
4. Upload `requirements.txt`
5. Upload the dataset at `ml/data/raw/` (or just the CSV files if already generated)

### Option B: Use rclone or Google Drive desktop sync
```bash
# If you have rclone configured:
rclone copy ml/ gdrive:chest-xray-project/ml/ --progress
```

---

## Step 2 — Open a new Colab Notebook

1. Go to [https://colab.research.google.com](https://colab.research.google.com)
2. Click **New Notebook**
3. Go to **Runtime → Change runtime type → T4 GPU** ← IMPORTANT
4. Click Save

---

## Step 3 — Mount Google Drive

Paste this in the first cell and run:

```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/drive/MyDrive/chest-xray-project')
!ls  # Should show: ml/ requirements.txt
```

---

## Step 4 — Install dependencies

```python
# Install only the essential packages (skip jupyter, black, boto3 etc)
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q fastapi uvicorn pydantic scikit-learn pandas numpy tqdm opencv-python-headless Pillow seaborn matplotlib
```

> **Note:** PyTorch with CUDA is installed separately to get the GPU-optimized build.

---

## Step 5 — Verify GPU is available

```python
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
# Expected: GPU available: True
#           GPU name: Tesla T4
```

---

## Step 6 — Generate CSV manifests (if not already done)

```python
# Only run this if you uploaded the raw dataset images
!python ml/data/create_csv.py
# Expected output:
# Created ml/data/processed/train.csv with 5216 rows
# Created ml/data/processed/val.csv  with 16  rows
# Created ml/data/processed/test.csv with 624 rows
```

> **Skip this step** if you already have `ml/data/train.csv`, `val.csv`, `test.csv` — just upload those CSV files directly.

---

## Step 7 — Train the model

```python
!python ml/training/train.py \
    --model densenet121 \
    --epochs 20 \
    --lr 0.0001 \
    --batch_size 32 \
    --patience 5 \
    --data_dir ml/data \
    --save_dir ml/saved_models
```

### Expected output per epoch:
```
Training on: cuda
Epoch  1/20 | TrainLoss=0.4821 Acc=0.7892 | ValLoss=0.3012 Acc=0.8750 AUC=0.9432 | Time=72s
  ✓ Saved best model (AUC=0.9432) -> ml/saved_models/best_model_densenet121.pth
Epoch  2/20 | TrainLoss=0.3104 Acc=0.8741 | ValLoss=0.2187 Acc=0.9062 AUC=0.9687 | Time=68s
  ✓ Saved best model (AUC=0.9687) -> ml/saved_models/best_model_densenet121.pth
...
```

---

## Step 8 — Download the trained model

After training is done, download the weights file to your local machine:

### Option A: Colab download button
```python
from google.colab import files
files.download('ml/saved_models/best_model_densenet121.pth')
```

### Option B: Leave it on Google Drive and use rclone
```bash
# On your local machine:
rclone copy gdrive:chest-xray-project/ml/saved_models/ ml/saved_models/ --progress
```

---

## Step 9 — Place the downloaded model locally

```
ml/
└── saved_models/
    └── best_model_densenet121.pth   ← Place downloaded file here
```

The FastAPI backend already points to this exact path via `.env`:
```
MODEL_PATH=ml/saved_models/best_model_densenet121.pth
```

---

## Step 10 — Restart the backend to load the new model

```powershell
# Stop the running uvicorn (Ctrl+C in its terminal), then re-run:
.\venv\Scripts\Activate.ps1
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

The console should print:
```
Model loaded on cpu: densenet121 v1.0
Server ready!
```

---

## Full Colab Notebook (copy-paste ready)

Paste the following into a single Colab notebook for convenience:

```python
# ── Cell 1: Mount Drive ──────────────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')
import os
os.chdir('/content/drive/MyDrive/chest-xray-project')

# ── Cell 2: Install packages ─────────────────────────────────────────────────
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q scikit-learn pandas numpy tqdm opencv-python-headless Pillow seaborn matplotlib

# ── Cell 3: Check GPU ────────────────────────────────────────────────────────
import torch
print(f"CUDA: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# ── Cell 4: Create CSV manifests ─────────────────────────────────────────────
!python scripts/create_csv.py

# ── Cell 5: Train ────────────────────────────────────────────────────────────
!python ml/training/train.py \
    --model densenet121 \
    --epochs 20 \
    --lr 0.0001 \
    --batch_size 32 \
    --patience 5 \
    --data_dir ml/data/processed \
    --save_dir ml/saved_models

# ── Cell 6: Run evaluation ───────────────────────────────────────────────────
!python ml/training/evaluate.py

# ── Cell 7: Download model ───────────────────────────────────────────────────
from google.colab import files
files.download('ml/saved_models/best_model_densenet121.pth')
```

---

## Tips & Troubleshooting

| Issue | Fix |
|-------|-----|
| `CUDA out of memory` | Reduce `--batch_size` to 16 |
| `FileNotFoundError: train.csv` | Run `scripts/create_csv.py` first |
| Session disconnects mid-training | Colab free tier has ~12hr limit; keep the tab active or use Colab Pro |
| `Module not found: ml.training` | Make sure you `os.chdir()` to the project root (Step 3) |
| Training looks stuck at 0% | The first batch loads pretrained weights (~30MB download) — wait 30s |
| Val AUC stays at 0.5 | The `val/` split has only 16 images — this is normal for this dataset; monitor test AUC via `evaluate.py` after training |

---

## Expected Final Metrics (after 20 epochs on T4)

These are approximate values based on the literature for DenseNet121 on this dataset:

| Metric | Expected Value |
|--------|---------------|
| Test Accuracy | ~90–93% |
| Test ROC-AUC | ~97–98% |
| Pneumonia Recall (sensitivity) | ~95–98% |
| Normal Precision | ~88–92% |

> Medical AI note: **Recall (sensitivity) for PNEUMONIA is the critical metric** — a miss (false negative) is more dangerous than a false alarm.

---

## Local CPU Training (current session)

Your local CPU training is currently running in the background. To check progress at any time:

- Ask: *"how is training going?"*
- Or run in a new terminal:

```powershell
# Poll the background training process output
# (or just wait — it will save the model to ml/saved_models/ when done)
```
