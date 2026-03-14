# Full Code Audit — Medical Image Classification Project
> Generated: 2026-03-13 | Focus: Path mismatches, data structure misalignment, logic bugs

---

## Confirmed Data Structure (from uploaded image)

```
ml/
└── data/
    ├── processed/
    ├── raw/
    │   ├── test/
    │   │   ├── NORMAL/
    │   │   └── PNEUMONIA/
    │   ├── train/
    │   │   ├── NORMAL/
    │   │   └── PNEUMONIA/
    │   └── val/
    │       ├── NORMAL/
    │       └── PNEUMONIA/
    ├── test.csv
    ├── train.csv
    └── val.csv
```

**Key facts:**
- CSVs live at `ml/data/{split}.csv` — NOT inside `ml/data/raw/`
- Raw images live at `ml/data/raw/{split}/{CLASS}/*.jpeg`

---

## 1. Path & Directory Mismatches (Critical)

### 1.1 `scripts/create_csv.py` — Wrong `OUTPUT_DIR`

```python
# CURRENT (WRONG)
DATA_ROOT  = 'ml/data/raw'
OUTPUT_DIR = 'ml/data'          # ✅ This is actually correct

for img_path in folder.glob('*.jpeg'):   # ❌ Only matches .jpeg
```

**Issue C1 — `.jpg` files excluded:** `folder.glob('*.jpeg')` only matches `.jpeg` extension. The Kaggle chest X-ray dataset contains both `.jpeg` and `.jpg` files. Files with `.jpg` extension are silently skipped, producing incomplete CSVs. Fix:

```python
# FIX: glob both extensions
for ext in ['*.jpeg', '*.jpg', '*.png']:
    for img_path in folder.glob(ext):
        rows.append(...)
```

**Issue C2 — No `if __name__ == '__main__'` guard:** The loop `for s in ['train', 'val', 'test']: create_csv(s)` runs at module import time. Any accidental import (e.g. from a test file) will trigger CSV regeneration and overwrite existing files.

```python
# FIX
if __name__ == '__main__':
    for s in ['train', 'val', 'test']:
        create_csv(s)
```

---

### 1.2 `ml/notebooks/01_eda.ipynb` — Wrong `DATA_ROOT`

```python
# CURRENT (WRONG)
DATA_ROOT = '../../data/raw'
```

The notebook lives at `ml/notebooks/01_eda.ipynb`. Two levels up (`../..`) from `ml/notebooks/` is the **project root**, not `ml/`. So `../../data/raw` resolves to `<project_root>/data/raw` — a path that **does not exist**.

```python
# FIX
DATA_ROOT = '../data/raw'   # One level up from ml/notebooks/ = ml/data/raw
```

This same wrong path affects **Cell 2, Cell 3, Cell 4, and Cell 5** in the notebook, and also `scripts/run_eda.py` (which correctly uses `ml/data/raw` since it runs from project root — but the notebook does not run from project root).

**Also affects `scripts/run_eda.py` report save path:**
```python
plt.savefig('ml/reports/class_distribution.png')   # Fine when run from project root
plt.savefig('../../reports/class_distribution.png') # Wrong in notebook context
```
The notebook saves to `../../reports/` which also resolves incorrectly. Fix for notebook:
```python
plt.savefig('../reports/class_distribution.png')
```

---

### 1.3 `ml/notebooks/01_eda.ipynb` — `.jpeg`-only glob (same as create_csv)

```python
# Cell 2 - WRONG
counts = {s: {c: len(list(pathlib.Path(f'{DATA_ROOT}/{s}/{c}').glob('*.jpeg')))
```
Will give wrong counts if any files are `.jpg`. Should glob both extensions.

---

### 1.4 `ml/notebooks/02_data_pipeline_test.ipynb` — CSV path assumption

```python
train_loader, val_loader, test_loader = get_dataloaders(batch_size=8, num_workers=0)
```
`get_dataloaders()` defaults to `data_dir='ml/data'`, which means it looks for CSVs at `ml/data/train.csv`. This only works if the notebook is run **from the project root**. Running it from inside `ml/notebooks/` will fail.

**Fix:** Either document that notebooks must be run from project root, or pass an explicit path:
```python
train_loader, val_loader, test_loader = get_dataloaders(
    data_dir='../../data',   # relative to ml/notebooks/
    batch_size=8, num_workers=0
)
```

---

### 1.5 `ml/notebooks/03_transfer_learning.ipynb` — Same `data_dir` issue

```python
train_loader, val_loader, _ = get_dataloaders(batch_size=32, num_workers=0)
# get_dataloaders default data_dir='ml/data' — only works from project root
```
Same fix as 1.4 above.

---

### 1.6 `ml/notebooks/04_gradcam_test.ipynb` — Multiple wrong paths

**Path 1 — model checkpoint:**
```python
# CURRENT (WRONG)
ckpt = torch.load('../../ml/saved_models/best_model_densenet121.pth', ...)
```
From `ml/notebooks/`, `../../ml/` resolves to `<project_root>/ml/` — **this actually works** since the notebook is inside `ml/notebooks/` and `../..` goes up to project root, then `ml/` goes back in. However it is fragile and non-idiomatic. Cleaner fix:
```python
ckpt = torch.load('../saved_models/best_model_densenet121.pth', ...)
```

**Path 2 — test image glob:**
```python
# CURRENT (WRONG)
img_path_candidates = glob.glob('../../ml/data/raw/test/PNEUMONIA/*.jpeg')
```
From `ml/notebooks/`, `../../ml/data/raw/` = `<project_root>/ml/data/raw/`. This path is redundant but technically correct. Simpler:
```python
img_path_candidates = glob.glob('../data/raw/test/PNEUMONIA/*.jpeg')
```

**Path 3 — report save:**
```python
plt.savefig('../../ml/reports/gradcam_sample.png')
```
Same issue — redundant but technically resolves correctly. Simpler:
```python
plt.savefig('../reports/gradcam_sample.png')
```

**Path 4 — `.jpeg` only glob (same bug as above):**
```python
glob.glob('.../*.jpeg')  # Misses .jpg files
```

---

### 1.7 `ml/training/dataloader.py` — Default `data_dir` path inconsistency

```python
def get_dataloaders(data_dir: str = 'ml/data', ...):
    train_ds = ChestXRayDataset(f'{data_dir}/train.csv', ...)
```
The default assumes execution from the **project root**. This is correct for `train.py` and `evaluate.py` (which are run from root), but **breaks for notebooks** run from `ml/notebooks/`. This is a design issue — the default is contextual. Consider using an absolute path derived from `__file__`:

```python
import pathlib
_DEFAULT_DATA_DIR = str(pathlib.Path(__file__).parent.parent / 'data')

def get_dataloaders(data_dir: str = _DEFAULT_DATA_DIR, ...):
```

---

### 1.8 `ml/training/evaluate.py` — Same `data_dir` path issue

```python
def evaluate(model_path: str, ..., data_dir: str = 'ml/data', ...):
```
Same root-relative assumption. Works when called via `scripts/run_training.sh` (which runs from root), breaks if called from elsewhere.

---

### 1.9 `backend/tests/test_predict.py` — Hardcoded path to raw data (gitignored)

```python
# CRITICAL — will always fail on CI / fresh clone
with open('ml/data/raw/test/NORMAL/IM-0001-0001.jpeg', 'rb') as f:
```
`ml/data/raw/` is in `.gitignore`. This file never exists in CI. The test must use a synthetic fixture:

```python
# FIX — create a synthetic 224x224 JPEG in a pytest fixture
import io
from PIL import Image

@pytest.fixture
def sample_jpeg_bytes():
    img = Image.new('RGB', (224, 224), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    return buf.getvalue()

def test_predict_valid_image(sample_jpeg_bytes):
    with TestClient(app) as client:
        response = client.post('/api/predict',
            files={'file': ('test.jpg', sample_jpeg_bytes, 'image/jpeg')})
        assert response.status_code == 200
```

---

## 2. ML Training Issues

### 2.1 `ml/training/dataset.py` — Missing file error handling

```python
def __getitem__(self, idx):
    row = self.df.iloc[idx]
    image = Image.open(row['filepath']).convert('RGB')  # ❌ No error handling
```
If any filepath in the CSV points to a missing file (corrupted download, partial sync), the DataLoader crashes mid-epoch with an unhelpful `FileNotFoundError`. Fix:

```python
def __getitem__(self, idx):
    row = self.df.iloc[idx]
    try:
        image = Image.open(row['filepath']).convert('RGB')
    except (FileNotFoundError, OSError) as e:
        raise RuntimeError(f"Failed to load image at index {idx}: {row['filepath']}") from e
    label = int(row['label'])
    if self.transform:
        image = self.transform(image)
    return image, label
```

### 2.2 `ml/training/augmentations.py` — `inference_transform` is same object reference as `val_transforms`

```python
inference_transform = val_transforms   # ❌ Alias, not a copy
```
If `val_transforms` is ever mutated (e.g. `.transforms.append(...)` in a notebook experiment), `inference_transform` silently changes too, breaking production inference. Fix:

```python
from copy import deepcopy
inference_transform = deepcopy(val_transforms)
# Or simply redeclare it as an independent Compose with identical transforms
```

### 2.3 `ml/training/dataloader.py` — `pin_memory=True` unconditional

```python
train_loader = DataLoader(..., pin_memory=True)   # ❌
```
`pin_memory=True` on CPU-only Windows can cause slow initialization or subtle CUDA errors. Should be:
```python
pin_memory = torch.cuda.is_available()
DataLoader(..., pin_memory=pin_memory)
```

### 2.4 `ml/training/train.py` — `verbose=True` deprecated

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=3, factor=0.5, verbose=True)  # ❌ Deprecated PyTorch >= 2.2
```
Fix: Remove `verbose=True` and log LR manually if needed.

### 2.5 `ml/training/train.py` — No random seed

Results are non-reproducible between runs. Add near the top of `main()`:
```python
torch.manual_seed(42)
np.random.seed(42)
```

### 2.6 `ml/training/evaluate.py` — `plt.close()` never called (memory leak)

```python
plt.figure(figsize=(8,6))   # Confusion matrix figure
# ... code ...
plt.figure(figsize=(8,6))   # ROC curve figure — previous figure never closed
```
Fix: Add `plt.close()` after each `plt.savefig()` call.

### 2.7 `ml/training/evaluate.py` — Dead import

```python
from sklearn.metrics import (..., f1_score)   # f1_score imported but never used
```

---

## 3. Backend Issues

### 3.1 `backend/config/settings.py` — `os.getenv()` bypasses Pydantic `.env` loading (HIGH)

```python
class Settings(BaseSettings):
    DATABASE_URL: str = os.getenv('DATABASE_URL', 'sqlite:///./chest_xray_db.sqlite')  # ❌
    MODEL_PATH: str   = os.getenv('MODEL_PATH', 'ml/saved_models/...')                 # ❌
```
`os.getenv()` executes at **class definition time**, before `BaseSettings` reads `.env`. Variables defined only in `.env` (not in the actual environment) are never picked up. Fix:

```python
class Settings(BaseSettings):
    DATABASE_URL: str = 'sqlite:///./chest_xray_db.sqlite'
    MODEL_PATH: str   = 'ml/saved_models/best_model_densenet121.pth'
    MODEL_NAME: str   = 'densenet121'
    MODEL_VERSION: str = '1.0'
    MAX_UPLOAD_MB: int = 16
    LOG_LEVEL: str    = 'INFO'
    SECRET_KEY: str   = 'change-me-in-production'

    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')
```
Also fixes the Pydantic v1 `class Config` deprecation warning.

### 3.2 `backend/api/routes/predict.py` — `requires_grad=True` contradicts `no_grad()` (HIGH)

```python
tensor = preprocessor.process(image)
tensor.requires_grad = True            # ❌ Set here...

label, confidence, class_idx = inference_service.predict(tensor)
# Inside predict():
with torch.no_grad():                  # ❌ ...then disabled here
    logits = self.model(tensor)
```
`torch.no_grad()` context manager overrides `requires_grad` on the tensor — gradients are NOT computed in `predict()`. Then `generate_cam()` does a **second full forward+backward pass** to get gradients. This means:

1. `predict()` runs the model once with `no_grad` (wasted since GradCAM needs gradients anyway)
2. `generate_cam()` runs the model again from scratch to actually compute gradients

Fix: Remove `tensor.requires_grad = True` from `predict.py`. The GradCAM service handles its own forward pass internally. The double-pass is a known inefficiency (Issue I2) — ideally merge into one pass, but at minimum remove the contradictory flag.

### 3.3 `backend/api/routes/predict.py` — Invalid MIME type

```python
ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/jpg']  # ❌ 'image/jpg' is not valid
```
`image/jpg` is not a registered MIME type. The correct type is `image/jpeg`. Some browsers do send `image/jpg` but it should not be in a standards-compliant allowlist. More robustly, validate using PIL format detection after decode:

```python
ALLOWED_TYPES = ['image/jpeg', 'image/png']
# After PIL.Image.open(), also check:
if image.format not in ('JPEG', 'PNG'):
    raise HTTPException(400, detail='Only JPEG/PNG files are accepted')
```

### 3.4 `backend/api/routes/predict.py` — Full file buffered before size check

```python
contents = await file.read()           # ❌ Entire file in memory first
if len(contents) > settings.MAX_UPLOAD_MB * 1024 * 1024:
    raise HTTPException(400, ...)      # Too late — already fully buffered
```
A 1GB upload is fully loaded into RAM before being rejected. Fix: check `Content-Length` header first, or stream with a size limit.

### 3.5 `backend/services/gradcam.py` — Not thread-safe under concurrent requests (HIGH)

```python
class GradCAMService:
    def __init__(self, model):
        self.gradients = None    # ❌ Instance-level state
        self.activations = None  # ❌ Instance-level state
```
If two requests hit the server simultaneously, request B's backward pass overwrites `self.gradients` while request A is still using them. Fix: use local variables passed through the call chain rather than instance state, or use a threading lock.

### 3.6 `backend/services/inference.py` — `sys.path.insert` fragile import (HIGH)

```python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from ml.training.model import MODEL_REGISTRY
```
Breaks if the application is launched from a directory other than project root. Fix: install the project as a package via `pip install -e .` with a `pyproject.toml`, eliminating all `sys.path` hacks.

### 3.7 `backend/api/routes/health.py` — Unused import

```python
import torch   # ❌ Never used in this file
```

### 3.8 `backend/models/prediction.py` — Hash column/hash algorithm mismatch

```python
image_hash = Column(String(64), ...)    # Sized for SHA-256 (64 hex chars)
# In predict.py:
img_hash = hashlib.md5(contents).hexdigest()   # MD5 = 32 hex chars
```
The column is sized for SHA-256 but MD5 is used. Either update to SHA-256 (better) or change the column size to `String(32)`.

---

## 4. Frontend Issues

### 4.1 `frontend/src/api/axiosClient.js` — Manual `Content-Type` breaks multipart boundary (HIGH)

```javascript
const response = await client.post('/api/predict', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },   // ❌ Remove this
});
```
When you manually set `Content-Type: multipart/form-data`, Axios does NOT append the required `boundary=----WebKitFormBoundary...` string. The server then cannot parse the multipart body and the upload fails. Fix: **remove the headers override entirely** — Axios sets the correct `Content-Type` with boundary automatically when the body is `FormData`.

```javascript
const response = await client.post('/api/predict', formData);
```

### 4.2 `frontend/src/pages/Home.jsx` — `URL.createObjectURL` never revoked (memory leak)

```javascript
setPreview(URL.createObjectURL(file));   // ❌ Object URL never revoked
```
Each uploaded image creates a blob URL that holds a reference to the file in browser memory. On repeated uploads these accumulate until page refresh. Fix:

```javascript
const handleUpload = async (file) => {
    if (preview) URL.revokeObjectURL(preview);   // Revoke previous
    const objectUrl = URL.createObjectURL(file);
    setPreview(objectUrl);
    // ...
};
// Also revoke on component unmount:
useEffect(() => {
    return () => { if (preview) URL.revokeObjectURL(preview); };
}, [preview]);
```

### 4.3 `frontend/src/components/ImageUploader.jsx` — Dropzone active while loading

```javascript
const { getRootProps, getInputProps } = useDropzone({
    onDrop: (acceptedFiles) => { if (acceptedFiles[0]) onUpload(acceptedFiles[0]); },
    // ❌ No disabled prop — user can drop another file while request is in-flight
});
```
Fix:
```javascript
const { getRootProps, getInputProps } = useDropzone({
    disabled: loading,
    onDrop: (acceptedFiles) => { if (!loading && acceptedFiles[0]) onUpload(acceptedFiles[0]); },
});
```

---

## 5. Docker Issues

### 5.1 `docker/Dockerfile.backend` — `curl` not installed, HEALTHCHECK always fails (HIGH)

```dockerfile
FROM python:3.10-slim
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*
# ...
HEALTHCHECK CMD curl -f http://localhost:8000/api/health || exit 1   # ❌ curl not installed
```
`python:3.10-slim` does not include `curl`. The health check will always fail, marking the container as unhealthy immediately. Fix: add `curl` to the apt-get install line.

```dockerfile
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 curl && \
    rm -rf /var/lib/apt/lists/*
```

### 5.2 `docker-compose.yml` — Frontend + NGINX disconnect (HIGH)

```yaml
frontend:
  build:
    dockerfile: docker/Dockerfile.frontend
  # No ports exposed — frontend container's nginx serves on :80 internally but nothing reaches it

nginx:
  image: nginx:alpine
  volumes:
    - ./docker/nginx.conf:/etc/nginx/conf.d/default.conf
  # ❌ This nginx has NO access to the React build files from the frontend container
```
`Dockerfile.frontend` builds React and copies `dist/` into its OWN nginx container. But the standalone `nginx` service is a separate container with a bare `nginx:alpine` image — it only gets `nginx.conf` mounted, not the React static files. The two are completely disconnected.

**Fix:** Either:
- **Option A (simpler):** Remove the standalone `nginx` service and expose the `frontend` container's port 80 directly.
- **Option B (correct multi-container):** Have the `frontend` Dockerfile NOT bundle nginx, output the static `dist/` to a shared Docker volume, and mount that volume into the `nginx` service.

### 5.3 `docker-compose.yml` — Hardcoded plaintext password (HIGH)

```yaml
POSTGRES_PASSWORD: password   # ❌ Hardcoded plaintext
DATABASE_URL: postgresql://appuser:password@postgres:5432/chest_xray_db   # ❌
```
Fix: use environment variable substitution.
```yaml
POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
```

### 5.4 `docker/Dockerfile.frontend` — `npm ci` requires `package-lock.json` which is gitignored

```dockerfile
RUN npm ci   # ❌ Requires package-lock.json
```
`package-lock.json` is listed in `frontend/.gitignore`. On a fresh clone, Docker build will fail with `npm ci can only install packages when your package.json and package-lock.json are in sync`. Fix: either use `npm install` or commit `package-lock.json` (recommended for reproducibility).

### 5.5 No `.dockerignore` file

The entire project directory — including `venv/`, `ml/data/raw/` (~2.4GB), `.git/` — is sent as Docker build context. This makes every `docker build` extremely slow. A `.dockerignore` is essential:

```
venv/
.git/
ml/data/raw/
ml/data/processed/
node_modules/
frontend/node_modules/
*.pth
*.log
__pycache__/
```

---

## 6. Summary — Issues by Priority

| Priority | ID | File | Severity | Description |
|---|---|---|---|---|
| 1 | DC2 | docker-compose.yml | **CRITICAL** | Frontend + NGINX disconnect — React app not served |
| 2 | DF1 | Dockerfile.backend | **CRITICAL** | `curl` missing — HEALTHCHECK always fails |
| 3 | DC1 | docker-compose.yml | **CRITICAL** | Hardcoded DB password in plaintext |
| 4 | CC1 | Root | **CRITICAL** | No `.dockerignore` — 2.4GB+ build context |
| 5 | AX1 | axiosClient.js | **CRITICAL** | Manual `Content-Type` breaks multipart boundary — uploads fail |
| 6 | NB1 | 01_eda.ipynb | **HIGH** | `DATA_ROOT = '../../data/raw'` — resolves to wrong path |
| 7 | NB2 | 01_eda.ipynb | **HIGH** | Report save paths `../../reports/` resolve incorrectly |
| 8 | NB3 | 02_data_pipeline_test.ipynb | **HIGH** | `data_dir` default works only from project root |
| 9 | NB4 | 03_transfer_learning.ipynb | **HIGH** | Same `data_dir` issue as above |
| 10 | NB5 | 04_gradcam_test.ipynb | **HIGH** | Redundant `../../ml/` prefix in paths (fragile) |
| 11 | P1 | predict.py | **HIGH** | `requires_grad=True` + `no_grad()` are contradictory |
| 12 | G1 | gradcam.py | **HIGH** | Instance-level gradient state — not thread-safe |
| 13 | S1 | settings.py | **HIGH** | `os.getenv()` bypasses Pydantic `.env` loading |
| 14 | I1 | inference.py | **HIGH** | `sys.path.insert` — fragile, breaks outside project root |
| 15 | TS1 | test_predict.py | **HIGH** | Hardcoded path to gitignored raw data — always fails on CI |
| 16 | C1/D1 | create_csv.py + dataset.py | **MEDIUM** | Only `*.jpeg` globbed — `.jpg` files excluded from dataset |
| 17 | C2 | create_csv.py | **MEDIUM** | No `__main__` guard — runs on import |
| 18 | HO1 | Home.jsx | **MEDIUM** | `createObjectURL` never revoked — memory leak |
| 19 | U1 | ImageUploader.jsx | **MEDIUM** | Dropzone active during loading — double-submit risk |
| 20 | DFF1 | Dockerfile.frontend | **MEDIUM** | `npm ci` requires gitignored `package-lock.json` |
| 21 | L1 | dataloader.py | **MEDIUM** | `pin_memory=True` unconditional — issues on CPU-only Windows |
| 22 | I2 | inference.py + gradcam.py | **MEDIUM** | Double forward pass — inefficient |
| 23 | ORM2 | prediction.py | **LOW** | `image_hash` column sized for SHA-256, but MD5 (32 chars) used |
| 24 | A1 | augmentations.py | **LOW** | `inference_transform` is same object as `val_transforms` |
| 25 | T2 | train.py | **LOW** | `verbose=True` deprecated in PyTorch >= 2.2 |
| 26 | T5 | train.py | **LOW** | No random seed — non-reproducible training runs |
| 27 | E1 | evaluate.py | **LOW** | `plt.close()` never called — memory leak |
| 28 | E2 | evaluate.py | **LOW** | `f1_score` imported but never used |
| 29 | H1 | health.py | **LOW** | `import torch` unused |
| 30 | P5 | predict.py | **LOW** | `import torch` unused |
