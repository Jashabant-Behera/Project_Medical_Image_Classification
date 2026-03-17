# Medical Image Classification System
### Chest X-Ray Pneumonia Detection — Complete Developer Documentation
> Version 1.0 · March 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project File Structure](#2-project-file-structure)
3. [File-by-File Explanation](#3-file-by-file-explanation)
4. [Function-by-Function Explanation](#4-function-by-function-explanation)
5. [Class Explanation](#5-class-explanation)
6. [Notebook Cell-by-Cell Explanation](#6-notebook-cell-by-cell-explanation)
7. [Data Flow Explanation](#7-data-flow-explanation)
8. [Dependencies](#8-dependencies)
9. [System Execution Flow](#9-system-execution-flow)
10. [Additional Notes](#10-additional-notes)
11. [Quick Reference Card](#11-quick-reference-card)

---

## 1. Project Overview

### What the Project Does

This project is a full-stack, end-to-end AI platform for detecting Pneumonia from chest X-ray images. A clinician or researcher uploads a JPEG/PNG chest X-ray image through a browser-based React interface. The image is sent to a FastAPI backend, which runs it through a fine-tuned DenseNet121 deep-learning model. The backend returns:

- A binary classification: **NORMAL** or **PNEUMONIA**
- A **confidence score** (softmax probability)
- An **inference time** in milliseconds
- A **Grad-CAM heatmap overlay** visually highlighting which regions of the X-ray most influenced the decision

### Main Objective & Problem Solved

Pneumonia is a leading cause of mortality worldwide. Radiologists manually reading X-rays is time-consuming and subject to human fatigue. This system provides:

- Automated pneumonia screening from raw chest X-ray images
- Visual explainability (Grad-CAM) so clinicians understand why the model flagged an image
- A persistent audit trail of every prediction stored in a database
- A portable, Docker-containerized deployment that runs identically on a developer laptop or a cloud VM

### Overall Architecture

```
Browser (React + Vite + Tailwind CSS)
    ↓  POST /api/predict  (multipart/form-data)
FastAPI Backend  (Uvicorn ASGI server)
    ├── ImagePreprocessor  → 224×224 normalized tensor
    ├── InferenceService (Singleton) → DenseNet121 → label + confidence
    ├── GradCAMService  → heatmap PNG (base64)
    └── SQLAlchemy ORM → SQLite / PostgreSQL  (prediction_logs table)
    ↑  JSON response  { prediction, confidence, gradcam_image, ... }
Browser renders ResultCard + GradCamViewer

Docker Compose orchestrates: postgres | backend | frontend | nginx
```

### Key Technologies

| Layer | Technology / Library | Purpose |
|---|---|---|
| ML Model | PyTorch + Torchvision (DenseNet121) | Transfer learning for binary image classification |
| Explainability | Grad-CAM (manual implementation) | Saliency heatmaps highlighting decision regions |
| Backend API | FastAPI + Uvicorn | Async REST API serving predictions |
| ORM / DB | SQLAlchemy + Alembic + SQLite/PostgreSQL | Schema management and prediction logging |
| Image Processing | Pillow + OpenCV-headless | Image decoding, resizing, colormap overlays |
| Frontend | React 18 + Vite + Tailwind CSS | Browser drag-and-drop upload UI |
| HTTP Client | Axios | Frontend-to-backend API calls |
| Containerisation | Docker + Docker Compose + Nginx | Reproducible multi-service deployment |
| Data Science | NumPy, Pandas, scikit-learn, Matplotlib, Seaborn | EDA, metrics, visualizations |
| Config | Pydantic Settings + python-dotenv | .env-driven configuration management |
| Testing | pytest + httpx + FastAPI TestClient | Backend integration tests |

---

## 2. Project File Structure

```
Project_Medical_Image_Classification/
│
├── backend/                              ← FastAPI application
│   ├── __init__.py
│   ├── main.py                           ← App entry point, lifespan, CORS, routers
│   ├── alembic.ini                       ← Alembic migration configuration
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── health.py                 ← GET  /api/health
│   │       ├── history.py                ← GET  /api/predictions
│   │       └── predict.py                ← POST /api/predict
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py                   ← Pydantic BaseSettings (.env loading)
│   │
│   ├── db/
│   │   ├── __init__.py
│   │   ├── session.py                    ← SQLAlchemy engine + session + Base
│   │   └── migrations/
│   │       ├── env.py                    ← Alembic migration environment
│   │       ├── script.py.mako            ← Migration file template
│   │       └── versions/
│   │           └── 1b8ada2f52a3_create_prediction_logs_table.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── prediction.py                 ← ORM model → prediction_logs table
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── gradcam.py                    ← Grad-CAM heatmap generation
│   │   ├── inference.py                  ← Model singleton + prediction
│   │   └── preprocessor.py               ← PIL Image → normalized tensor
│   │
│   ├── tests/
│   │   ├── __init__.py
│   │   └── test_predict.py               ← Integration tests for all endpoints
│   │
│   └── utils/
│       ├── __init__.py
│       └── response_models.py            ← Pydantic response schemas
│
├── ml/                                   ← Machine learning pipeline
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── create_csv.py                 ← Scans raw/ and generates CSV manifests
│   │   ├── train.csv                     ← Training set manifest (filepath, label, class)
│   │   ├── val.csv                       ← Validation set manifest
│   │   ├── test.csv                      ← Test set manifest
│   │   ├── raw/                          ← Kaggle dataset images (gitignored)
│   │   │   ├── train/
│   │   │   │   ├── NORMAL/
│   │   │   │   └── PNEUMONIA/
│   │   │   ├── val/
│   │   │   │   ├── NORMAL/
│   │   │   │   └── PNEUMONIA/
│   │   │   └── test/
│   │   │       ├── NORMAL/
│   │   │       └── PNEUMONIA/
│   │   └── processed/                    ← Reserved for future preprocessed data
│   │
│   ├── notebooks/
│   │   ├── 01_eda.ipynb                  ← Dataset EDA and class distribution
│   │   ├── 02_data_pipeline_test.ipynb   ← DataLoader smoke test
│   │   ├── 03_transfer_learning.ipynb    ← Model build and mini-epoch verification
│   │   └── 04_gradcam_test.ipynb         ← Grad-CAM generation and visualization
│   │
│   ├── reports/                          ← Auto-generated plots and metrics (gitignored)
│   │   ├── class_distribution.png
│   │   ├── sample_images.png
│   │   ├── confusion_matrix.png
│   │   ├── roc_curve.png
│   │   ├── gradcam_sample.png
│   │   └── metrics.json
│   │
│   ├── saved_models/                     ← Trained weights (gitignored)
│   │   ├── best_model_densenet121.pth
│   │   └── training_history.json
│   │
│   └── training/
│       ├── __init__.py
│       ├── augmentations.py              ← train / val / inference transforms
│       ├── dataloader.py                 ← DataLoader + WeightedRandomSampler
│       ├── dataset.py                    ← ChestXRayDataset (CSV-based)
│       ├── evaluate.py                   ← Post-training metrics + plots
│       ├── model.py                      ← DenseNet121 / ResNet50 / EfficientNet + registry
│       └── train.py                      ← Training loop CLI entry point
│
├── frontend/                             ← React + Vite SPA
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   ├── .gitignore
│   │
│   ├── public/
│   │   └── favicon.svg                   ← Custom project favicon
│   │
│   └── src/
│       ├── main.jsx                      ← React app bootstrap
│       ├── App.jsx                       ← BrowserRouter + nav shell
│       ├── index.css                     ← Tailwind base import
│       │
│       ├── api/
│       │   └── axiosClient.js            ← predictImage() + fetchHistory()
│       │
│       ├── components/
│       │   ├── GradCamViewer.jsx         ← Side-by-side original + heatmap
│       │   ├── ImageUploader.jsx         ← Drag-and-drop upload zone
│       │   └── ResultCard.jsx            ← Prediction label + confidence bar
│       │
│       └── pages/
│           └── Home.jsx                  ← Main page + state management
│
├── docker/
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── nginx.conf
│
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── COLAB_TRAINING_GUIDE.md               ← Google Colab GPU training guide
└── dataset.md                            ← Dataset structure and class imbalance notes
```

### Folder Purpose Summary

| Folder | Purpose |
|---|---|
| `backend/` | FastAPI application: routes, services, models, database, config |
| `ml/` | All machine learning code: dataset, training, evaluation, notebooks, saved models |
| `ml/data/` | CSV manifests pointing to raw image files; `raw/` holds Kaggle images (gitignored) |
| `ml/training/` | PyTorch training pipeline: dataset, augmentations, dataloader, model, train, evaluate |
| `ml/notebooks/` | Jupyter notebooks for interactive exploration and verification |
| `ml/saved_models/` | Trained `.pth` checkpoints and training history JSON |
| `ml/reports/` | Generated plots: class distribution, sample images, confusion matrix, ROC curve |
| `frontend/` | React+Vite SPA: drag-and-drop upload UI, result display, Grad-CAM viewer |
| `docker/` | Dockerfiles and Nginx config for containerised multi-service deployment |

---

## 3. File-by-File Explanation

### ML Layer

---

#### `ml/data/create_csv.py`

**Purpose:** Scans `ml/data/raw/{train|val|test}/{NORMAL|PNEUMONIA}/` and generates three CSV files — `train.csv`, `val.csv`, and `test.csv` — with columns: `filepath`, `label` (0/1), `class` ('NORMAL'/'PNEUMONIA').

> **Known Bug:** Only globs `*.jpeg` — misses `.jpg` files. The Kaggle dataset contains both.  
> **Fix:** Iterate `['*.jpeg', '*.jpg', '*.png']` instead.

**Interacts with:** `ml/data/raw/` (reads), `ml/data/*.csv` (writes), `ml/training/dataset.py` (consumes the CSVs).

---

#### `ml/training/dataset.py`

**Purpose:** Defines `ChestXRayDataset`, a PyTorch Dataset that reads a CSV manifest and returns `(image_tensor, label)` pairs for use in DataLoaders.

| Component | Type | Description |
|---|---|---|
| `ChestXRayDataset` | Class | Custom `torch.utils.data.Dataset` subclass; reads filepaths and labels from a CSV |
| `__init__(csv_path, transform)` | Method | Loads CSV into DataFrame; stores transform pipeline |
| `__len__()` | Method | Returns total number of samples in this split |
| `__getitem__(idx)` | Method | Opens image from filepath at row `idx`, converts to RGB, applies transform, returns `(tensor, int_label)` |
| `get_class_weights()` | Method | Computes per-sample weights inversely proportional to class frequency for `WeightedRandomSampler` |
| `self.df` | Attribute | pandas DataFrame containing `filepath`, `label`, `class` columns |
| `self.class_names` | Attribute | `['NORMAL', 'PNEUMONIA']` reference list |

**Interacts with:** `dataloader.py` (consumed by `get_dataloaders`), `ml/data/*.csv` (reads filepaths), Pillow (`Image.open`).

---

#### `ml/training/augmentations.py`

**Purpose:** Defines three `torchvision.transforms.Compose` pipelines: `train_transforms` (with augmentation), `val_transforms` (clean), and `inference_transform` (alias for `val_transforms` used by the backend).

| Variable | Description |
|---|---|
| `IMG_SIZE = 224` | Target spatial size; standard for ImageNet pretrained models |
| `IMAGENET_MEAN / STD` | Normalization constants from ImageNet; required because the model uses pretrained weights |
| `train_transforms` | `Resize → RandomHorizontalFlip → RandomRotation → ColorJitter → RandomAffine → ToTensor → Normalize` |
| `val_transforms` | `Resize → ToTensor → Normalize` (no augmentation) |
| `inference_transform` | Same object as `val_transforms`; imported by backend preprocessor and Grad-CAM notebook |

**Interacts with:** `dataset.py` (transform param), `preprocessor.py` (inference_transform), notebooks 02–04.

---

#### `ml/training/dataloader.py`

**Purpose:** Creates three `DataLoader` objects (train, val, test) with a proper sampling strategy to combat class imbalance.

| Component | Description |
|---|---|
| `get_dataloaders()` | Factory function; instantiates all three DataLoaders. Accepts `data_dir`, `batch_size`, `num_workers`, `use_sampler` |
| `WeightedRandomSampler` | Used for the training loader; draws samples proportional to inverse class frequency so NORMAL is not underrepresented |
| `pin_memory=True` | Speeds up GPU transfer — causes warnings on CPU-only Windows machines (should be conditional on `torch.cuda.is_available()`) |

**Interacts with:** `dataset.py` (ChestXRayDataset), `augmentations.py` (transforms), `train.py` and `evaluate.py` (consumers).

---

#### `ml/training/model.py`

**Purpose:** Defines three neural network builders and a `MODEL_REGISTRY` dict that maps string names to builder functions.

| Component | Description |
|---|---|
| `NUM_CLASSES = 2` | Module-level constant for NORMAL / PNEUMONIA |
| `build_densenet121(pretrained, freeze_layers)` | Loads DenseNet121 with ImageNet weights. Optionally freezes all layers except `denseblock4` + `norm5`. Replaces classifier with `Linear(1024,256) → ReLU → Dropout(0.4) → Linear(256,2)` |
| `build_resnet50(pretrained)` | Loads ResNet50, freezes all layers, replaces `fc` with a 256-unit head |
| `build_efficientnet_b0(pretrained)` | Loads EfficientNet-B0, freezes all, replaces classifier head |
| `MODEL_REGISTRY` | Dict mapping `'densenet121'`, `'resnet50'`, `'efficientnet_b0'` to their builder functions |

**Interacts with:** `train.py` (instantiation), `evaluate.py` (evaluation), `inference.py` (production loading), notebooks 03 and 04.

---

#### `ml/training/train.py`

**Purpose:** Full training loop script with CLI argument parsing. Runs multiple epochs, evaluates on the val set, saves the best checkpoint by ROC-AUC, and implements early stopping and LR scheduling.

| Function | Description |
|---|---|
| `train_epoch(model, loader, criterion, optimizer, device)` | Single training epoch: iterates batches, forward pass, `loss.backward()`, `optimizer.step()`. Returns avg loss and accuracy |
| `eval_epoch(model, loader, criterion, device)` | Single evaluation epoch with `torch.no_grad()`; computes loss, accuracy, and ROC-AUC. Handles edge case where val split has only one class |
| `main(args)` | Orchestrates training: device detection, DataLoader creation, model + optimizer + scheduler instantiation, epoch loop, checkpoint saving, early stopping, history JSON export |

**Key config:** Adam optimizer, CrossEntropyLoss, `ReduceLROnPlateau` scheduler (patience=3, factor=0.5), early stopping (patience=5). Saves checkpoint to `ml/saved_models/best_model_{model}.pth` as a dict containing `epoch`, `model_state_dict`, `val_auc`, `val_acc`, `model_name`.

---

#### `ml/training/evaluate.py`

**Purpose:** Post-training evaluation script that runs the saved model against the held-out test set and generates comprehensive metrics and plots.

| Function | Output |
|---|---|
| `evaluate(model_path, model_name, data_dir, report_dir)` | Loads model from `.pth`, runs full test set inference, computes classification report, ROC-AUC, confusion matrix heatmap, ROC curve. Saves `metrics.json`, `confusion_matrix.png`, `roc_curve.png` to `ml/reports/` |

---

### Backend Layer

---

#### `backend/config/settings.py`

**Purpose:** Centralizes all configuration using Pydantic `BaseSettings`. Reads from `.env` file and environment variables.

| Field | Default Value | Description |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./chest_xray_db.sqlite` | SQLAlchemy connection string; defaults to SQLite for local dev |
| `MODEL_PATH` | `ml/saved_models/best_model_densenet121.pth` | Path to trained model weights |
| `MODEL_NAME` | `densenet121` | Key for `MODEL_REGISTRY` lookup |
| `MODEL_VERSION` | `1.0` | Version string included in API responses |
| `MAX_UPLOAD_MB` | `16` | Maximum upload file size in megabytes |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `SECRET_KEY` | `change-me-in-production` | Placeholder for future JWT/session use |

> **Known Bug (S1):** The original code uses `os.getenv()` as default values, which bypasses Pydantic's `.env` loading. The correct pattern is to set bare Python defaults (e.g., `DATABASE_URL: str = 'sqlite://...'`) and let `BaseSettings` handle `.env` reading.

---

#### `backend/db/session.py`

**Purpose:** Creates the SQLAlchemy engine and session factory. Provides the `Base` declarative class and the `get_db` dependency.

| Component | Description |
|---|---|
| `engine` | SQLAlchemy Engine created from `DATABASE_URL`; SQLite gets `check_same_thread=False` for FastAPI threading |
| `SessionLocal` | `sessionmaker` bound to engine; `autocommit=False`, `autoflush=False` |
| `Base` | `DeclarativeBase` subclass; all ORM models inherit from this |
| `get_db()` | FastAPI dependency (generator); yields a session and closes it in `finally` block |

---

#### `backend/models/prediction.py`

**Purpose:** Defines the SQLAlchemy ORM model for the `prediction_logs` table that records every inference request.

| Column | Type | Description |
|---|---|---|
| `id` | Integer PK autoincrement | Unique prediction record ID |
| `filename` | String(255) | Original uploaded filename |
| `prediction` | String(20) | `'NORMAL'` or `'PNEUMONIA'` |
| `confidence` | Float | Model softmax probability for the predicted class |
| `inference_ms` | Integer | Time taken for inference in milliseconds |
| `model_version` | String(50) | Version string from settings |
| `image_hash` | String(64) | MD5 hex digest of image bytes for deduplication |
| `created_at` | DateTime | Timestamp set at insert time (application-side) |

---

#### `backend/services/preprocessor.py`

**Purpose:** Wraps the `val_transforms` pipeline into a service class used by the prediction route to convert a PIL Image into a normalized tensor.

| Component | Description |
|---|---|
| `ImagePreprocessor` | Simple class with a `process(image)` method |
| `self.transform` | `Compose: Resize(224,224) → ToTensor → Normalize(ImageNet stats)` |
| `process(image)` | Converts PIL Image to RGB, applies transform, adds batch dimension → shape `[1, 3, 224, 224]` |

---

#### `backend/services/inference.py`

**Purpose:** Implements a thread-safe Singleton pattern for the ML model. The model is loaded once at server startup and reused for every request, avoiding expensive repeated disk reads.

| Component | Description |
|---|---|
| `InferenceService` | Singleton class using `__new__` to ensure only one instance exists |
| `load_model()` | Detects CUDA availability, instantiates model from `MODEL_REGISTRY`, loads weights from `.pth` checkpoint, sets `model.eval()` |
| `predict(tensor)` | Moves tensor to device, runs forward pass inside `torch.no_grad()`, returns `(class_name, confidence_float, class_index_int)` |
| `CLASS_NAMES` | `['NORMAL', 'PNEUMONIA']` — index 0 = normal, index 1 = pneumonia |
| `inference_service` | Module-level singleton instance; imported by `predict.py` route |

> **Known Limitation:** Uses `sys.path.insert()` to import `ml.training.model` — fragile if server is not launched from the project root. Proper fix is `pip install -e .` with a `pyproject.toml`.

---

#### `backend/services/gradcam.py`

**Purpose:** Generates Gradient-weighted Class Activation Mapping (Grad-CAM) overlays. Registers PyTorch forward and backward hooks on the DenseNet121 final normalization layer (`features.norm5`) to capture gradients and activations needed to compute the saliency map.

| Method | Description |
|---|---|
| `__init__(model, target_layer_name)` | Resolves the target layer by dotted name, registers forward hook that saves activations AND registers a tensor backward hook to save gradients |
| `_get_layer(layer_name)` | Traverses model attribute hierarchy by splitting on dots |
| `_save_gradient(grad)` | Backward hook callback; stores grad in `self.gradients` |
| `_save_activation_and_gradient(module, input, output)` | Forward hook; stores activations and conditionally registers backward hook on output tensor (bypasses DenseNet inplace ReLU issue) |
| `generate_cam(input_tensor, class_idx)` | Runs forward + backward pass; pools gradients across spatial dims; multiplies with activations; applies ReLU; normalizes to `[0,1]` |
| `generate_overlay(input_tensor, original_image, class_idx)` | Calls `generate_cam`, resizes to original image size, applies `COLORMAP_JET`, blends 50/50 with original via `cv2.addWeighted` |
| `overlay_to_base64(overlay_image)` | Encodes PIL image as PNG → base64 string with `data:image/png;base64,` prefix |

> **Thread-Safety Concern (G1):** `self.gradients` and `self.activations` are instance state — concurrent requests can overwrite each other's values. In production, use per-request local variables or a `threading.Lock()`.

---

#### `backend/api/routes/predict.py`

**Purpose:** The main prediction endpoint. Validates the uploaded file, preprocesses it, runs inference, generates Grad-CAM, logs to database, and returns a structured JSON response.

```
POST /api/predict
  Content-Type: multipart/form-data
  Body: file (JPEG or PNG, max 16 MB)
  Response: { prediction, confidence, gradcam_image, inference_ms, model_version, timestamp }
```

| Step | Action |
|---|---|
| 1. Validation | Check `content_type` is in `[image/jpeg, image/png]`; read body; check size against `MAX_UPLOAD_MB` |
| 2. Decode | `PIL.Image.open()` from `BytesIO`; convert to RGB |
| 3. Preprocess | `ImagePreprocessor.process()` → `[1, 3, 224, 224]` tensor |
| 4. Inference | `inference_service.predict()` → `(label, confidence, class_idx)` |
| 5. Grad-CAM | `GradCAMService` (lazy init) → `generate_overlay()` → `overlay_to_base64()` |
| 6. DB Log | Create `PredictionLog` with MD5 hash; `db.commit()` |
| 7. Response | Return `PredictionResponse` Pydantic model |

---

#### `backend/api/routes/health.py`

**Purpose:** Simple health check endpoint. Reports whether the model is loaded and which device it is running on. Used by Docker `HEALTHCHECK` and frontend polling.

```
GET /api/health
  Response: { status, model_loaded, model_name, device }
```

---

#### `backend/api/routes/history.py`

**Purpose:** Returns paginated prediction history from the database with total count.

```
GET /api/predictions?limit=50&skip=0
  Response: { total, items: [...PredictionHistoryItem] }
```

---

#### `backend/utils/response_models.py`

**Purpose:** Defines all Pydantic response schemas that FastAPI uses to serialize/validate API responses.

| Model | Fields |
|---|---|
| `PredictionResponse` | `prediction` (str), `confidence` (float), `gradcam_image` (str base64), `inference_ms` (int), `model_version` (str), `timestamp` (datetime) |
| `HealthResponse` | `status` (str), `model_loaded` (bool), `model_name` (str), `device` (str) |
| `PredictionHistoryItem` | `id`, `filename`, `prediction`, `confidence`, `inference_ms`, `created_at` — with `from_attributes=True` for ORM interop |
| `PredictionHistoryResponse` | `total` (int), `items` (List[PredictionHistoryItem]) |

---

#### `backend/main.py`

**Purpose:** Application entry point. Wires together all routes, middleware, and startup/shutdown logic using FastAPI's lifespan context manager.

| Component | Description |
|---|---|
| `lifespan(app)` | Async context manager: on startup creates DB tables (`Base.metadata.create_all`) and loads the ML model (`inference_service.load_model()`); yield; shutdown |
| `CORSMiddleware` | Allows all origins (`allow_origins=["*"]`) — must be restricted in production |
| Route inclusion | Includes `predict`, `health`, and `history` routers |
| `GET /` | Root endpoint returning a welcome JSON with docs link |

---

#### `backend/db/migrations/env.py`

**Purpose:** Alembic migration environment. Connects the Alembic migration runner to the project's SQLAlchemy metadata so that `alembic revision --autogenerate` and `alembic upgrade head` work correctly.

**Key actions:** injects project root into `sys.path`, imports `Base` and `settings`, sets `config.set_main_option('sqlalchemy.url', settings.DATABASE_URL)`, and defines `run_migrations_offline` / `run_migrations_online`.

---

### Frontend Layer

---

#### `frontend/src/api/axiosClient.js`

**Purpose:** Centralizes all HTTP calls to the backend API.

| Export | Description |
|---|---|
| `predictImage(file)` | Creates `FormData`, POSTs to `/api/predict`. **Note:** Do not set `Content-Type` manually — Axios auto-sets it with the correct multipart boundary |
| `fetchHistory(limit)` | GETs `/api/predictions?limit={limit}`; returns history array |
| `BASE_URL` | Reads from `import.meta.env.VITE_API_URL`; falls back to `http://localhost:8000` |

---

#### `frontend/src/components/ImageUploader.jsx`

**Purpose:** Drag-and-drop upload zone using `react-dropzone`. Accepts JPEG/PNG up to 16 MB. Shows loading state while the request is in flight.

| Prop | Type | Description |
|---|---|---|
| `onUpload` | Function | Callback called with the accepted `File` object |
| `loading` | Boolean | When true, shows "Analyzing X-Ray..." text; dropzone should also be `disabled` |

---

#### `frontend/src/components/ResultCard.jsx`

**Purpose:** Displays the prediction result with color-coded styling (red = PNEUMONIA, green = NORMAL), a confidence progress bar, model version, and inference time.

| Prop | Description |
|---|---|
| `result` | `PredictionResponse` object from the API |

---

#### `frontend/src/components/GradCamViewer.jsx`

**Purpose:** Side-by-side image display: original X-ray (browser object URL) on the left, Grad-CAM heatmap (base64 PNG) on the right.

| Prop | Description |
|---|---|
| `originalPreview` | Browser object URL from `URL.createObjectURL(file)` — should be revoked after use to avoid memory leaks |
| `gradcamImage` | Base64 string (`data:image/png;base64,...`) returned by the API |

---

#### `frontend/src/pages/Home.jsx`

**Purpose:** Top-level page component. Manages application state and orchestrates the three child components.

| State | Description |
|---|---|
| `loading` | Boolean; true while API call is in flight |
| `result` | Null or API response object |
| `error` | Null or error message string |
| `preview` | Object URL string for the uploaded image preview |

> **Known Issue (HO1):** `URL.createObjectURL(file)` is called on each upload but never revoked, creating a browser memory leak on repeated uploads. Fix: call `URL.revokeObjectURL(preview)` before reassigning.

---

### Docker & Deployment

---

#### `docker/Dockerfile.backend`

Multi-stage build: Stage 1 (builder) installs all Python packages. Stage 2 (runtime) starts from `python:3.10-slim`, copies site-packages from builder, installs OpenCV system dependencies (`libgl1-mesa-glx`, `libglib2.0-0`), copies application code and model weights, and sets the `HEALTHCHECK`.

> **Critical Bug (DF1):** The `HEALTHCHECK` uses `curl`, which is **not** installed in `python:3.10-slim`. The health check will always fail.  
> **Fix:** Add `curl` to the `apt-get install` line.

---

#### `docker/Dockerfile.frontend`

Multi-stage build: Stage 1 (`node:18-alpine`) installs npm deps and builds the Vite production bundle to `/app/dist`. Stage 2 (`nginx:alpine`) copies `dist/` to `/usr/share/nginx/html` and the `nginx.conf`. This image self-contains the static file server.

> **Known Issue (DFF1):** `npm ci` requires `package-lock.json`, which is listed in `.gitignore`. Either commit the lock file or switch to `npm install`.

---

#### `docker/nginx.conf`

Nginx reverse proxy configuration. Serves the React SPA from `/usr/share/nginx/html` (`location /`), forwards `/api/` requests to the FastAPI container at `http://backend:8000`, sets `client_max_body_size 20M` to allow large X-ray uploads, and sets `proxy_read_timeout 120s` to accommodate Grad-CAM generation time.

---

#### `docker-compose.yml`

Orchestrates four services: `postgres` (postgres:15-alpine with health check), `backend` (built from `Dockerfile.backend`), `frontend` (built from `Dockerfile.frontend`), `nginx` (bare nginx:alpine with `nginx.conf` mounted).

> **Critical Bug (DC2):** The standalone `nginx` service only has `nginx.conf` mounted — it has **NO** access to the React build files (those are inside the `frontend` container's filesystem).  
> **Fix Option A:** Remove the standalone `nginx` service and expose the `frontend` container's port 80 directly.  
> **Fix Option B:** Use a shared Docker volume for the `dist/` folder.

> **Security Bug (DC1):** `POSTGRES_PASSWORD` is hardcoded as `"password"` plaintext.  
> **Fix:** Use `${POSTGRES_PASSWORD}` from a `.env` file.

---

## 4. Function-by-Function Explanation

### ML Training Functions

---

#### `train_epoch(model, loader, criterion, optimizer, device)`

| Item | Detail |
|---|---|
| Parameters | `model` (nn.Module), `loader` (DataLoader), `criterion` (loss fn), `optimizer`, `device` (torch.device) |
| Returns | `(avg_loss: float, accuracy: float)` |
| Purpose | Executes one complete pass through the training DataLoader updating model weights |

**Internal steps:**

1. Sets `model.train()` to enable dropout and batch norm training mode
2. Iterates batches from loader (wrapped in `tqdm` for progress display)
3. Moves `imgs` and `labels` to device
4. Calls `optimizer.zero_grad()` to clear accumulated gradients
5. Runs forward pass: `outputs = model(imgs)`
6. Computes `loss = criterion(outputs, labels)` (CrossEntropyLoss)
7. `loss.backward()` computes gradients
8. `optimizer.step()` updates weights
9. Accumulates `total_loss`, correct predictions, and total count
10. Returns avg loss per batch and overall accuracy

---

#### `eval_epoch(model, loader, criterion, device)`

| Item | Detail |
|---|---|
| Parameters | Same as `train_epoch` minus `optimizer` |
| Returns | `(avg_loss: float, accuracy: float, roc_auc: float)` |
| Purpose | Evaluates model performance without updating weights; computes ROC-AUC for model selection |

Decorated with `@torch.no_grad()` to disable gradient tracking for efficiency. Collects all predictions and labels across the entire loader, then computes ROC-AUC using sklearn. Handles the edge case where all validation samples are the same class (returns `0.5` instead of raising `ValueError`).

---

#### `get_dataloaders(data_dir, batch_size, num_workers, use_sampler)`

| Item | Detail |
|---|---|
| Parameters | `data_dir` (str), `batch_size` (int, default 32), `num_workers` (int, default 4), `use_sampler` (bool, default True) |
| Returns | `(train_loader, val_loader, test_loader)` — three DataLoader objects |
| Purpose | Creates all three data loaders in one call; handles class imbalance via `WeightedRandomSampler` |

When `use_sampler=True`: calls `dataset.get_class_weights()` which returns per-sample float weights (higher weight for minority class). Creates `WeightedRandomSampler` with `replacement=True`. Sets `shuffle=False` on the training loader (cannot use both sampler and shuffle simultaneously).

---

#### `get_class_weights()` — ChestXRayDataset

| Item | Detail |
|---|---|
| Parameters | `self` |
| Returns | `torch.FloatTensor` of shape `[N]` where N = dataset length |
| Purpose | Enables `WeightedRandomSampler` to oversample the minority class (NORMAL) to balance training |

**Formula:** `weight_per_class = total_samples / (num_classes * count_per_class)`. Then each sample gets the weight of its class. This is the standard inverse frequency weighting approach.

---

#### `build_densenet121(pretrained, freeze_layers)`

| Item | Detail |
|---|---|
| Parameters | `pretrained` (bool, default True), `freeze_layers` (bool, default True) |
| Returns | `nn.Module` — modified DenseNet121 |
| Purpose | Creates the primary classification model using ImageNet pretrained weights with transfer learning |

**Internal logic:**

1. Loads `DenseNet121_Weights.IMAGENET1K_V1` if `pretrained=True`
2. If `freeze_layers=True`: sets `requires_grad=False` on ALL parameters
3. Unfreezes `features.denseblock4` and `features.norm5` (last dense block + batch norm)
4. Replaces `model.classifier` with: `Linear(1024, 256) → ReLU → Dropout(0.4) → Linear(256, 2)`
5. New classifier head is always trainable (`requires_grad=True` by default since freshly initialized)

The frozen backbone extracts rich features learned from ImageNet; the unfrozen `denseblock4` fine-tunes high-level features for the X-ray domain; the new head learns the binary classification boundary.

---

#### `generate_cam(input_tensor, class_idx)` — GradCAMService

| Item | Detail |
|---|---|
| Parameters | `input_tensor` `[1,3,224,224]` tensor, `class_idx` int |
| Returns | `numpy.ndarray` of shape `[H,W]` normalized to `[0,1]` |
| Purpose | Computes the Grad-CAM saliency map for the specified class |

**Internal steps:**

1. Sets `model.eval()`, zeros gradients
2. Runs forward pass: `self.model(input_tensor)` — triggers the forward hook, saving activations
3. Creates a one-hot output gradient vector and calls `output.backward(gradient=one_hot)` — triggers the backward hook, saving gradients
4. Pools gradients: `pooled = gradients.mean(dim=[2,3], keepdim=True)`
5. Computes weighted activation map: `cam = (pooled * activations).sum(dim=1).squeeze()`
6. Applies ReLU (retain only positive contributions)
7. Normalizes to `[0,1]`: `cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)`

---

### Backend API Functions

---

#### `predict(file, db)` — POST /api/predict

| Item | Detail |
|---|---|
| Parameters | `file` (UploadFile from FastAPI), `db` (Session from `Depends(get_db)`) |
| Returns | `PredictionResponse` Pydantic model |
| Called by | Any HTTP client POSTing to `/api/predict` |

Full execution pipeline: MIME validation → file size check → PIL decode → tensor preprocessing → inference → Grad-CAM → MD5 hash → DB insert → JSON response. The `gradcam_service` is lazily initialized on the first request and reused thereafter (potential race condition under concurrent load).

---

#### `load_model()` — InferenceService

| Item | Detail |
|---|---|
| Parameters | `self` |
| Returns | None (side-effects: sets `self.device`, `self.model`, `self._loaded`) |
| Called by | `backend/main.py` lifespan on server startup |

Guards with `if self._loaded: return` to prevent double loading. Detects CUDA. Loads model architecture from `MODEL_REGISTRY`. Calls `torch.load()` with `weights_only=True` (security: prevents arbitrary code execution from pickle). Loads state dict. Sets `model.eval()`.

---

#### `predict(tensor)` — InferenceService

| Item | Detail |
|---|---|
| Parameters | `tensor: torch.Tensor` of shape `[1,3,224,224]` |
| Returns | `Tuple[str, float, int]` — `(class_name, confidence, class_index)` |
| Called by | `predict.py` route after preprocessing |

Moves tensor to device. Runs model inside `torch.no_grad()` context. Applies softmax to logits. Takes argmax for class index. Returns class name from `CLASS_NAMES` list, confidence as float, and class index for Grad-CAM.

---

## 5. Class Explanation

### `ChestXRayDataset`

Inherits from `torch.utils.data.Dataset`. Serves as the bridge between CSV manifests and the DataLoader. Each item is a tuple `(image_tensor, label_int)`. The `get_class_weights()` method enables the training loader to balance the 3:1 PNEUMONIA:NORMAL skew without modifying disk data.

---

### `InferenceService` (Singleton)

Implements the Singleton pattern via `__new__`: the class maintains a single instance (`_instance`) and returns it on subsequent instantiation calls. The `_loaded` flag ensures the expensive model-loading operation (reading a ~85MB `.pth` file from disk) happens exactly once during the server lifecycle, making subsequent inference calls fast (pure memory operations).

In the system architecture, `InferenceService` is the central ML component. It sits between the route layer (`predict.py`) and the model weights file. The `GradCAMService` depends on the model instance it holds.

---

### `GradCAMService`

Wraps the model with PyTorch hook infrastructure. The core insight is that DenseNet's inplace ReLU operations break standard module-level backward hooks. The solution is to register the backward hook on the activation **tensor itself** inside the forward hook (`output.register_hook`), which captures the gradient before inplace operations can interfere.

In the system: instantiated once per server lifetime after `inference_service.model` is available; `generate_overlay()` is called inside the predict route for every uploaded image.

---

### `Settings` (Pydantic BaseSettings)

Central configuration hub. `BaseSettings` reads from the `.env` file first, then environment variables, then Python defaults — in that priority order. Used by `session.py` (`DATABASE_URL`), `inference.py` (`MODEL_PATH`, `MODEL_NAME`), `predict.py` (`MAX_UPLOAD_MB`, `MODEL_VERSION`).

---

## 6. Notebook Cell-by-Cell Explanation

### `01_eda.ipynb` — Exploratory Data Analysis

#### Cell 0 — Working Directory Fix
**Type:** Code  
Walks up the directory tree from the notebook's location until it finds `requirements.txt` (the project root marker), then calls `os.chdir()` to set CWD. This ensures all subsequent relative paths (e.g., `ml/data/raw`) resolve correctly regardless of where Jupyter was launched from.  
**Output:** prints `"Working directory set to: <project_root>"`

#### Cell 1 — Imports and Constants
**Type:** Code  
Imports: `pathlib`, `numpy`, `matplotlib`, `seaborn`, `PIL.Image`, `collections.Counter`. Sets `DATA_ROOT = Path('ml/data/raw')` and creates `ml/reports/` directory. No output.

#### Cell 2 — Image Count per Split and Class
**Type:** Code  
Defines `count_images(folder)` which globs for `*.jpeg`, `*.jpg`, and `*.png`. Builds a nested dict: `counts[split][class] = count`. Prints the counts for all three splits.  
**Expected output:** `TRAIN: {'NORMAL': 1341, 'PNEUMONIA': 3875} | VAL: {8, 8} | TEST: {234, 390}`

#### Cell 3 — Class Distribution Bar Chart
**Type:** Code  
Creates a 1×3 subplot figure with one bar chart per split. Blue bars = NORMAL, red bars = PNEUMONIA. Reveals the 3:1 class imbalance in training. Saves to `ml/reports/class_distribution.png`.

#### Cell 4 — Sample Image Grid
**Type:** Code  
Displays a 2×4 grid of sample images (first 4 from NORMAL and PNEUMONIA training folders). Reveals visual characteristics of normal vs pneumonia X-rays. Saves to `ml/reports/sample_images.png`.

#### Cell 5 — Image Size Distribution
**Type:** Code  
Iterates all NORMAL training images, records `(width, height)`. Prints min, max, mean for both dimensions. Output confirms wide size variation (912–2916 width, 672–2663 height), motivating the 224×224 resize in `augmentations.py`.

---

### `02_data_pipeline_test.ipynb`

#### Cell 0 — Working Directory Fix
Same as `01_eda` Cell 0. Also inserts project root into `sys.path` so `ml.training.*` imports work.

#### Cell 1 — DataLoader Smoke Test
**Type:** Code  
Calls `get_dataloaders(data_dir='ml/data', batch_size=8, num_workers=0)`. Fetches one batch from `train_loader`. Prints: `Image batch shape: torch.Size([8, 3, 224, 224])`, `Label batch: tensor([...0s and 1s...])`, and loader lengths. Verifies the entire pipeline from CSV → Dataset → DataLoader → tensor batch works end-to-end.

---

### `03_transfer_learning.ipynb`

#### Cells 0–2 — Setup, Device Detection, Data Loading
Standard setup: fix CWD, detect CUDA/CPU, instantiate train and val loaders with `batch_size=32`, `num_workers=0`.

#### Cell 3 — Model Build
**Type:** Code  
Calls `MODEL_REGISTRY['densenet121'](pretrained=True, freeze_layers=True)`. Downloads ImageNet weights (~85MB) on first run. Freezes backbone, unfreezes `denseblock4`, replaces classifier.

#### Cell 4 — Trainable Parameter Count
**Type:** Code  
Counts total vs trainable parameters.  
**Expected output:** `Total: ~7.2M | Trainable: ~2.4M (33.6%)`. Confirms that only the last dense block and new classifier head are trainable.

#### Cell 5 — Mini-epoch Test
**Type:** Code  
Runs 61 training batches (not a full epoch) with `CrossEntropyLoss` + Adam optimizer. Prints loss every 20 batches. Verifies forward + backward pass, gradient flow, and optimizer update work correctly.

---

### `04_gradcam_test.ipynb`

#### Cell 0 — Working Directory Fix
Same pattern. Fixes CWD and `sys.path`.

#### Cell 1 — Imports
Imports `torch`, `pathlib`, `glob`, `PIL`, `matplotlib`, `MODEL_REGISTRY`, `inference_transform`, `GradCAMService`.

#### Cell 2 — Load Trained Model
**Type:** Code  
Checks for `ml/saved_models/best_model_densenet121.pth`; raises `FileNotFoundError` with instructions if missing. Loads model architecture, applies saved weights with `weights_only=True`, sets eval mode. Prints confirmation with device.

#### Cell 3 — Find Test Sample
**Type:** Code  
Dynamically globs `ml/data/raw/test/PNEUMONIA/` for any image extension. Takes the first sorted match. Opens with PIL, applies `inference_transform`, adds batch dimension. Prints filename and original image dimensions. This avoids hardcoded filenames.

#### Cell 4 — Prediction
**Type:** Code  
Runs a separate `torch.no_grad()` inference pass to get class index and confidence for display. Note: `GradCAMService` does its OWN forward+backward pass internally, so this is a display-only pass.

#### Cell 5 — Grad-CAM Visualization
**Type:** Code  
Creates `GradCAMService` with `target_layer_name='features.norm5'`. Generates a fresh tensor clone for the CAM pass. Calls `generate_overlay()`. Plots original + heatmap side-by-side. Saves to `ml/reports/gradcam_sample.png`.

---

## 7. Data Flow Explanation

### Training Data Flow

```
Kaggle Dataset  (ml/data/raw/{split}/{CLASS}/*.jpeg)
    ↓  ml/data/create_csv.py
CSV Manifests  (ml/data/train.csv | val.csv | test.csv)
    columns: filepath, label (0/1), class
    ↓  ChestXRayDataset.__getitem__(idx)
PIL.Image.open(filepath).convert("RGB")
    ↓  train_transforms or val_transforms
torch.Tensor [3, 224, 224]  (normalized, augmented)
    ↓  DataLoader (batch_size=32, WeightedRandomSampler)
Batch tensor [32, 3, 224, 224] + labels [32]
    ↓  DenseNet121 forward pass
Logits [32, 2]
    ↓  CrossEntropyLoss
Scalar loss → backward() → Adam step
    ↓  Best checkpoint saved by ROC-AUC
ml/saved_models/best_model_densenet121.pth
```

### Inference Data Flow (Production)

```
Browser: User drops chest X-ray JPEG/PNG
    ↓  axiosClient.predictImage(file)
    ↓  POST /api/predict  (multipart/form-data)
FastAPI predict route:
  1. MIME + size validation
  2. PIL.Image.open(BytesIO(contents)).convert("RGB")
  3. ImagePreprocessor.process() → [1, 3, 224, 224] tensor
  4. InferenceService.predict(tensor)
       → DenseNet121 forward pass (no_grad)
       → softmax → argmax → (label, confidence, class_idx)
  5. GradCAMService.generate_overlay(tensor, original, class_idx)
       → forward+backward pass for gradient capture
       → cam [7×7] → resize to original → COLORMAP_JET → blend
       → overlay_to_base64() → "data:image/png;base64,..."
  6. PredictionLog insert → db.commit()
    ↓  JSON: { prediction, confidence, gradcam_image, inference_ms, ... }
Browser:
  ResultCard renders prediction + confidence bar
  GradCamViewer renders side-by-side original + heatmap
```

---

## 8. Dependencies

### Python Dependencies (`requirements.txt`)

| Package | Why Used |
|---|---|
| `torch` / `torchvision` | Core deep learning framework; DenseNet121 model architecture and pretrained weights; all tensor operations |
| `fastapi` | High-performance async Python web framework for the REST API |
| `uvicorn[standard]` | ASGI server that runs the FastAPI application |
| `python-multipart` | Required by FastAPI to parse `multipart/form-data` file uploads |
| `pydantic` / `pydantic-settings` | Data validation for API request/response schemas and `.env`-driven configuration |
| `Pillow` | PIL Image: decode uploaded images, convert colorspaces, save overlays |
| `opencv-python-headless` | Headless OpenCV: `applyColorMap` (jet colormap) and `addWeighted` (heatmap blending) |
| `albumentations` | Advanced image augmentation library (declared but not actively used in current code) |
| `sqlalchemy` | ORM for `prediction_logs` database table; engine, session, and model definition |
| `alembic` | Database schema migration tool; version-controlled schema changes |
| `psycopg2-binary` | PostgreSQL adapter for SQLAlchemy (used in Docker Compose deployment) |
| `scikit-learn` | `roc_auc_score`, `classification_report`, `confusion_matrix` in `evaluate.py` |
| `numpy` | Array operations throughout ML pipeline and Grad-CAM computation |
| `pandas` | Reading CSV manifests in `ChestXRayDataset` |
| `matplotlib` / `seaborn` | EDA plots, confusion matrix heatmaps, ROC curves in `evaluate.py` and notebooks |
| `tqdm` | Progress bars in training loop |
| `jupyter` / `ipykernel` | Running the four analysis notebooks |
| `pytest` / `httpx` / `pytest-asyncio` | Backend integration testing |
| `boto3` | AWS S3 model storage (optional, declared but not actively used) |
| `black` / `flake8` / `isort` | Code formatting and linting tools |

### Frontend Dependencies (`package.json`)

| Package | Why Used |
|---|---|
| `react` / `react-dom` | Core UI library and DOM renderer |
| `react-router-dom` | Client-side routing (currently only one route: `/`) |
| `react-dropzone` | Drag-and-drop file upload zone with MIME and size validation |
| `axios` | HTTP client for `POST /api/predict` and `GET /api/predictions` |
| `vite` | Lightning-fast frontend build tool and dev server |
| `tailwindcss` / `@tailwindcss/postcss` / `autoprefixer` | Utility-first CSS framework for styling all components |

---

## 9. System Execution Flow

### Local Development Startup

1. Developer activates venv: `.\venv\Scripts\Activate.ps1` (Windows) or `source venv/bin/activate` (Unix)
2. Runs: `uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload`
3. FastAPI app instantiated; lifespan context manager triggered
4. `Base.metadata.create_all(bind=engine)` — creates `prediction_logs` table in SQLite if not exists
5. `inference_service.load_model()` — detects device (CPU/CUDA), loads DenseNet121, reads `best_model_densenet121.pth`, sets eval mode. Prints `"Model loaded on {device}: densenet121 v1.0"`
6. Server ready. Uvicorn listens on `http://0.0.0.0:8000`
7. In a second terminal: `cd frontend && npm run dev`
8. Vite dev server starts on `http://localhost:5173` with HMR
9. Browser opens `localhost:5173`; React renders `Home.jsx` with `ImageUploader`

### Per-Request Prediction Flow

1. User drags an X-ray into the upload zone
2. `react-dropzone` validates MIME type and size client-side
3. `onUpload(file)` called → `handleUpload()` sets `loading=true`, creates object URL for preview
4. `axiosClient.predictImage(file)` builds `FormData` and POSTs to `http://localhost:8000/api/predict`
5. FastAPI receives: validates `Content-Type`, reads body, checks MB limit
6. `PIL.Image.open(BytesIO(contents)).convert("RGB")` decodes the image
7. `ImagePreprocessor.process(image)` → `Resize(224,224) → ToTensor → Normalize → unsqueeze` → `[1,3,224,224]`
8. `InferenceService.predict(tensor)`: forward pass → softmax → `(label, confidence, class_idx)` returned
9. `GradCAMService.generate_overlay()`: second forward+backward pass; gradient pooling; colormap; blend → PIL Image
10. `overlay_to_base64()` → `"data:image/png;base64,..."` string
11. MD5 hash computed; `PredictionLog` inserted; `db.commit()`
12. `PredictionResponse` JSON returned
13. React: `loading=false`, `result=data`. `ResultCard` renders prediction+confidence. `GradCamViewer` renders side-by-side images

### Docker Compose Startup

1. `docker-compose up --build -d`
2. `postgres` service starts; healthcheck polls `pg_isready` every 10s
3. `backend` service waits for postgres healthcheck to pass, then builds and starts; loads model from copied `.pth` file
4. `frontend` service builds React bundle; its internal nginx serves on port 80
5. `nginx` service starts; proxies `/api/` to `backend:8000`, serves `/` from `/usr/share/nginx/html`

> **Note (DC2 Bug):** The standalone nginx does not have access to the React files — the `frontend` service's own nginx is the correct static file server. Remove the standalone `nginx` service from `docker-compose.yml` and expose `frontend` port 80 directly.

---

## 10. Additional Notes

### Design Decisions

| Decision | Rationale |
|---|---|
| DenseNet121 as primary model | Dense connections enable strong gradient flow; excellent performance on medical imaging benchmarks; ~7M params trainable on CPU in reasonable time |
| `features.norm5` as Grad-CAM target | Final normalization layer captures the highest-level semantic features before global pooling; produces the most semantically meaningful saliency maps |
| SQLite default with PostgreSQL option | SQLite requires zero configuration for local development; switching to PostgreSQL for production requires only a `DATABASE_URL` change |
| Singleton `InferenceService` | Loading an 85MB model from disk for every request would add ~400ms latency; singleton ensures one-time cost at startup |
| Lazy `GradCAMService` init | `GradCAMService` wraps the model with hooks; initializing it once after model load avoids repeated hook registration |
| CSV-based dataset manifests | Decouples file discovery from runtime loading; CSVs can be versioned, filtered, and shared without moving image files |
| `WeightedRandomSampler` over oversampling | Addresses 3:1 class imbalance statistically without duplicating files on disk |
| Working directory fix in notebooks | All notebooks use a root-walker pattern to find project root, making them runnable from any CWD |

---

### Known Issues & Bugs

| ID | Severity | Issue | Fix |
|---|---|---|---|
| DC2 | 🔴 CRITICAL | Docker nginx and frontend containers disconnected; React app not served | Remove standalone nginx OR use shared volume for `dist/` |
| DF1 | 🔴 CRITICAL | `curl` missing in backend image; `HEALTHCHECK` always fails | Add `curl` to `apt-get install` in `Dockerfile.backend` |
| DC1 | 🔴 HIGH | Hardcoded `POSTGRES_PASSWORD` in `docker-compose.yml` | Use `${POSTGRES_PASSWORD}` from `.env` |
| G1 | 🔴 HIGH | `GradCAMService` gradients/activations are instance state — not thread-safe | Use per-request local variables or `threading.Lock()` |
| S1 | 🔴 HIGH | `os.getenv()` defaults in `Settings` bypass Pydantic `.env` loading | Use bare Python defaults in `BaseSettings` fields |
| AX1 | 🔴 HIGH | Manual `Content-Type` header in axios breaks multipart boundary | Remove `headers` override from `predictImage()` |
| TS1 | 🔴 HIGH | Test hardcodes path to gitignored raw data file; fails on CI | Use pytest fixture to create a synthetic JPEG in memory |
| C1/D1 | 🟡 MEDIUM | `*.jpeg` glob misses `.jpg` files from Kaggle dataset | Add `*.jpg` and `*.png` to glob patterns in `create_csv.py` and `dataset.py` |
| HO1 | 🟡 MEDIUM | `URL.createObjectURL` never revoked — browser memory leak | Call `URL.revokeObjectURL(preview)` before reassigning |
| I2 | 🟡 MEDIUM | Double forward pass for inference + Grad-CAM — wasteful | Merge into single forward+backward pass |

---

### Performance Considerations

- CPU inference takes ~200–800ms per image depending on hardware. GPU reduces this to ~20–50ms
- Grad-CAM requires a second full forward+backward pass, roughly doubling inference time. A single-pass implementation would halve this
- The val split (16 images) is too small for reliable AUC estimation during training. Consider using a larger portion of train data as validation
- For production scale, consider async database writes (run DB insert in a background task) to reduce API response latency
- Model quantization (INT8) could reduce memory footprint by ~4× with minimal accuracy loss for CPU deployment

---

### Potential Improvements

- Install project as a package (`pip install -e .` with `pyproject.toml`) to eliminate all `sys.path.insert()` hacks
- Add `/v1/` API prefix for versioning; breaking changes would use `/v2/`
- Add async SQLAlchemy session for true async DB operations without blocking the event loop
- Implement request-level `GradCAMService` instantiation (or use `threading.Lock`) for thread safety
- Add a front-end result history page using the `GET /api/predictions` endpoint
- Replace MD5 with SHA-256 for image hashing (or align column size to 32 chars for MD5)
- Add model warm-up call after loading to avoid cold-start latency on the first real request
- Integrate Google Colab training workflow (see `COLAB_TRAINING_GUIDE.md`) with automated model download and deployment
- Add multi-class support for distinguishing bacterial vs viral pneumonia

---

### Best Practices Used

- **Transfer learning with selective unfreezing** — only fine-tuning the top layers preserves learned image features while adapting to the X-ray domain
- **Weighted sampling for class imbalance** — avoids naive oversampling that would create duplicate training examples
- **Pydantic for configuration and response validation** — catches type errors at startup rather than at runtime
- **SQLAlchemy with dependency injection** (`Depends(get_db)`) — ensures sessions are always closed, even on exceptions
- **`weights_only=True` in `torch.load()`** — security best practice preventing arbitrary code execution from untrusted checkpoint files
- **Docker multi-stage builds** — keeps final images small by separating build dependencies from runtime dependencies
- **Alembic for schema migrations** — version-controlled database evolution safe for production deployments
- **Lifespan context manager in FastAPI** — proper startup/shutdown resource management replacing deprecated `@app.on_event`

---

## 11. Quick Reference Card

### Key Commands

| Task | Command |
|---|---|
| Install dependencies | `pip install -r requirements.txt` |
| Generate CSV manifests | `python ml/data/create_csv.py` |
| Train model (local) | `python ml/training/train.py --model densenet121 --epochs 20` |
| Evaluate model | `python ml/training/evaluate.py` |
| Start backend | `uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload` |
| Start frontend | `cd frontend && npm install && npm run dev` |
| Run tests | `pytest backend/tests/` |
| DB migration | `cd backend && alembic upgrade head` |
| Docker build + run | `docker-compose up --build -d` |
| Docker logs | `docker-compose logs -f backend` |

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/predict` | Upload X-ray; returns prediction + Grad-CAM |
| `GET` | `/api/health` | Model status and device info |
| `GET` | `/api/predictions` | Paginated prediction history (limit, skip params) |
| `GET` | `/docs` | FastAPI auto-generated Swagger UI |
| `GET` | `/` | Root welcome message |

### Important File Paths

| Item | Path |
|---|---|
| Trained model weights | `ml/saved_models/best_model_densenet121.pth` |
| Training history | `ml/saved_models/training_history.json` |
| Test metrics | `ml/reports/metrics.json` |
| Environment config | `.env` (copy from `.env.example`) |
| Training CSV | `ml/data/train.csv` |
| Validation CSV | `ml/data/val.csv` |
| Test CSV | `ml/data/test.csv` |

### Dataset Split Summary

| Split | NORMAL | PNEUMONIA | Total |
|---|---|---|---|
| Train | 1,341 | 3,875 | 5,216 |
| Val | 8 | 8 | 16 |
| Test | 234 | 390 | 624 |

> **Note:** Pneumonia recall (sensitivity) is the critical metric — a missed positive (false negative) is clinically more dangerous than a false alarm.
