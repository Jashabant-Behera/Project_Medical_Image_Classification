# Code Review — Medical Image Classification Project
> Generated: 2026-03-13 | Scope: All layers (ML, Backend, Frontend, Docker)

---

## Table of Contents
1. [Overall Architecture & Flow](#1-overall-architecture--flow)
2. [ML Layer Review](#2-ml-layer-review)
3. [Backend Layer Review](#3-backend-layer-review)
4. [Frontend Layer Review](#4-frontend-layer-review)
5. [Docker & Deployment Review](#5-docker--deployment-review)
6. [Cross-Cutting Issues](#6-cross-cutting-issues)
7. [Issue Summary Table](#7-issue-summary-table)

---

## 1. Overall Architecture & Flow

### End-to-End Request Flow

```
User (Browser)
  │── drop image ──> ImageUploader.jsx
  │                      │
  │                      ▼
  │               axiosClient.predictImage()
  │                  POST /api/predict (multipart/form-data)
  │                      │
  │               FastAPI (backend/main.py)
  │                      │
  │               predict.py route
  │                 ├── validate content_type & size
  │                 ├── PIL.Image.open()
  │                 ├── ImagePreprocessor.process() → [1,3,224,224] tensor
  │                 ├── InferenceService.predict()  → label, confidence, class_idx
  │                 ├── GradCAMService.generate_overlay() → PIL Image
  │                 ├── overlay_to_base64()          → data:image/png;base64,...
  │                 ├── PredictionLog → db.commit()
  │                 └── PredictionResponse JSON
  │
  │<──── JSON {prediction, confidence, gradcam_image, inference_ms, ...}
  │
  ├── ResultCard.jsx   (shows label + confidence bar)
  └── GradCamViewer.jsx (side-by-side original + heatmap)
```

### Overall Assessment
The architecture is clean, well-separated, and follows the correct flow. The main concerns are **correctness bugs**, **security gaps**, **test fragility**, and several **code quality** items identified below.

---

## 2. ML Layer Review

### 2.1 `ml/training/dataset.py`

| # | Severity | Issue | Location |
|---|----------|-------|----------|
| D1 | Medium | `glob('*.jpeg')` only matches `.jpeg` extension. PNG images and `.jpg` files in the dataset are silently skipped. | Line 10 |
| D2 | Medium | No error handling if `row['filepath']` does not exist on disk — will crash mid-epoch with generic `FileNotFoundError`. | Line 19 |
| D3 | Low | `class_names` attribute is defined but never used inside the class itself (no `idx_to_class` lookup method provided). | Line 12 |
| D4 | Low | Missing `__repr__` or `__str__` for debugging purposes. | — |

**Flow note:** `get_class_weights()` correctly computes inverse-frequency weights. Logic is sound.

---

### 2.2 `ml/training/augmentations.py`

| # | Severity | Issue | Location |
|---|----------|-------|----------|
| A1 | Low | `inference_transform = val_transforms` is an alias (same object reference). If `val_transforms` is ever mutated, `inference_transform` changes too. Should be a separate `Compose`. | Line 27 |
| A2 | Low | `transforms.ToTensor()` is deprecated in favour of `transforms.v2` pipeline in newer torchvision. Not breaking now but will be in future versions. | Lines 15, 21 |

**Flow note:** Train/val split is correct. Augmentations are medically appropriate (flip, slight rotation, colour jitter for X-ray variance). No issue with normalization constants.

---

### 2.3 `ml/training/dataloader.py`

| # | Severity | Issue | Location |
|---|----------|-------|----------|
| L1 | **HIGH** | `pin_memory=True` is set unconditionally. On Windows CPU-only machines this can cause slow DataLoader init or subtle errors. Should be `pin_memory=(device.type == 'cuda')`. | Lines 21, 23, 25 |
| L2 | Medium | `num_workers=4` default — on Windows, `num_workers > 0` requires the `if __name__ == '__main__'` guard in the calling script. If `get_dataloaders` is called from a notebook directly, it can hang or crash. | Line 7 |
| L3 | Low | No `persistent_workers=True` flag — would improve performance when `num_workers > 0`. | — |

---

### 2.4 `ml/training/model.py`

| # | Severity | Issue | Location |
|---|----------|-------|----------|
| M1 | Medium | `build_densenet121` unfreezes `denseblock4` when `freeze_layers=True`, but the new classifier head is always trainable. This is correct but undocumented — can confuse readers. | Lines 19–36 |
| M2 | Low | `build_resnet50` and `build_efficientnet_b0` have no `freeze_layers` parameter — inconsistent API vs. `build_densenet121`. | Lines 39–65 |
| M3 | Low | `NUM_CLASSES = 2` is a module-level constant but ResNet/EfficientNet builders duplicate the value inline instead of referencing it. | Lines 46, 63 |

---

### 2.5 `ml/training/train.py`

| # | Severity | Issue | Location |
|---|----------|-------|----------|
| T1 | Medium | Scheduler patience (3) and early-stopping patience (5) are independent and can interact unexpectedly — LR drops before early stop triggers. Relationship should be documented. | Lines 63–64, 94 |
| T2 | Medium | `verbose=True` on `ReduceLROnPlateau` is deprecated in PyTorch >= 2.2 — will cause a `UserWarning`. | Line 64 |
| T3 | Low | `train_acc` is not logged in history dict — makes post-training overfitting analysis harder. | Line 83 |
| T4 | Low | `training_history.json` saved to `save_dir` — mixes model checkpoints with training logs. | Line 98 |
| T5 | Low | No random seed set (`torch.manual_seed`) — results are non-reproducible between runs. | — |

---

### 2.6 `ml/training/evaluate.py`

| # | Severity | Issue | Location |
|---|----------|-------|----------|
| E1 | Medium | `plt.figure()` called twice but `plt.close()` never called between — memory leak for large runs or notebook contexts. | Lines 49, 58 |
| E2 | Medium | `f1_score` is imported but never used. Dead import. | Line 6 |
| E3 | Low | `torch, numpy as np` on the same import line is non-standard and against PEP 8. | Line 2 |

---

### 2.7 `scripts/create_csv.py`

| # | Severity | Issue | Location |
|---|----------|-------|----------|
| C1 | **HIGH** | Only `*.jpeg` files are globbed. The Kaggle Chest X-Ray dataset also contains `.jpg` files. These will be silently excluded, creating an incomplete dataset. | Line 10 |
| C2 | Medium | Script runs at module import time — no `if __name__ == '__main__'` guard. Importing anywhere would trigger CSV creation. | Lines 19–20 |
| C3 | Low | Uses `str.replace('\\', '/')` — idiomatic alternative is `pathlib.Path.as_posix()`. | Line 11 |

---

## 3. Backend Layer Review

### 3.1 `backend/config/settings.py`

| # | Severity | Issue | Location |
|---|----------|-------|----------|
| S1 | **HIGH** | `os.getenv(...)` used as Pydantic field defaults. `os.getenv()` runs at class definition time (before `BaseSettings` reads `.env`), so `.env`-only variables are NOT picked up by `os.getenv()`. Should use Pydantic default values directly. | Lines 7–13 |
| S2 | Medium | `class Config` inner class is Pydantic v1 style. Pydantic v2 uses `model_config = SettingsConfigDict(...)`. Deprecation warnings on every startup. | Lines 15–18 |
| S3 | Low | `SECRET_KEY` is defined but never used anywhere in the codebase. Dead field. | Line 13 |
| S4 | Low | `from pathlib import Path` is imported but never used. | Line 2 |

---

### 3.2 `backend/db/session.py`

| # | Severity | Issue | Location |
|---|----------|-------|----------|
| DB1 | Medium | `get_db()` is a synchronous generator inside `async def` routes — blocks the event loop. For production, use `asyncpg` + SQLAlchemy async session. | Lines 15–20 |
| DB2 | Low | No connection pool size configuration (`pool_size`, `max_overflow`) for PostgreSQL. | Line 9 |

---

### 3.3 `backend/models/prediction.py`

| # | Severity | Issue | Location |
|---|----------|-------|----------|
| ORM1 | Medium | `created_at` uses application-side `datetime.now()`. More reliable to use `server_default=func.now()` (DB-side timestamp). | Line 14 |
| ORM2 | Low | `image_hash` column is `String(64)` sized for SHA256, but `predict.py` uses MD5 (32 chars). Inconsistent. | Line 13 |

---

### 3.4 `backend/services/inference.py`

| # | Severity | Issue | Location |
|---|----------|-------|----------|
| I1 | **HIGH** | `sys.path.insert(0, ...)` to resolve `ml.training.model`. Fragile — breaks if launched outside project root. Should use `PYTHONPATH` or proper packaging. | Lines 2–3 |
| I2 | Medium | Double forward pass: `predict()` runs the model with `no_grad`, then `generate_cam()` runs it again with gradients. Wasteful — should be merged into a single pass. | Lines 28–36 |
| I3 | Medium | `print()` used for logging instead of the `logging` module. Inconsistent with `main.py`. | Line 26 |
| I4 | Low | Singleton `_loaded` flag set in `__new__` — unusual pattern. Cleaner to use `__init__` with a class-level guard. | Lines 12–16 |

---

### 3.5 `backend/services/gradcam.py`

| # | Severity | Issue | Location |
|---|----------|-------|----------|
| G1 | **HIGH** | `self.gradients` and `self.activations` are instance-level state — stale under concurrent requests. Not thread-safe. | Lines 35–39 |
| G2 | Medium | `gradcam_service` global variable in `predict.py` is lazily initialized — race condition under concurrent async requests. | predict.py Lines 17, 46–47 |
| G3 | Low | Forward hook registered in `__init__` is never removed (`hook.remove()`). Accumulates if multiple instances are created. | Line 23 |
| G4 | Low | `overlay_to_base64` creates a `BytesIO` buffer but never explicitly closes it. Should use `with io.BytesIO() as buf`. | Lines 79–82 |

---

### 3.6 `backend/api/routes/predict.py`

| # | Severity | Issue | Location |
|---|----------|-------|----------|
| P1 | **HIGH** | `tensor.requires_grad = True` is set, then `predict()` wraps inference in `torch.no_grad()` — contradictory. Grad-CAM then needs a separate forward pass. Should use a clean tensor clone for Grad-CAM and skip `requires_grad` setup here. | Lines 37–38, 48 |
| P2 | Medium | No `db.rollback()` on exception — broken session state if DB commit fails. | Lines 53–56 |
| P3 | Medium | `'image/jpg'` is not a valid MIME type (correct is `'image/jpeg'`). File type checking via MIME is unreliable — PIL format detection is more robust. | Line 19 |
| P4 | Medium | Full file read into memory before size check — a 1GB upload would be fully buffered before rejection. | Lines 27–28 |
| P5 | Low | `import torch` is unused in this file. Dead import. | Line 13 |

---

### 3.7 `backend/api/routes/health.py`

| # | Severity | Issue | Location |
|---|----------|-------|----------|
| H1 | Medium | `import torch` is unused. Dead import. | Line 1 |
| H2 | Medium | Accesses `inference_service._loaded` directly. Should expose a public property `is_ready`. | Line 10 |
| H3 | Low | Health check does not verify DB connectivity. Production health check should ping the DB. | — |

---

### 3.8 `backend/api/routes/history.py`

| # | Severity | Issue | Location |
|---|----------|-------|----------|
| HI1 | Medium | Two separate queries for `COUNT(*)` and data records — `total` can be stale between them. Should wrap in a single transaction. | Lines 12–15 |
| HI2 | Low | No maximum cap on `limit` query param — a client could dump the entire DB. Should cap at e.g. 200. | Line 10 |

---

### 3.9 `backend/tests/test_predict.py`

| # | Severity | Issue | Location |
|---|----------|-------|----------|
| TS1 | **HIGH** | `test_predict_valid_image` hardcodes path to `ml/data/raw/test/NORMAL/IM-0001-0001.jpeg` — in `.gitignore`, always fails on CI/fresh clone. Should use a synthetic test fixture image. | Line 17 |
| TS2 | Medium | Module-level `client = TestClient(app)` is created but never used (each test creates its own). Dead code. | Line 7 |
| TS3 | Medium | `import shutil` is unused. Dead import. | Line 5 |
| TS4 | Medium | Tests load the real ML model on every run — slow and depends on un-committed files. Should mock `inference_service`. | — |
| TS5 | Low | No test for `/api/predictions` (history) endpoint. | — |
| TS6 | Low | No test for oversized file (>16MB) rejection. | — |

---

## 4. Frontend Layer Review

### 4.1 `frontend/src/api/axiosClient.js`

| # | Severity | Issue | Location |
|---|----------|-------|----------|
| AX1 | Medium | `'Content-Type': 'multipart/form-data'` set manually for FormData POST. Axios auto-sets the correct boundary when omitted. Setting it manually breaks the boundary string and can cause the server to fail file parsing. **Should be removed.** | Line 11 |
| AX2 | Low | No request timeout — slow server hangs the browser indefinitely. Add `timeout: 60000`. | Line 5 |
| AX3 | Low | No global error interceptor for non-JSON 500 responses. | — |

---

### 4.2 `frontend/src/components/ImageUploader.jsx`

| # | Severity | Issue | Location |
|---|----------|-------|----------|
| U1 | Medium | Dropzone is still active while `loading=true` — user can drop another file mid-request. `onDrop` should be no-op when loading, or use the `disabled` prop. | Lines 4–8 |
| U2 | Medium | `<button>` inside dropzone has no `type="button"` — default button type is `submit`, can cause accidental form submission. | Line 23 |
| U3 | Low | `onDropRejected` is not handled — silently rejects oversized or wrong-type files with no user feedback. | — |

---

### 4.3 `frontend/src/pages/Home.jsx`

| # | Severity | Issue | Location |
|---|----------|-------|----------|
| HO1 | Medium | `URL.createObjectURL(file)` never revoked — memory leak, one object URL per uploaded image accumulates until page refresh. | Line 15 |
| HO2 | Medium | Old `preview` briefly shows while new file is being set. Should `setPreview(null)` first. | Lines 14–15 |
| HO3 | Low | No dedicated loading spinner/progress component — only text inside the uploader zone. | — |
| HO4 | Low | Browser tab title not updated dynamically — shows Vite default. | — |

---

### 4.4 `frontend/src/components/GradCamViewer.jsx`

| # | Severity | Issue | Location |
|---|----------|-------|----------|
| GV1 | Low | Images have no `width`/`height` attributes — causes cumulative layout shift (CLS). | Lines 15, 20 |
| GV2 | Low | No loading state for Grad-CAM image — large base64 strings cause visible flicker. | — |

---

### 4.5 `frontend/src/components/ResultCard.jsx`

No significant issues. Implementation is clean and correct.

| # | Severity | Issue | Location |
|---|----------|-------|----------|
| RC1 | Low | `result.timestamp` not shown — adding it would improve result traceability in the UI. | — |

---

### 4.6 `frontend/tailwind.config.js`

| # | Severity | Issue | Location |
|---|----------|-------|----------|
| TW1 | Medium | `theme: { extend: {} }` has no custom design tokens. All classes are inline strings scattered across components — a design system in `extend` would help avoid duplication. | — |

---

## 5. Docker & Deployment Review

### 5.1 `docker/Dockerfile.backend`

| # | Severity | Issue | Location |
|---|----------|-------|----------|
| DF1 | **HIGH** | `HEALTHCHECK` uses `curl` but `curl` is NOT installed in `python:3.10-slim`. Health check will always fail. Must add `curl` to `apt-get install`. | Line 28 |
| DF2 | **HIGH** | Dockerfile pins Python to `3.10` but `requirements.txt` is fully unpinned. Future `pip install` can pull incompatible versions. Both must be pinned. | Lines 1, 7 |
| DF3 | Medium | ALL packages (including `jupyter`, `boto3`, `black`, `flake8`) installed in production image — massively inflates image size. Split into `requirements-prod.txt` and `requirements-dev.txt`. | Lines 3–5 |
| DF4 | Medium | Container runs as root — security risk. Add `RUN useradd -m appuser && USER appuser`. | — |
| DF5 | Low | `ENV MODEL_PATH` and `ENV LOG_LEVEL` in Dockerfile are overridden by `docker-compose.yml` `env_file` — confusing precedence. | Lines 23–24 |

---

### 5.2 `docker/Dockerfile.frontend`

| # | Severity | Issue | Location |
|---|----------|-------|----------|
| DFF1 | Medium | `npm ci` requires `package-lock.json`, which is in `.gitignore`. Dockerfile and `.gitignore` are in conflict — Docker build will fail on a fresh clone. Use `npm install` or commit the lock file. | Lines 3–4 |
| DFF2 | Low | No `.dockerignore` — `node_modules`, `.git`, `*.pth` model files sent to build context, slowing builds significantly. | — |

---

### 5.3 `docker-compose.yml`

| # | Severity | Issue | Location |
|---|----------|-------|----------|
| DC1 | **HIGH** | `POSTGRES_PASSWORD: password` is hardcoded plaintext. Use `${POSTGRES_PASSWORD}` from environment. | Line 7 |
| DC2 | **HIGH** | The `frontend` service and `nginx` service are disconnected. `nginx` mounts only `docker/nginx.conf` but has no access to the built React files. The `Dockerfile.frontend` already embeds files into its own nginx — so a separate `nginx` service is redundant and wrong. Remove the standalone `nginx` service OR remove the separate `frontend` service and let `Dockerfile.frontend` handle serving. | Lines 30–47 |
| DC3 | Medium | `env_file: .env` for backend service but `.env` is in `.gitignore`  — `docker-compose up` fails on a fresh clone. Needs documentation or a setup step. | Line 21 |
| DC4 | Low | No `restart: unless-stopped` policy. Containers won't recover from crashes. | — |

---

### 5.4 `docker/nginx.conf`

No critical issues. Configuration is functional.

| # | Severity | Issue | Location |
|---|----------|-------|----------|
| NG1 | Low | No gzip compression — JS bundles and JSON responses served uncompressed. | — |
| NG2 | Low | No security headers (`X-Frame-Options`, `X-Content-Type-Options`, CSP). | — |

---

## 6. Cross-Cutting Issues

| # | Severity | Issue | Affected Files |
|---|----------|-------|----------------|
| CC1 | **HIGH** | No `.dockerignore` — entire project (venv, `.git`, `ml/data/`) sent as Docker build context. | Root |
| CC2 | **HIGH** | `sys.path.insert()` hacks in 3 files. Making the project installable via `pip install -e .` with a `pyproject.toml` eliminates all of these. | inference.py, train.py, evaluate.py |
| CC3 | Medium | No file logging — all logs to stdout only. Production services need rotating file handler or structured JSON logging. | backend/main.py |
| CC4 | Medium | `alembic.ini` lives in `backend/` — Alembic commands must be run from `backend/`, undocumented and fragile. | backend/alembic.ini |
| CC5 | Medium | `Base.metadata.create_all()` called in `lifespan()` bypasses Alembic migrations entirely. If using Alembic, `create_all()` should be replaced with `alembic upgrade head` as a startup step. | backend/main.py Line 18 |
| CC6 | Medium | `allow_origins=['*']` — acceptable for dev but must be restricted in production. | backend/main.py Line 32 |
| CC7 | Low | No `pyproject.toml` / `setup.py` — project not installable as a package; root cause of all `sys.path` hacks. | Root |
| CC8 | Low | No API versioning (no `/v1/` prefix). Breaking API changes would require all clients to update simultaneously. | backend/api/routes/ |
| CC9 | Low | Pydantic v1 `class Config` pattern in multiple places emits deprecation warnings on every startup. | settings.py, response_models.py |

---

## 7. Issue Summary Table

| ID | File | Severity | Short Description |
|----|------|----------|-------------------|
| D1 | dataset.py | Medium | Only `.jpeg` globbed, `.jpg`/`.png` skipped |
| D2 | dataset.py | Medium | No error handling for missing image files |
| A1 | augmentations.py | Low | `inference_transform` is same object reference as `val_transforms` |
| L1 | dataloader.py | **HIGH** | `pin_memory=True` unconditional, issues on CPU-only Windows |
| C1 | create_csv.py | **HIGH** | Only `*.jpeg` globbed, misses `.jpg` files |
| C2 | create_csv.py | Medium | No `if __name__ == '__main__'` guard |
| T2 | train.py | Medium | `verbose=True` deprecated in PyTorch >= 2.2 |
| T5 | train.py | Low | No random seed — non-reproducible runs |
| E1 | evaluate.py | Medium | `plt.close()` never called — memory leak |
| E2 | evaluate.py | Medium | `f1_score` imported but unused |
| S1 | settings.py | **HIGH** | `os.getenv()` defaults bypass Pydantic `.env` loading |
| S2 | settings.py | Medium | Pydantic v1 `class Config` — deprecation warnings |
| DB1 | session.py | Medium | Sync DB calls inside async routes — event loop blocking |
| ORM2 | prediction.py | Low | `image_hash` column sized for SHA256 but MD5 is used |
| I1 | inference.py | **HIGH** | `sys.path.insert` fragile — breaks outside project root |
| I2 | inference.py | Medium | Double forward pass for inference + GradCAM — wasteful |
| G1 | gradcam.py | **HIGH** | Stale gradients/activations under concurrent requests |
| G2 | gradcam.py | Medium | `gradcam_service` global init race condition |
| P1 | predict.py | **HIGH** | `requires_grad=True` then `no_grad()` — contradictory setup |
| P2 | predict.py | Medium | No DB rollback on exception |
| P3 | predict.py | Medium | `image/jpg` is not a valid MIME type |
| P4 | predict.py | Medium | Entire file buffered in memory before size check |
| H1 | health.py | Medium | `import torch` unused |
| HI2 | history.py | Low | No max cap on `limit` query param |
| TS1 | test_predict.py | **HIGH** | Test hardcodes path to un-committed raw data file |
| TS4 | test_predict.py | Medium | Real model loaded in tests — slow, fragile |
| AX1 | axiosClient.js | Medium | Manual `Content-Type: multipart/form-data` breaks Axios boundary |
| U1 | ImageUploader.jsx | Medium | Dropzone active while loading — double-submit risk |
| HO1 | Home.jsx | Medium | `createObjectURL` never revoked — memory leak |
| DF1 | Dockerfile.backend | **HIGH** | `curl` not installed — HEALTHCHECK always fails |
| DF2 | Dockerfile.backend | **HIGH** | Unpinned `requirements.txt` in Docker — non-reproducible |
| DF3 | Dockerfile.backend | Medium | Dev/notebook packages included in production image |
| DF4 | Dockerfile.backend | Medium | Container runs as root — security risk |
| DFF1 | Dockerfile.frontend | Medium | `npm ci` requires `package-lock.json` but it is gitignored |
| DC1 | docker-compose.yml | **HIGH** | Hardcoded DB password in plaintext |
| DC2 | docker-compose.yml | **HIGH** | Frontend + NGINX service disconnect — React app not served |
| CC1 | Root | **HIGH** | No `.dockerignore` — massive slow build context |
| CC2 | Multiple | **HIGH** | `sys.path.insert()` hacks — fragile imports |
| CC5 | main.py | Medium | `create_all()` bypasses Alembic migrations |

---

## Priority Fix Order (Do First)

| Priority | ID | Reason |
|----------|----|--------|
| 1 | DC2 | Docker app won't serve React at all |
| 2 | DF1 | Healthcheck always fails silently |
| 3 | DC1 | Hardcoded password is a security risk |
| 4 | CC1 | Missing `.dockerignore` — Docker builds painfully slow |
| 5 | S1 | `.env` values silently not loaded |
| 6 | P1 / I2 | Double forward pass and contradictory `requires_grad` logic |
| 7 | G1 | GradCAM not thread-safe — stale results under load |
| 8 | TS1 | Tests always fail on CI/fresh clone |
| 9 | AX1 | Axios Content-Type header breaks multipart boundary |
| 10 | C1 / D1 | `.jpg` files excluded from dataset — incomplete training data |
