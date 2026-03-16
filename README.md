# Chest X-Ray Pneumonia Classifier

An end-to-end AI platform for detecting **Pneumonia from chest X-ray images** using a fine-tuned DenseNet121 deep learning model, a FastAPI backend, and a React + Vite frontend. Includes **Grad-CAM visual explainability** to highlight the regions of the X-ray that drove the prediction.

---

## Demo Flow

```
User uploads X-ray → FastAPI preprocesses → DenseNet121 predicts
→ Grad-CAM generates heatmap → Result + overlay returned to browser
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | PyTorch + Torchvision (DenseNet121) |
| Explainability | Grad-CAM (manual hook implementation) |
| Backend | FastAPI + Uvicorn |
| Database | SQLAlchemy + Alembic + SQLite (dev) / PostgreSQL (prod) |
| Image Processing | Pillow + OpenCV-headless |
| Frontend | React 18 + Vite + Tailwind CSS |
| Containerisation | Docker + Docker Compose + Nginx |

---

## Project Structure

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
│   ├── reports/                          ← Auto-generated (gitignored output)
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
│   │   └── favicon.svg
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

---

## Local Setup

### 1. Clone and create virtual environment

```bash
git clone <repo-url>
cd Project_Medical_Image_Classification

# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Download the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle and extract it so the folder structure matches:

```
ml/data/raw/
├── train/  NORMAL/  PNEUMONIA/
├── val/    NORMAL/  PNEUMONIA/
└── test/   NORMAL/  PNEUMONIA/
```

### 4. Generate CSV manifests

```bash
python ml/data/create_csv.py
```

This scans `ml/data/raw/` and writes `train.csv`, `val.csv`, and `test.csv` into `ml/data/`.

### 5. Train the model

```bash
# Local CPU (slow — see COLAB_TRAINING_GUIDE.md for GPU training)
python ml/training/train.py \
  --model densenet121 \
  --epochs 20 \
  --lr 0.0001 \
  --batch_size 32 \
  --patience 5 \
  --data_dir ml/data \
  --save_dir ml/saved_models
```

Trained weights are saved to `ml/saved_models/best_model_densenet121.pth`.

### 6. Run the backend

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at [http://localhost:8000/docs](http://localhost:8000/docs)

### 7. Run the frontend

```bash
cd frontend
npm install
npm run dev
```

App available at [http://localhost:5173](http://localhost:5173)

---

## Docker Deployment

```bash
# Copy and configure environment
cp .env.example .env

# Build and start all services
docker-compose up --build -d

# Check logs
docker-compose logs -f backend
```

App served at [http://localhost](http://localhost) via Nginx.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/predict` | Upload X-ray image → prediction + Grad-CAM |
| `GET` | `/api/health` | Model load status and device info |
| `GET` | `/api/predictions` | Paginated prediction history |
| `GET` | `/docs` | Swagger UI |

### Example request

```bash
curl -X POST http://localhost:8000/api/predict \
  -F "file=@chest_xray.jpeg"
```

### Example response

```json
{
  "prediction": "PNEUMONIA",
  "confidence": 0.9731,
  "gradcam_image": "data:image/png;base64,...",
  "inference_ms": 312,
  "model_version": "1.0",
  "timestamp": "2026-03-16T10:42:00Z"
}
```

---

## Database Migrations

```bash
# Apply migrations
cd backend
alembic upgrade head

# Create a new migration after model changes
alembic revision --autogenerate -m "description"
```

---

## Running Tests

```bash
pytest backend/tests/
```

---

## Training on Google Colab (Recommended)

Local CPU training takes ~4–6 hours for 20 epochs. Google Colab T4 GPU reduces this to ~25–30 minutes.

See **[COLAB_TRAINING_GUIDE.md](./COLAB_TRAINING_GUIDE.md)** for the full step-by-step guide including setup, training commands, and model download instructions.

---

## Dataset

See **[dataset.md](./dataset.md)** for full dataset structure, class counts, and notes on the class imbalance (3,875 PNEUMONIA vs 1,341 NORMAL in training).

| Split | NORMAL | PNEUMONIA | Total |
|---|---|---|---|
| Train | 1,341 | 3,875 | 5,216 |
| Val | 8 | 8 | 16 |
| Test | 234 | 390 | 624 |

---

## Expected Model Performance

After 20 epochs on DenseNet121 (T4 GPU):

| Metric | Expected |
|---|---|
| Test Accuracy | ~90–93% |
| Test ROC-AUC | ~97–98% |
| Pneumonia Recall | ~95–98% |

> **Note:** Pneumonia recall (sensitivity) is the critical metric — a missed positive (false negative) is clinically more dangerous than a false alarm.

---

## Environment Variables

Copy `.env.example` to `.env` and configure:

```env
DATABASE_URL=postgresql://postgres:password@localhost:5432/chest_xray_db
MODEL_PATH=ml/saved_models/best_model_densenet121.pth
MODEL_NAME=densenet121
MODEL_VERSION=1.0
MAX_UPLOAD_MB=16
LOG_LEVEL=INFO
SECRET_KEY=your-secret-key-here
```

Leave `DATABASE_URL` unset to default to SQLite for local development.