# Medical Image Classification (Chest X-Ray)

An end-to-end AI platform built to detect and classify Pneumonia from Chest X-Ray images utilizing a Deep Learning pipeline, a Python FastAPI backend, and a modern React Vite frontend UI.

## Overview
This project takes raw JPEG/PNG scans of Chest X-Ray images and runs them through a sophisticated deep learning computer vision model built in PyTorch (Transfer Learning via **DenseNet121**). It not only provides binary structural analysis (`NORMAL` vs `PNEUMONIA`), but also features a **Grad-CAM** visual explainability layer highlighting the exact regions of the chest that triggered the neural networking conclusions.

## Technology Stack
*   **Machine Learning**: PyTorch, Torchvision (DenseNet121), Scikit-Learn
*   **Backend Application**: FastAPI, Uvicorn, SQLAlchemy, Alembic, SQLite (or PostgreSQL)
*   **Frontend Interface**: React 18, Vite, React-Router-DOM, Tailwind CSS (via PostCSS)
*   **Deployment**: Docker, Docker Compose, NGINX

## Architecture & Project Phases
The project construction was broken down into 8 sequential developmental phases. You can read the detailed implementations of each phase in the `.md` files residing in the root folder:

1.  [`phase1_dataset_eda.md`](./phase1_dataset_eda.md) - Dataset Acquisition & Exploratory Data Analysis
2.  [`phase2_data_pipeline.md`](./phase2_data_pipeline.md) - Dataset, Augmentations & DataLoaders
3.  [`phase3_model_development.md`](./phase3_model_development.md) - Transfer Learning with DenseNet121
4.  [`phase4_training_loop.md`](./phase4_training_loop.md) - Full Training Loop, Evaluation & Metrics
5.  [`phase5_gradcam.md`](./phase5_gradcam.md) - Grad-CAM Visual Explainability
6.  [`phase6_fastapi_backend.md`](./phase6_fastapi_backend.md) - REST API, Database & Inference Singletons
7.  [`phase7_react_frontend.md`](./phase7_react_frontend.md) - Upload UI, Results & Grad-CAM Viewer
8.  [`phase8_dockerization.md`](./phase8_dockerization.md) - Containerization & Orchestration

## How to Run Locally

### 1. Model Baseline (Dependencies)
Ensure that you have activated the virtual environment:
```bash
# Windows
.\venv\Scripts\Activate.ps1
# Unix/Mac
source ./venv/bin/activate
```

### 2. Start the Backend API (FastAPI)
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
*The API Interactive documentation will be fully reachable dynamically at [http://localhost:8000/docs](http://localhost:8000/docs).*

### 3. Start the Frontend Application (Vite/React)
In an entirely separate terminal window:
```bash
cd frontend
npm run dev
```
*The dynamic UI will be accessible locally via [http://localhost:5173/](http://localhost:5173/).*