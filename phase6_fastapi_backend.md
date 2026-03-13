# Phase 6: FastAPI Backend — REST API, Database & Inference Service

## Objective
Convert all Python ML logic scripts into an asynchronous API microservice utilizing FastAPI exposing inference modules. Dynamically map endpoints into an SQLAlchemy ORM schema persisting image inference requests natively over SQLite (`prediction_logs`), and utilizing Alembic for systemic migrations automatically.

## Files Created
1. `backend/config/settings.py` - Pydantic base configuration mappings routing `.env` keys (`DATABASE_URL`, `MODEL_PATH`, `MAX_MB`). Defaults to standard `sqlite` if environment strings fail natively dynamically.
2. `backend/db/session.py` - SQLAlchemy session engines pooling configurations and threading setups.
3. `backend/models/prediction.py` - Schema representing the `prediction_logs` table (hashes, names, confidences).
4. `backend/db/migrations/env.py` - Core Alembic schema target settings triggering `alembic init` & `alembic upgrade head` dynamically.
5. `backend/services/inference.py` - Pre-boots the DenseNet Tensor graph loading PyTorch `best_model.pth` metadata uniquely as a thread-safe Singleton memory structure once on launch, bypassing 400ms sequential disc read latencies per-call organically.
6. `backend/services/preprocessor.py` & `gradcam.py` - Microservices interfacing binary IO conversions & overlay structures.
7. `backend/api/routes/` - Restful endpoint collections defining `/predict`, `/health`, and `/history` mapping response sequences natively (`backend/utils/response_models.py`).
8. `backend/main.py` - Uvicorn root lifespan event application, connecting endpoints seamlessly and building automated documentation routes `http://localhost:8000/docs`.
9. `backend/tests/test_predict.py` - Pytest validation files querying endpoints mock logic asserting exact schema matching dynamically (`200 OK -> Prediction -> >= 0.0 confidence`).
