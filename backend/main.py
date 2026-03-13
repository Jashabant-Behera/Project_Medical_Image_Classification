import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from backend.db.session import Base, engine
from backend.services.inference import inference_service
from backend.api.routes import predict, health, history
from backend.config.settings import settings
from contextlib import asynccontextmanager
 
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info('Creating database tables...')
    Base.metadata.create_all(bind=engine)
    logger.info('Loading ML model...')
    inference_service.load_model()
    logger.info('Server ready!')
    yield
 
app = FastAPI(
    title='Chest X-Ray Pneumonia Classifier',
    description='AI-powered pneumonia detection from chest X-ray images',
    version='1.0.0',
    lifespan=lifespan
)
 
app.add_middleware(CORSMiddleware,
    allow_origins=['*'],   # Restrict in production
    allow_methods=['*'],
    allow_headers=['*'])
 
app.include_router(predict.router)
app.include_router(health.router)
app.include_router(history.router)
 
@app.get('/')
async def root(): return {'message': 'Chest X-Ray Classifier API', 'docs': '/docs'}