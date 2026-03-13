from pydantic_settings import BaseSettings
from pathlib import Path
import os
 
class Settings(BaseSettings):
    # Defaulting to sqlite if not set to ensure immediate out-of-the-box Windows local execution
    DATABASE_URL: str = os.getenv('DATABASE_URL', 'sqlite:///./chest_xray_db.sqlite')
    MODEL_PATH: str = os.getenv('MODEL_PATH', 'ml/saved_models/best_model_densenet121.pth')
    MODEL_NAME: str = os.getenv('MODEL_NAME', 'densenet121')
    MODEL_VERSION: str = os.getenv('MODEL_VERSION', '1.0')
    MAX_UPLOAD_MB: int = int(os.getenv('MAX_UPLOAD_MB', '16'))
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    SECRET_KEY: str = os.getenv('SECRET_KEY', 'change-me-in-production')
 
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        extra = 'ignore'
 
settings = Settings()
