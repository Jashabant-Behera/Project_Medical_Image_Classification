from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime, timezone
from backend.db.session import Base
 
class PredictionLog(Base):
    __tablename__ = 'prediction_logs'
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    filename = Column(String(255), nullable=False)
    prediction = Column(String(20), nullable=False)
    confidence = Column(Float, nullable=False)
    inference_ms = Column(Integer)
    model_version = Column(String(50), default='1.0')
    image_hash = Column(String(64))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
