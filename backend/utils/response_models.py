from pydantic import BaseModel
from datetime import datetime
from typing import Optional
 
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    gradcam_image: str            # base64 PNG
    inference_ms: int
    model_version: str
    timestamp: datetime
 
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    device: str
 
class PredictionHistoryItem(BaseModel):
    id: int
    filename: str
    prediction: str
    confidence: float
    inference_ms: Optional[int]
    created_at: datetime

    class Config: 
        from_attributes = True
 
class PredictionHistoryResponse(BaseModel):
    total: int
    items: list[PredictionHistoryItem]
