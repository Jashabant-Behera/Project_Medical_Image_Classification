import torch
from fastapi import APIRouter
from backend.services.inference import inference_service
from backend.utils.response_models import HealthResponse
 
router = APIRouter(prefix='/api', tags=['Health'])
 
@router.get('/health', response_model=HealthResponse)
async def health():
    device = str(inference_service.device) if inference_service._loaded else 'not loaded'
    model_name = inference_service.model.__class__.__name__ if inference_service._loaded else 'none'
    return HealthResponse(status='ok', model_loaded=inference_service._loaded,
                          model_name=model_name, device=device)
