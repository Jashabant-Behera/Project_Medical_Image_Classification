import io, time, hashlib
from datetime import datetime, timezone
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from PIL import Image
from backend.db.session import get_db
from backend.models.prediction import PredictionLog
from backend.services.inference import inference_service
from backend.services.gradcam import GradCAMService
from backend.services.preprocessor import ImagePreprocessor
from backend.utils.response_models import PredictionResponse
from backend.config.settings import settings
import torch
 
router = APIRouter(prefix='/api', tags=['Prediction'])
preprocessor = ImagePreprocessor()
gradcam_service = None   # Initialized after model loads
 
ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/jpg']
 
@router.post('/predict', response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    global gradcam_service
    # Validation
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, detail='Only JPEG/PNG files are accepted')
    contents = await file.read()
    if len(contents) > settings.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(400, detail=f'File exceeds {settings.MAX_UPLOAD_MB}MB limit')
    # Decode image
    try:
        image = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception:
        raise HTTPException(400, detail='Could not decode image')
    
    # Preprocess
    tensor = preprocessor.process(image)
    tensor.requires_grad = True # Required for GradCAM
    
    # Inference
    t0 = time.time()
    label, confidence, class_idx = inference_service.predict(tensor)
    inference_ms = int((time.time() - t0) * 1000)
    
    # Grad-CAM
    if gradcam_service is None:
        gradcam_service = GradCAMService(inference_service.model)
    overlay = gradcam_service.generate_overlay(tensor, image, class_idx)
    gradcam_b64 = gradcam_service.overlay_to_base64(overlay)
    
    # Log to DB
    img_hash = hashlib.md5(contents).hexdigest()
    db.add(PredictionLog(filename=file.filename, prediction=label,
                         confidence=confidence, inference_ms=inference_ms,
                         model_version=settings.MODEL_VERSION, image_hash=img_hash))
    db.commit()
    
    return PredictionResponse(prediction=label, confidence=round(confidence, 4),
                               gradcam_image=gradcam_b64, inference_ms=inference_ms,
                               model_version=settings.MODEL_VERSION,
                               timestamp=datetime.now(timezone.utc))
