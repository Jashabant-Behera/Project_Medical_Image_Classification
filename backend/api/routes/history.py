from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from backend.db.session import get_db
from backend.models.prediction import PredictionLog
from backend.utils.response_models import PredictionHistoryResponse
 
router = APIRouter(prefix='/api', tags=['History'])
 
@router.get('/predictions', response_model=PredictionHistoryResponse)
async def get_predictions(limit: int = 50, skip: int = 0,
                           db: Session = Depends(get_db)):
    total = db.query(PredictionLog).count()
    items = (db.query(PredictionLog)
             .order_by(PredictionLog.created_at.desc())
             .offset(skip).limit(limit).all())
    return PredictionHistoryResponse(total=total, items=items)
