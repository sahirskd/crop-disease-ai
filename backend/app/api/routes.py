import uuid
from datetime import datetime
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from PIL import Image
import io
from app.services.predictor import CropDiseasePredictor
from app.core.config import settings
from app.models.schemas import PredictionResponse, PredictionRecord
from app.services.database import get_db, save_prediction, get_history
from sqlalchemy.orm import Session

router = APIRouter()

# Singleton predictor (loaded once on startup)
_predictor: CropDiseasePredictor = None


def get_predictor() -> CropDiseasePredictor:
    global _predictor
    if _predictor is None:
        _predictor = CropDiseasePredictor(
            model_path=settings.MODEL_PATH,
            metadata_path=settings.METADATA_PATH,
            class_names_path=settings.CLASS_NAMES_PATH,
        )
    return _predictor


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    predictor: CropDiseasePredictor = Depends(get_predictor),
):
    """Upload a leaf image and receive disease prediction + Grad-CAM heatmap."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=400, detail="Image too large (max 10MB)")

    try:
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    result = predictor.predict(image)
    prediction_id = str(uuid.uuid4())

    # Persist to DB (without heatmap blob for storage efficiency)
    save_prediction(db, {
        "id": prediction_id,
        "filename": file.filename,
        "plant": result["plant"],
        "disease": result["disease"],
        "confidence": result["confidence"],
        "is_healthy": result["is_healthy"],
        "created_at": datetime.utcnow(),
    })

    return {**result, "id": prediction_id}


@router.get("/diseases")
async def list_diseases(predictor: CropDiseasePredictor = Depends(get_predictor)):
    """Return all 38 supported plant/disease classes."""
    classes = predictor.class_names
    parsed = []
    for cls in classes:
        parts = cls.split("___")
        parsed.append({
            "class": cls,
            "plant": parts[0].replace("_", " "),
            "disease": parts[1].replace("_", " ") if len(parts) > 1 else "Unknown",
        })
    return {"total": len(parsed), "diseases": parsed}


@router.get("/history", response_model=list[PredictionRecord])
async def history(limit: int = 20, db: Session = Depends(get_db)):
    """Return recent prediction history."""
    return get_history(db, limit=limit)
