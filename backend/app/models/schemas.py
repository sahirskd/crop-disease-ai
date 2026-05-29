from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class TopPrediction(BaseModel):
    class_: str
    confidence: float

    class Config:
        populate_by_name = True
        fields = {"class_": "class"}


class Treatment(BaseModel):
    severity: str
    description: str
    actions: List[str]


class PredictionResponse(BaseModel):
    id: str
    plant: str
    disease: str
    predicted_class: str
    confidence: float
    is_healthy: bool
    top_predictions: List[dict]
    gradcam_heatmap: str  # base64 encoded PNG
    treatment: Treatment
    needs_expert_review: bool


class PredictionRecord(BaseModel):
    id: str
    filename: Optional[str]
    plant: str
    disease: str
    confidence: float
    is_healthy: bool
    created_at: datetime

    class Config:
        from_attributes = True
