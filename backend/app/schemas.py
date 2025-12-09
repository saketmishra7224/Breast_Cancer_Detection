from typing import List, Dict
from pydantic import BaseModel, Field


class FeatureVector(BaseModel):
    features: Dict[str, float] = Field(..., description="Feature name to value map")


class BatchFeatures(BaseModel):
    rows: List[FeatureVector]


class PredictionResponse(BaseModel):
    prediction: int
    probability_benign: float
    probability_malignant: float


class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]


class MetricItem(BaseModel):
    model: str
    accuracy: float
    f1: float
    roc_auc: float
    cv_mean_roc_auc: float
    cv_std_roc_auc: float
