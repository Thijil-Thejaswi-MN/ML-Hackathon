"""Pydantic request/response models.  OWNER: M4.   Contract: README 3.13 [FROZEN]."""
from typing import Dict, List, Optional
from pydantic import BaseModel


class PredictRequest(BaseModel):
    trip_id: str


class Metrics(BaseModel):
    top1_accuracy_hmm: float
    top1_accuracy_classifier: float
    route_continuity_hmm: float
    route_continuity_classifier: float
    sequence_f1: float
    candidate_miss_rate: float


class PredictResponse(BaseModel):
    trip_id: str
    n_observations: int
    gps: List[List[float]]                 # [lat, lon, t]
    predicted_route: List[List[float]]     # [lat, lon]
    classifier_route: Optional[List[List[float]]] = None
    predicted_arcs: List[int]
    metrics: Metrics
    timings_ms: Dict[str, float]
    # NOTE: no reference route / labels field exists on purpose (README R2)
