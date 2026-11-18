"""End-to-end inference.  OWNER: M4.   Contract: README 3.10/3.13 [FROZEN signature].

predict_trip() sees ONLY TripData (track/nodes/arcs). It has no access to any route or label (README R2).
Used by the API and by offline evaluation scripts so numbers match what is demonstrated.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class PredictionResult:
    trip_id: str
    arc_per_obs: np.ndarray            # HMM-decoded arc for every observation (n_obs,)
    classifier_arc_per_obs: np.ndarray  # classifier-only argmax arc per observation (-1 if none)
    matched: np.ndarray                # bool (n_obs,)
    route_arcs: list = field(default_factory=list)
    route_polyline: list = field(default_factory=list)       # [[lat, lon], ...]
    classifier_polyline: list = field(default_factory=list)
    breaks: list = field(default_factory=list)
    timings_ms: dict = field(default_factory=dict)


def predict_trip(trip, bundle) -> PredictionResult:
    """load graph -> candidates -> features -> model proba (-> calibrate) -> HMM/Viterbi -> route. Time each stage."""
    raise NotImplementedError
