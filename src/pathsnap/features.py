"""Feature engineering.  OWNER: M1.   Contract: README 3.5.

build_features() is used IDENTICALLY offline (dataset builder) and online (pipeline.predict_trip) -> no train/serve skew.
Column order is saved to artifacts/features.json.
"""
from __future__ import annotations
import pandas as pd

FEATURE_COLUMNS: list[str] = []   # M1: fill and keep in sync with artifacts/features.json


def build_features(trip, graph, projector, candidates: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Return candidates + FEATURE_COLUMNS (float32). Must not use labels or the route."""
    raise NotImplementedError
