"""Candidate generation.  OWNER: M1.   Contract: README 3.4.

Uses ONLY track/nodes/arcs. Adaptive radius, top-K, spatial index (no O(n_obs * n_arcs) loops).
"""
from __future__ import annotations
import pandas as pd


def generate_candidates(trip, graph, projector, cfg: dict) -> pd.DataFrame:
    """Return DataFrame[trip_id, obs_idx, arc_id, rank, dist_m, ...]. Observations with zero candidates
    must still be recoverable (e.g. via n_obs and the set of obs_idx present)."""
    raise NotImplementedError
