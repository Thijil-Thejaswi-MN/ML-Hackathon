"""Derive per-observation reference arcs from the .route.  OWNER: M1.   Contract: README 3.3.

ALLOWED to read the route (with api/evaluator.py). Never imported by pipeline.py / decoding / features / candidates.
"""
from __future__ import annotations
import numpy as np


def derive_obs_labels(trip, route: np.ndarray, projector) -> tuple[np.ndarray, np.ndarray]:
    """Return (true_arc_id[n_obs], confidence[n_obs]).

    Suggested: monotone alignment (DP / Viterbi over route index, non-decreasing) with point-to-segment
    cost (+ optional heading term). Document limitations in docs/DATA_NOTES.md.
    """
    raise NotImplementedError
