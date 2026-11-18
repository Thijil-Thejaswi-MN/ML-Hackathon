"""HMM + Viterbi decoder.  OWNER: M3.   Contract: README 3.10.

States = candidate arcs per observation (arc ids preserved). Emission = classifier probability. Transition = network connectivity.
Must handle: no candidates, time gaps (chain breaks), unreachable transitions. Return ONE arc per observation.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class DecodeResult:
    arc_per_obs: np.ndarray                 # (n_obs,)
    matched: np.ndarray                     # (n_obs,) bool
    segments: list = field(default_factory=list)
    route_arcs: list = field(default_factory=list)
    route_polyline: list = field(default_factory=list)   # [[lat, lon], ...]
    breaks: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)


def decode(cands, trip, graph, projector, cfg: dict) -> DecodeResult:
    """cands: DataFrame[trip_id, obs_idx, arc_id, p, ...] (p = classifier probability)."""
    raise NotImplementedError
