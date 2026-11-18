"""Road-network structures.  OWNER: M3 (M1 may add helpers via PR).

Needed by candidates.py (spatial index), features.py (arc geometry, degrees) and decoding/ (shortest paths).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class RoadGraph:
    """Built once per trip from TripData; derived from inputs only (safe to cache per trip)."""
    xy_nodes: np.ndarray       # (n_nodes, 2) metres
    arcs: np.ndarray           # (n_arcs, 2) from/to node ids
    arc_len: np.ndarray        # (n_arcs,) metres
    arc_bearing: np.ndarray    # (n_arcs,) radians
    # TODO(M3): sparse adjacency (node graph with arc-length weights), reverse-twin map, node->arcs map


def build_graph(trip, projector) -> RoadGraph:
    raise NotImplementedError


def bounded_shortest_paths(graph: RoadGraph, sources, cutoff_m: float):
    """scipy.sparse.csgraph.dijkstra(..., limit=cutoff_m) with memoisation. Used by the HMM transition model."""
    raise NotImplementedError
