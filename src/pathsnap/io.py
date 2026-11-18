"""Raw-file I/O.  OWNER: M1.   Contract: README 3.1.

Raw files per trip: <id>.track (lon, lat, t) | <id>.nodes (lon, lat) | <id>.arcs (from, to) | <id>.route (arc ids).
Whitespace separated, no headers, CRLF line endings. Trip ids are STRINGS (keep leading zeros).

IMPORTANT (README R2): load_trip() never reads the route. Only labels.py and api/evaluator.py call load_route().
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class TripData:
    trip_id: str
    split: str
    track: np.ndarray   # (n_obs, 3) float64: lon, lat, t_seconds   (row index == obs_idx)
    nodes: np.ndarray   # (n_nodes, 2) float64: lon, lat           (row index == node_id)
    arcs: np.ndarray    # (n_arcs, 2) int64: from_node, to_node    (row index == arc_id)

    @property
    def n_obs(self) -> int:
        return int(self.track.shape[0])


def trip_dir(root: str | Path, split: str, trip_id: str) -> Path:
    return Path(root) / split / trip_id


def list_trip_ids(root: str | Path, split: str) -> list[str]:
    """Sorted trip ids found on disk (never hardcode ids anywhere)."""
    base = Path(root) / split
    return sorted(p.name for p in base.iterdir() if p.is_dir())


def load_trip(root: str | Path, split: str, trip_id: str) -> TripData:
    """Load GPS + road network ONLY (no route / ground truth)."""
    d = trip_dir(root, split, trip_id)
    track = np.loadtxt(d / f"{trip_id}.track", ndmin=2)
    nodes = np.loadtxt(d / f"{trip_id}.nodes", ndmin=2)
    arcs = np.loadtxt(d / f"{trip_id}.arcs", dtype=np.int64, ndmin=2)
    assert track.shape[1] == 3 and nodes.shape[1] == 2 and arcs.shape[1] == 2, "unexpected column count"
    # coordinate-order sanity check (lon, lat)
    assert np.all(np.abs(track[:, 0]) <= 180) and np.all(np.abs(track[:, 1]) <= 90), "lon/lat out of range"
    return TripData(trip_id, split, track, nodes, arcs)


def load_route(root: str | Path, split: str, trip_id: str) -> np.ndarray:
    """Reference route (arc ids). ALLOWED CALLERS: labels.py, api/evaluator.py, scripts/dataset building only."""
    d = trip_dir(root, split, trip_id)
    return np.loadtxt(d / f"{trip_id}.route", dtype=np.int64, ndmin=1)
