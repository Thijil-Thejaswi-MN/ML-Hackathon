"""Projection and geometry helpers.  OWNER: M1.   Contract: README 3.1.

Trips are scattered worldwide and each has its own network -> use a PER-TRIP local metric projection.
"""
from __future__ import annotations
import numpy as np


class LocalProjector:
    """Per-trip lon/lat -> metres (x, y). Justify the CRS choice in the report."""
    def __init__(self, lon0: float, lat0: float):
        raise NotImplementedError("M1: e.g. pyproj azimuthal-equidistant or trip-centroid UTM")

    @classmethod
    def from_track(cls, track: np.ndarray) -> "LocalProjector":
        raise NotImplementedError

    def forward(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:  # -> (n, 2) metres
        raise NotImplementedError

    def inverse(self, xy: np.ndarray) -> np.ndarray:                    # -> (n, 2) lon, lat
        raise NotImplementedError


def point_to_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Vectorised distance (m) and projection fraction in [0, 1] from points p to segments a->b."""
    raise NotImplementedError


def bearing(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Bearing (radians) of vectors a->b in the projected plane."""
    raise NotImplementedError
