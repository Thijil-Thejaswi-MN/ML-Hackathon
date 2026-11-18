"""Projection and geometry helpers.  OWNER: M1.   Contract: README 3.1.

Each trip has its own small road network (a city-sized area), so we use a PER-TRIP local metric
projection: an equirectangular (plate carree) projection centred on the trip centroid, scaled by the
WGS84 radii of curvature at the centre latitude.

Why this is accurate enough: for a point at distance D from the centre, the scale error of a local
tangent-plane projection grows roughly with (D / R_earth)^2 plus a cos(lat) drift term. For trips spanning
a few km (every feature distance is < 1 km) the error is centimetres - far below GPS noise (~5-10 m).
Checked in scripts/00_inventory.py: consecutive-point distances agree with haversine to < 0.3 m. It needs only numpy (no pyproj), so it behaves identically on every laptop.
Never compute distances in raw degrees.
"""
from __future__ import annotations
import numpy as np

_A = 6378137.0              # WGS84 semi-major axis (m)
_E2 = 6.69437999014e-3      # WGS84 first eccentricity squared


class LocalProjector:
    """Per-trip lon/lat (degrees) -> local metres (x east, y north) around (lon0, lat0)."""

    def __init__(self, lon0: float, lat0: float):
        self.lon0 = float(lon0)
        self.lat0 = float(lat0)
        phi = np.radians(self.lat0)
        s2 = np.sin(phi) ** 2
        n = _A / np.sqrt(1 - _E2 * s2)                    # prime-vertical radius of curvature
        m = _A * (1 - _E2) / (1 - _E2 * s2) ** 1.5        # meridional radius of curvature
        self.kx = np.radians(1.0) * n * np.cos(phi)       # metres per degree longitude
        self.ky = np.radians(1.0) * m                     # metres per degree latitude

    @classmethod
    def from_track(cls, track: np.ndarray) -> "LocalProjector":
        lon, lat = track[:, 0], track[:, 1]
        return cls(float(np.median(lon)), float(np.median(lat)))

    def forward(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        lon = np.asarray(lon, dtype=np.float64)
        lat = np.asarray(lat, dtype=np.float64)
        return np.column_stack([(lon - self.lon0) * self.kx, (lat - self.lat0) * self.ky])

    def inverse(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64)
        return np.column_stack([xy[:, 0] / self.kx + self.lon0, xy[:, 1] / self.ky + self.lat0])


def haversine_m(lon1, lat1, lon2, lat2) -> np.ndarray:
    """Great-circle distance (m); used only to validate the projection."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371008.8 * np.arcsin(np.sqrt(a))


def point_to_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Vectorised distance (m) and projection fraction t in [0, 1] from points p to segments a->b.

    p, a, b: (n, 2) arrays (broadcastable). Returns (dist, t, foot) where foot is the closest point.
    """
    ab = b - a
    ap = p - a
    denom = (ab * ab).sum(-1)
    num = (ap * ab).sum(-1)
    t = np.where(denom > 0, num / np.where(denom > 0, denom, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    foot = a + t[..., None] * ab
    d = np.sqrt(((p - foot) ** 2).sum(-1))
    return d, t, foot


def signed_offset(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed perpendicular offset (m): >0 if p is left of a->b, <0 if right."""
    ab = b - a
    ap = p - a
    cross = ab[..., 0] * ap[..., 1] - ab[..., 1] * ap[..., 0]
    n = np.sqrt((ab ** 2).sum(-1))
    return np.where(n > 0, cross / np.where(n > 0, n, 1.0), 0.0)


def bearing(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Bearing (radians, atan2(dx, dy) => 0 = north, clockwise) of vectors a->b in the projected plane."""
    d = np.asarray(b) - np.asarray(a)
    return np.arctan2(d[..., 0], d[..., 1])


def angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Absolute smallest difference between two angles (radians), in [0, pi]."""
    d = np.abs((np.asarray(a) - np.asarray(b) + np.pi) % (2 * np.pi) - np.pi)
    return d


def vehicle_heading(xy: np.ndarray, t: np.ndarray, w: int = 3, min_disp_m: float = 3.0):
    """Heading (rad) from the displacement between points i-w and i+w, speed (m/s), and a 'valid' flag
    (False when the vehicle barely moved, where heading is just noise)."""
    n = len(xy)
    lo = np.clip(np.arange(n) - w, 0, n - 1)
    hi = np.clip(np.arange(n) + w, 0, n - 1)
    disp = xy[hi] - xy[lo]
    dist = np.sqrt((disp ** 2).sum(1))
    dt = np.maximum(t[hi] - t[lo], 1e-6)
    head = np.arctan2(disp[:, 0], disp[:, 1])
    valid = dist >= min_disp_m
    return head, dist / dt, valid
