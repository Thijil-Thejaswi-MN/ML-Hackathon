"""Candidate generation.  OWNER: M1.   Contract: README 3.4.

Uses ONLY track/nodes/arcs. Adaptive radius, top-K, spatial index (no O(n_obs * n_arcs) loops).

How it works
------------
1. Every arc is densified into sample points spaced <= `sample_step_m` apart; a cKDTree indexes them.
   If an arc passes within r of a GPS point, one of its samples is within r + step/2, so a ball query with
   that radius finds every arc within r (no misses from long arcs, unlike a midpoint-only index).
2. Exact point-to-segment distance is computed for the arcs found; arcs farther than r are dropped.
3. Adaptive radius: try radii_m in increasing order; an observation keeps the first radius that gives
   >= k_min candidates (or the largest radius if none does).
4. Keep the k_max nearest arcs per observation, rank 0 = nearest.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .geo import point_to_segment

DEFAULT_CAND_CFG = {"radii_m": [50, 100, 200], "k_max": 30, "k_min": 3, "sample_step_m": 10.0}


class ArcIndex:
    """KD-tree over densified arc samples. Build once per trip (derived from the network only)."""

    def __init__(self, graph, step_m: float = 10.0):
        a, b = graph.a_xy, graph.b_xy
        n_seg = np.maximum(1, np.ceil(graph.arc_len / step_m).astype(np.int64))
        arc_of = np.repeat(np.arange(graph.n_arcs), n_seg + 1)
        # fraction 0..1 within each arc
        starts = np.concatenate([[0], np.cumsum(n_seg + 1)[:-1]])
        local = np.arange(arc_of.size) - np.repeat(starts, n_seg + 1)
        frac = local / np.repeat(n_seg, n_seg + 1)
        pts = a[arc_of] + frac[:, None] * (b[arc_of] - a[arc_of])
        self.tree = cKDTree(pts)
        self.arc_of = arc_of
        self.step = float(step_m)
        self.graph = graph

    def query(self, xy: np.ndarray, r: float):
        """All (obs, arc, dist, t) pairs with exact distance <= r."""
        hits = self.tree.query_ball_point(xy, r + self.step / 2 + 1e-6)
        lens = np.fromiter((len(h) for h in hits), dtype=np.int64, count=len(hits))
        if lens.sum() == 0:
            e = np.empty(0, dtype=np.int64)
            return e, e, np.empty(0), np.empty(0)
        obs = np.repeat(np.arange(len(hits)), lens)
        arc = self.arc_of[np.concatenate([np.asarray(h, dtype=np.int64) for h in hits if len(h)])]
        # unique (obs, arc)
        key = np.unique(obs * self.graph.n_arcs + arc)
        obs, arc = key // self.graph.n_arcs, key % self.graph.n_arcs
        d, t, _ = point_to_segment(xy[obs], self.graph.a_xy[arc], self.graph.b_xy[arc])
        keep = d <= r
        return obs[keep], arc[keep], d[keep], t[keep]


def generate_candidates(trip, graph, projector, cfg: dict, index: ArcIndex | None = None) -> pd.DataFrame:
    """Return DataFrame[trip_id, obs_idx, arc_id, rank, dist_m, proj_t, radius_m].

    Observations with zero candidates are absent from the frame; downstream code recovers them from
    trip.n_obs (see n_candidates_per_obs()).
    """
    cfg = {**DEFAULT_CAND_CFG, **(cfg or {})}
    index = index or ArcIndex(graph, cfg["sample_step_m"])
    xy = projector.forward(trip.track[:, 0], trip.track[:, 1])
    n = len(xy)
    remaining = np.arange(n)
    parts = []
    radii = sorted(cfg["radii_m"])
    for k, r in enumerate(radii):
        if remaining.size == 0:
            break
        o, a, d, t = index.query(xy[remaining], r)
        o = remaining[o]
        cnt = np.bincount(o, minlength=n)[remaining]
        last = k == len(radii) - 1
        ok = remaining[(cnt >= cfg["k_min"]) | last]
        sel = np.isin(o, ok)
        parts.append(pd.DataFrame({"obs_idx": o[sel], "arc_id": a[sel], "dist_m": d[sel],
                                   "proj_t": t[sel], "radius_m": float(r)}))
        remaining = np.setdiff1d(remaining, ok)
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["obs_idx", "arc_id", "dist_m", "proj_t", "radius_m"])
    df = df.sort_values(["obs_idx", "dist_m", "arc_id"], kind="mergesort")
    df["rank"] = df.groupby("obs_idx").cumcount()
    df = df[df["rank"] < cfg["k_max"]].reset_index(drop=True)
    df.insert(0, "trip_id", trip.trip_id)
    df = df.astype({"obs_idx": np.int32, "arc_id": np.int32, "rank": np.int16,
                    "dist_m": np.float32, "proj_t": np.float32, "radius_m": np.float32})
    return df


def n_candidates_per_obs(cands: pd.DataFrame, n_obs: int) -> np.ndarray:
    return np.bincount(cands["obs_idx"].to_numpy(), minlength=n_obs)
