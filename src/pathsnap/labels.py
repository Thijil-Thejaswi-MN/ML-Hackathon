"""Derive per-observation reference arcs from the .route.  OWNER: M1.   Contract: README 3.3.

ALLOWED to read the route (with api/evaluator.py). Never imported by pipeline.py / decoding / features / candidates.

Method - monotone alignment by dynamic programming
--------------------------------------------------
The .route is the ordered list of arcs the vehicle drove; GPS points progress along it in time. So the label
of point i is route[j_i] where j_0 <= j_1 <= ... <= j_{n-1} (non-decreasing route position). This matters
because routes revisit places (loops, U-turns, two-way roads), where "nearest route arc" picks the wrong pass.

Cost of assigning point i to route position j:
    c(i, j) = dist_m(p_i, arc_j) + w_head * (1 - cos(vehicle_heading_i - arc_bearing_j)) / 2   (heading only when moving)
Transition from position k (point i-1) to j (point i):
    stay  (j == k):  0
    move  (j >  k):  beta + gamma * (length of route arcs strictly between k and j)
The skip term lets the alignment jump over short arcs passed between two GPS fixes (or during a gap) but
discourages jumping far along the route. Solved exactly in O(n_obs * n_route) with a prefix-minimum trick.

Limitations (see docs/DATA_NOTES.md): labels are derived, not verified truth. Points recorded before the route
starts / after it ends, systematic GPS offsets, and parallel roads can produce wrong or low-confidence labels.
"""
from __future__ import annotations
import numpy as np

from .geo import point_to_segment, bearing, vehicle_heading

DEFAULT_LABEL_CFG = {"w_head": 15.0, "beta": 2.0, "gamma": 0.05, "moving_mps": 2.0, "conf_sigma_m": 20.0}


def align_to_route(xy_obs, t_obs, a_xy, b_xy, arc_len, arc_bear, cfg=None):
    """Core DP. Returns (route_pos[n_obs], dist_m[n_obs])."""
    cfg = {**DEFAULT_LABEL_CFG, **(cfg or {})}
    n, r = len(xy_obs), len(a_xy)
    head, speed, hvalid = vehicle_heading(xy_obs, t_obs)
    use_head = hvalid & (speed >= cfg["moving_mps"])
    cos_b, sin_b = np.cos(arc_bear), np.sin(arc_bear)
    # L[j] = cumulative length of route arcs 0..j ; skipped length between k and j = L[j-1] - L[k]
    L = np.cumsum(arc_len)
    Lm1 = np.concatenate([[0.0], L[:-1]])
    beta, gamma = cfg["beta"], cfg["gamma"]
    back = np.empty((n, r), dtype=np.int16 if r < 32767 else np.int32)

    def cost(i):
        d, _, _ = point_to_segment(np.broadcast_to(xy_obs[i], (r, 2)), a_xy, b_xy)
        if use_head[i]:
            c = np.cos(head[i]) * cos_b + np.sin(head[i]) * sin_b   # cos(head - bearing)
            d = d + cfg["w_head"] * (1 - c) / 2
        return d

    D = cost(0)
    back[0] = np.arange(r)
    idx = np.arange(r)
    for i in range(1, n):
        # best predecessor k < j for a move: min_k (D[k] - gamma*L[k]) + beta + gamma*L[j-1]
        g = D - gamma * L
        pm = np.minimum.accumulate(g)
        am = _argmin_accumulate(g, pm)
        move = np.full(r, np.inf)
        move[1:] = pm[:-1] + beta + gamma * Lm1[1:]
        move_from = np.empty(r, dtype=np.int64)
        move_from[0] = 0
        move_from[1:] = am[:-1]
        stay_better = D <= move
        best = np.where(stay_better, D, move)
        back[i] = np.where(stay_better, idx, move_from)
        D = best + cost(i)
    # backtrack
    pos = np.empty(n, dtype=np.int64)
    pos[-1] = int(np.argmin(D))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]
    dist, _, _ = point_to_segment(xy_obs, a_xy[pos], b_xy[pos])
    return pos, dist


def _argmin_accumulate(g: np.ndarray, pm: np.ndarray) -> np.ndarray:
    """Index of the running minimum (first occurrence) for each prefix."""
    is_new = np.empty(len(g), dtype=bool)
    is_new[0] = True
    is_new[1:] = g[1:] < pm[:-1]
    idx = np.where(is_new, np.arange(len(g)), 0)
    return np.maximum.accumulate(idx)


def derive_obs_labels(trip, route: np.ndarray, projector, graph=None, cfg: dict | None = None):
    """Return (true_arc_id[n_obs], confidence[n_obs]) ; confidence = exp(-0.5 (d / sigma)^2) in (0, 1].

    Use derive_obs_labels_full() to also get the distance (m) and route position.
    """
    arc, conf, _, _ = derive_obs_labels_full(trip, route, projector, graph, cfg)
    return arc, conf


def derive_obs_labels_full(trip, route, projector, graph=None, cfg=None):
    cfg = {**DEFAULT_LABEL_CFG, **(cfg or {})}
    route = np.asarray(route, dtype=np.int64)
    xy_obs = projector.forward(trip.track[:, 0], trip.track[:, 1])
    xy_nodes = graph.xy_nodes if graph is not None else projector.forward(trip.nodes[:, 0], trip.nodes[:, 1])
    a = xy_nodes[trip.arcs[route, 0]]
    b = xy_nodes[trip.arcs[route, 1]]
    alen = np.sqrt(((b - a) ** 2).sum(1))
    abear = bearing(a, b)
    pos, dist = align_to_route(xy_obs, trip.track[:, 2], a, b, alen, abear, cfg)
    conf = np.exp(-0.5 * (dist / cfg["conf_sigma_m"]) ** 2)
    return route[pos], conf, dist, pos


def nearest_route_arc(trip, route, projector):
    """Naive baseline label: nearest route arc, ignoring order. Used only to report label agreement."""
    route = np.asarray(route, dtype=np.int64)
    xy_obs = projector.forward(trip.track[:, 0], trip.track[:, 1])
    xy_nodes = projector.forward(trip.nodes[:, 0], trip.nodes[:, 1])
    a = xy_nodes[trip.arcs[route, 0]]
    b = xy_nodes[trip.arcs[route, 1]]
    out = np.empty(len(xy_obs), dtype=np.int64)
    for s in range(0, len(xy_obs), 512):
        p = xy_obs[s:s + 512, None, :]
        d, _, _ = point_to_segment(p, a[None], b[None])
        out[s:s + 512] = route[np.argmin(d, axis=1)]
    return out
