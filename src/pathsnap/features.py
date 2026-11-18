"""Feature engineering.  OWNER: M1.   Contract: README 3.5.

build_features() is used IDENTICALLY offline (dataset builder) and online (pipeline.predict_trip) -> no train/serve skew.
Column order is saved to artifacts/features.json. Uses only track / nodes / arcs and the candidate table.

One row per (observation, candidate arc). Feature groups and why:
  geometry     - how close the GPS point is to the arc, absolutely and relative to the other candidates
  direction    - vehicle heading vs arc bearing; the ONLY thing that separates an arc from its reverse twin
  window       - do the neighbouring GPS points also lie along this arc? (smooths single-point noise)
  kinematics   - speed / time gaps; heading is unreliable when stopped, distances unreliable after gaps
  context      - candidate count, node degrees, twin flag
  continuity   - is this arc the same as / directly connected to the best candidates of the previous and
                 next observation? (sequence hint for the classifier; the HMM does the full job later)
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .geo import point_to_segment, signed_offset, vehicle_heading, angle_diff

DEFAULT_FEAT_CFG = {"heading_window": 5, "neighbour_window": 5, "context_top": 3}

FEATURE_COLUMNS: list[str] = [
    # geometry
    "dist_m", "rank", "dist_minus_min", "dist_ratio_min", "dist_minus_second", "proj_t", "proj_edge",
    "signed_offset_m", "arc_len_m", "radius_m",
    # direction
    "cos_head", "head_diff", "head_valid", "cos_head_short", "cos_head_long",
    # window
    "win_mean_d2", "win_max_d2", "win_mean_d", "win_max_d", "win_inside_frac",
    # kinematics
    "speed_mps", "accel_mps2", "dt_prev", "dt_next", "stationary",
    # context
    "n_cands", "has_twin", "is_twin_of_nearest", "from_out_deg", "to_in_deg", "to_out_deg",
    # continuity
    "same_as_prev_top", "linked_from_prev_top", "same_as_next_top", "linked_to_next_top",
    "prev_nearest_d_to_arc", "next_nearest_d_to_arc",
]


def _obs_kinematics(xy: np.ndarray, t: np.ndarray):
    n = len(xy)
    step = np.zeros(n)
    dt = np.zeros(n)
    if n > 1:
        step[1:] = np.sqrt((np.diff(xy, axis=0) ** 2).sum(1))
        dt[1:] = np.diff(t)
    dt_prev = dt.copy()
    dt_prev[0] = 0.0
    dt_next = np.zeros(n)
    dt_next[:-1] = dt[1:]
    _, speed, _ = vehicle_heading(xy, t, w=2)
    accel = np.zeros(n)
    if n > 2:
        accel[1:] = np.diff(speed) / np.maximum(dt[1:], 1e-6)
    accel = np.clip(accel, -20, 20)
    return speed, accel, dt_prev, dt_next


def build_features(trip, graph, projector, candidates: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Return candidates + FEATURE_COLUMNS (float32). Must not use labels or the reference route."""
    cfg = {**DEFAULT_FEAT_CFG, **(cfg or {})}
    df = candidates.copy()
    n = trip.n_obs
    xy = projector.forward(trip.track[:, 0], trip.track[:, 1])
    t = trip.track[:, 2]
    o = df["obs_idx"].to_numpy().astype(np.int64)
    arc = df["arc_id"].to_numpy().astype(np.int64)
    A, B = graph.a_xy[arc], graph.b_xy[arc]
    d = df["dist_m"].to_numpy().astype(np.float64)

    # ---- geometry
    g = df.groupby("obs_idx")["dist_m"]
    dmin = g.transform("min").to_numpy()
    # candidates are sorted by distance within each observation, so rank 0 / rank 1 = best / second best
    cnt = np.bincount(o, minlength=n)
    rk_all = df["rank"].to_numpy()
    first_d = np.full(n, np.inf)
    second = np.full(n, np.inf)
    first_d[o[rk_all == 0]] = d[rk_all == 0]
    second[o[rk_all == 1]] = d[rk_all == 1]
    other_best = np.where(rk_all == 0, second[o], first_d[o])  # best distance among the OTHER candidates
    df["dist_minus_min"] = d - dmin
    df["dist_ratio_min"] = d / (dmin + 1.0)
    df["dist_minus_second"] = np.clip(d - other_best, -500, 500)
    pt = df["proj_t"].to_numpy()
    df["proj_edge"] = np.minimum(pt, 1 - pt)
    df["signed_offset_m"] = signed_offset(xy[o], A, B)
    df["arc_len_m"] = graph.arc_len[arc]

    # ---- direction
    hw = int(cfg["heading_window"])
    head, _, hvalid = vehicle_heading(xy, t, w=hw)
    head_s, _, hv_s = vehicle_heading(xy, t, w=1)
    head_l, _, hv_l = vehicle_heading(xy, t, w=3 * hw)
    ab = graph.arc_bearing[arc]
    df["cos_head"] = np.where(hvalid[o], np.cos(head[o] - ab), 0.0)
    df["head_diff"] = np.where(hvalid[o], angle_diff(head[o], ab), np.pi / 2)
    df["head_valid"] = hvalid[o].astype(np.float32)
    df["cos_head_short"] = np.where(hv_s[o], np.cos(head_s[o] - ab), 0.0)
    df["cos_head_long"] = np.where(hv_l[o], np.cos(head_l[o] - ab), 0.0)

    # ---- window: distance of neighbouring points to THIS arc
    def window(w):
        ds, inside = [], []
        for k in range(-w, w + 1):
            if k == 0:
                continue
            j = np.clip(o + k, 0, n - 1)
            dk, tk, _ = point_to_segment(xy[j], A, B)
            ds.append(dk)
            inside.append((tk > 0) & (tk < 1))
        ds = np.stack(ds, 1)
        return ds.mean(1), ds.max(1), np.stack(inside, 1).mean(1)
    m2, x2, _ = window(2)
    mw, xw, inside = window(int(cfg["neighbour_window"]))
    df["win_mean_d2"], df["win_max_d2"] = m2, x2
    df["win_mean_d"], df["win_max_d"], df["win_inside_frac"] = mw, xw, inside

    # ---- kinematics
    speed, accel, dt_prev, dt_next = _obs_kinematics(xy, t)
    df["speed_mps"] = np.clip(speed[o], 0, 80)
    df["accel_mps2"] = accel[o]
    df["dt_prev"] = np.clip(dt_prev[o], 0, 3600)
    df["dt_next"] = np.clip(dt_next[o], 0, 3600)
    df["stationary"] = (speed[o] < 1.0).astype(np.float32)

    # ---- context
    df["n_cands"] = cnt[o]
    tw = graph.twin[arc]
    df["has_twin"] = (tw >= 0).astype(np.float32)
    nearest_arc = np.full(n, -1, dtype=np.int64)
    r0 = df["rank"].to_numpy() == 0
    nearest_arc[o[r0]] = arc[r0]
    df["is_twin_of_nearest"] = ((tw >= 0) & (tw == nearest_arc[o])).astype(np.float32)
    fr, to = graph.arcs[arc, 0], graph.arcs[arc, 1]
    df["from_out_deg"] = graph.out_deg[fr]
    df["to_in_deg"] = graph.in_deg[to]
    df["to_out_deg"] = graph.out_deg[to]

    # ---- continuity with neighbouring observations' top candidates (by distance rank only)
    K = int(cfg["context_top"])
    top = np.full((n + 2, K), -1, dtype=np.int64)          # padded: row j+1 = obs j
    rk = df["rank"].to_numpy()
    sel = rk < K
    top[o[sel] + 1, rk[sel]] = arc[sel]
    P, N = top[o], top[o + 2]                               # previous / next observation
    Pv, Nv = P >= 0, N >= 0
    P_to = np.where(Pv, graph.arcs[np.maximum(P, 0), 1], -2)
    N_from = np.where(Nv, graph.arcs[np.maximum(N, 0), 0], -2)
    df["same_as_prev_top"] = (P == arc[:, None]).any(1).astype(np.float32)
    df["linked_from_prev_top"] = (P_to == fr[:, None]).any(1).astype(np.float32)
    df["same_as_next_top"] = (N == arc[:, None]).any(1).astype(np.float32)
    df["linked_to_next_top"] = (N_from == to[:, None]).any(1).astype(np.float32)
    pa, na = nearest_arc[np.clip(o - 1, 0, n - 1)], nearest_arc[np.clip(o + 1, 0, n - 1)]
    def d_to(arcs_other):
        ok = arcs_other >= 0
        res = np.full(len(o), 500.0)
        if ok.any():
            # distance between this arc's foot point and the neighbour's nearest arc
            foot = A + df["proj_t"].to_numpy()[:, None] * (B - A)
            dd, _, _ = point_to_segment(foot[ok], graph.a_xy[arcs_other[ok]], graph.b_xy[arcs_other[ok]])
            res[ok] = dd
        return np.minimum(res, 500.0)
    df["prev_nearest_d_to_arc"] = d_to(pa)
    df["next_nearest_d_to_arc"] = d_to(na)

    for c in FEATURE_COLUMNS:
        df[c] = df[c].astype(np.float32)
    return df
