"""HMM + Viterbi decoder.  OWNER: M3.   Contract: README 3.10.

Model (designed by M3, see notebooks/pathsnap_m3_hmm_route_matching.ipynb):
  * States: the candidate arcs of each GPS observation (arc ids preserved).
  * Emission: log of the classifier probability p, optionally combined with a Gaussian GPS-distance term
        log e = log p - lambda * d^2 / (2 sigma^2),  normalised within the observation's candidate set.
  * Transition (Newson & Krumm style): compare the along-network distance between the two positions with the
    straight-line GPS displacement:  log T = -|d_network - d_gps| / beta.
    d_network = rest of arc a + shortest path (to-node of a -> from-node of b) + start of arc b, or the
    along-arc distance if a == b. Paths longer than v_max * dt + slack are unreachable (log epsilon).
    Moving backwards along the same directed arc costs an extra `backward_penalty`.
  * Chain breaks: observations without candidates and time gaps > t_gap_s split the trip into runs; each run is
    decoded independently, unmatched observations inherit the nearest decoded arc (matched=False).

Implementation: road-network distances are computed ONCE per trip (bounded Dijkstra from the to-nodes of all
candidate arcs, scipy.sparse.csgraph) and stored as a sorted key/value table; each Viterbi step is then a
vectorised K_prev x K_cur matrix operation. Uses only track / nodes / arcs and the candidate probabilities.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from scipy.sparse.csgraph import dijkstra


@dataclass
class DecodeResult:
    arc_per_obs: np.ndarray                 # (n_obs,)
    matched: np.ndarray                     # (n_obs,) bool
    segments: list = field(default_factory=list)
    route_arcs: list = field(default_factory=list)
    route_polyline: list = field(default_factory=list)   # [[lat, lon], ...]
    breaks: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)


DEFAULT_HMM_CFG = {"beta_m": 50.0, "epsilon": 1e-6, "t_gap_s": 30.0, "v_max_mps": 40.0, "dijkstra_slack_m": 100.0,
                   "backward_penalty": 5.0, "emission_distance_weight": 0.0, "emission_sigma_m": 20.0,
                   "stitch_max_m": 3000.0}


class NetworkDistances:
    """Bounded node-to-node network distances, sources = to-nodes, targets = from-nodes of candidate arcs."""

    def __init__(self, graph, src_nodes: np.ndarray, dst_nodes: np.ndarray, cutoff_m: float, chunk: int = 64):
        self.src = np.unique(src_nodes)
        self.dst = np.unique(dst_nodes)
        self.n_dst = len(self.dst)
        keys, vals = [], []
        for s in range(0, len(self.src), chunk):
            idx = self.src[s:s + chunk]
            D = np.atleast_2d(dijkstra(graph.adj, directed=True, indices=idx, limit=cutoff_m))[:, self.dst]
            r, c = np.nonzero(np.isfinite(D))
            keys.append((r + s).astype(np.int64) * self.n_dst + c)
            vals.append(D[r, c].astype(np.float32))
        self.keys = np.concatenate(keys) if keys else np.empty(0, np.int64)
        self.vals = np.concatenate(vals) if vals else np.empty(0, np.float32)   # keys are already sorted

    def lookup(self, src_nodes: np.ndarray, dst_nodes: np.ndarray) -> np.ndarray:
        """Matrix (len(src), len(dst)) of distances; inf where unreachable within the cutoff."""
        si = np.searchsorted(self.src, src_nodes)
        di = np.searchsorted(self.dst, dst_nodes)
        q = si[:, None].astype(np.int64) * self.n_dst + di[None, :]
        pos = np.clip(np.searchsorted(self.keys, q), 0, max(len(self.keys) - 1, 0))
        out = np.full(q.shape, np.inf)
        if len(self.keys):
            hit = self.keys[pos] == q
            out[hit] = self.vals[pos[hit]]
        return out


def _emission_log_probs(p: np.ndarray, dist_m: np.ndarray, cfg: dict) -> np.ndarray:
    """Normalised log emission for one observation's candidates (M3's formula)."""
    log_e = np.log(np.clip(p, 1e-12, 1.0))
    w = float(cfg["emission_distance_weight"])
    if w > 0:
        sigma = max(float(cfg["emission_sigma_m"]), 1e-9)
        log_e = log_e - w * dist_m ** 2 / (2.0 * sigma ** 2)
    m = log_e.max()
    return log_e - (m + np.log(np.exp(log_e - m).sum()))


def _viterbi_run(run, states, fracs, emis, obs_xy, t, graph, net: NetworkDistances, cfg):
    beta = max(float(cfg["beta_m"]), 1e-9)
    log_eps = np.log(max(float(cfg["epsilon"]), 1e-300))
    v_max, slack = float(cfg["v_max_mps"]), float(cfg["dijkstra_slack_m"])
    back_pen = float(cfg["backward_penalty"])
    arcs, alen = graph.arcs, graph.arc_len

    delta = emis[run[0]].copy()
    backs = []
    for k in range(1, len(run)):
        i0, i1 = run[k - 1], run[k]
        A, B = states[i0], states[i1]
        fa, fb = fracs[i0], fracs[i1]
        gps = float(np.hypot(*(obs_xy[i1] - obs_xy[i0])))
        cutoff = v_max * max(t[i1] - t[i0], 0.0) + slack
        same = A[:, None] == B[None, :]
        to_a, fr_b = arcs[A, 1], arcs[B, 0]
        middle = net.lookup(to_a, fr_b)
        middle[to_a[:, None] == fr_b[None, :]] = 0.0
        middle[middle > cutoff] = np.inf
        la, lb = alen[A][:, None], alen[B][None, :]
        net_d = np.where(same, np.abs(fb[None, :] - fa[:, None]) * la, (1 - fa[:, None]) * la + middle + fb[None, :] * lb)
        trans = np.where(np.isfinite(net_d), -np.abs(net_d - gps) / beta, log_eps)
        trans = trans - back_pen * (same & (fb[None, :] < fa[:, None]))
        scores = delta[:, None] + trans
        bp = np.argmax(scores, axis=0)
        delta = scores[bp, np.arange(len(B))] + emis[i1]
        backs.append(bp)
    path = np.empty(len(run), dtype=np.int64)
    path[-1] = int(np.argmax(delta))
    for k in range(len(run) - 1, 0, -1):
        path[k - 1] = backs[k - 1][path[k]]
    return np.array([states[run[k]][path[k]] for k in range(len(run))], dtype=np.int64)


def decode(cands, trip, graph, projector, cfg: dict) -> DecodeResult:
    """cands: DataFrame[obs_idx, arc_id, p, dist_m (, proj_t)] (p = classifier probability). One arc per observation."""
    t_start = perf_counter()
    cfg = {**DEFAULT_HMM_CFG, **(cfg or {})}
    n_obs = int(trip.n_obs)
    track = np.asarray(trip.track, dtype=float)
    t = track[:, 2]
    missing = {"obs_idx", "arc_id", "p"} - set(cands.columns)
    if missing:
        raise ValueError(f"Candidate table is missing columns: {sorted(missing)}")
    obs_xy = projector.forward(track[:, 0], track[:, 1])

    c = cands[["obs_idx", "arc_id", "p"] + [x for x in ("dist_m", "proj_t") if x in cands.columns]].copy()
    c = c.sort_values(["obs_idx", "arc_id"]).drop_duplicates(["obs_idx", "arc_id"])
    o = c["obs_idx"].to_numpy().astype(np.int64)
    a = c["arc_id"].to_numpy().astype(np.int64)
    if "proj_t" not in c:
        from ..geo import point_to_segment
        _, ft, _ = point_to_segment(obs_xy[o], graph.a_xy[a], graph.b_xy[a])
        c["proj_t"] = ft
    if "dist_m" not in c:
        from ..geo import point_to_segment
        c["dist_m"], _, _ = point_to_segment(obs_xy[o], graph.a_xy[a], graph.b_xy[a])
    p, fr, dm = c["p"].to_numpy(float), c["proj_t"].to_numpy(float), c["dist_m"].to_numpy(float)
    bounds = np.flatnonzero(np.r_[True, o[1:] != o[:-1], True])
    states, fracs, emis = {}, {}, {}
    for s, e in zip(bounds[:-1], bounds[1:]):
        i = int(o[s])
        states[i], fracs[i] = a[s:e], fr[s:e]
        emis[i] = _emission_log_probs(p[s:e], dm[s:e], cfg)

    # runs: consecutive observations with candidates and no large time gap
    obs_with = np.array(sorted(states), dtype=np.int64)
    runs = []
    if len(obs_with):
        cut = np.flatnonzero((np.diff(obs_with) != 1) | (np.diff(t[obs_with]) <= 0) |
                             (np.diff(t[obs_with]) > float(cfg["t_gap_s"]))) + 1
        runs = [r.tolist() for r in np.split(obs_with, cut)]

    # network distances once per trip; cutoff covers the largest time step inside any run
    dts = [t[r[1:]] - t[r[:-1]] for r in runs if len(r) > 1]
    max_dt = float(max((d.max() for d in dts), default=1.0))
    cutoff = float(cfg["v_max_mps"]) * max_dt + float(cfg["dijkstra_slack_m"])
    t0 = perf_counter()
    net = NetworkDistances(graph, graph.arcs[a, 1], graph.arcs[a, 0], cutoff) if len(a) else None
    t_net = perf_counter() - t0

    arc_per_obs = np.full(n_obs, -1, dtype=np.int64)
    matched = np.zeros(n_obs, dtype=bool)
    segments, breaks = [], []
    t0 = perf_counter()
    for k, run in enumerate(runs):
        arc_per_obs[run] = _viterbi_run(run, states, fracs, emis, obs_xy, t, graph, net, cfg)
        matched[run] = True
        segments.append((int(run[0]), int(run[-1])))
        if k > 0:
            breaks.append(int(run[0]))
    t_vit = perf_counter() - t0
    valid = np.flatnonzero(matched)
    if len(valid) and (~matched).any():   # unmatched obs inherit the nearest decoded arc (stay matched=False)
        um = np.flatnonzero(~matched)
        pos = np.clip(np.searchsorted(valid, um), 1, len(valid) - 1) if len(valid) > 1 else np.zeros(len(um), int)
        if len(valid) > 1:
            left, right = valid[pos - 1], valid[pos]
            near = np.where(um - left <= right - um, left, right)
        else:
            near = np.full(len(um), valid[0])
        arc_per_obs[um] = arc_per_obs[near]

    t0 = perf_counter()
    from . import route as stitching       # route-stitching module (network only)
    route_arcs, route_polyline, unfilled = stitching.stitch_route(arc_per_obs, graph, projector, cfg)
    t_route = perf_counter() - t0
    return DecodeResult(arc_per_obs=arc_per_obs, matched=matched, segments=segments, route_arcs=route_arcs,
                        route_polyline=route_polyline, breaks=breaks,
                        timings={"network_s": t_net, "viterbi_s": t_vit, "route": t_route * 1000.0,
                                 "route_unfilled_gaps": len(unfilled), "decode_seconds": perf_counter() - t_start})
