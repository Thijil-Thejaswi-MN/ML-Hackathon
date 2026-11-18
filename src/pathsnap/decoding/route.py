"""Stitch decoded arcs into a connected route.  OWNER: M3.   Contract: README 3.10.

Shortest-path fill between consecutive distinct arcs; flag unfillable breaks; produce a [lat, lon] polyline.
Uses only the road network (no reference route).
"""
from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import dijkstra


def _collapse(seq):
    seq = [int(x) for x in np.asarray(seq).ravel() if int(x) >= 0]
    return [x for i, x in enumerate(seq) if i == 0 or x != seq[i - 1]]


def stitch_route(arc_per_obs, graph, projector, cfg: dict):
    """Return (route_arcs, route_polyline [[lat, lon], ...], unfilled_pair_indices).

    For consecutive decoded arcs a -> b with to(a) != from(b), insert the arcs of the shortest network path
    to(a) ~> from(b) if it exists within cfg['stitch_max_m'] metres; otherwise keep the gap and record it.
    """
    seq = _collapse(arc_per_obs)
    if not seq:
        return [], [], []
    arcs = graph.arcs
    max_m = float((cfg or {}).get("stitch_max_m", 3000.0))
    pairs = [(k, seq[k], seq[k + 1]) for k in range(len(seq) - 1) if arcs[seq[k], 1] != arcs[seq[k + 1], 0]]

    paths = {}
    if pairs:
        src = np.unique([arcs[x, 1] for _, x, _ in pairs])
        n = graph.n_nodes
        key = arcs[:, 0].astype(np.int64) * n + arcs[:, 1]
        order = np.lexsort((graph.arc_len, key))           # shortest arc first for parallel arcs
        skey = key[order]
        for s in range(0, len(src), 32):
            idx = src[s:s + 32]
            D, P = dijkstra(graph.adj, directed=True, indices=idx, limit=max_m, return_predecessors=True)
            D, P = np.atleast_2d(D), np.atleast_2d(P)
            row = {int(v): r for r, v in enumerate(idx)}
            for k, x, y in pairs:
                r = row.get(int(arcs[x, 1]))
                if r is None:
                    continue
                tgt = int(arcs[y, 0])
                if not np.isfinite(D[r, tgt]):
                    paths[k] = None
                    continue
                nodes = [tgt]
                while nodes[-1] != idx[r]:
                    nodes.append(int(P[r, nodes[-1]]))
                nodes = nodes[::-1]
                q = np.array(nodes[:-1], dtype=np.int64) * n + np.array(nodes[1:], dtype=np.int64)
                paths[k] = order[np.searchsorted(skey, q)].tolist()

    route, unfilled = [seq[0]], []
    for k in range(len(seq) - 1):
        nxt = seq[k + 1]
        if arcs[seq[k], 1] != arcs[nxt, 0]:
            fill = paths.get(k)
            if fill is None:
                unfilled.append(k)
            else:
                route.extend(fill)
        route.append(nxt)
    route = _collapse(route)

    node_ids = [int(arcs[route[0], 0])] + [int(arcs[x, 1]) for x in route]
    lonlat = projector.inverse(graph.xy_nodes[node_ids])
    polyline = [[float(lat), float(lon)] for lon, lat in lonlat]
    return route, polyline, unfilled
