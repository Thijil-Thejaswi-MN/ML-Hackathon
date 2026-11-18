"""Road-network structures.  OWNER: M3 (M1 added the base structure + helpers).

Needed by candidates.py (spatial index), features.py (arc geometry, degrees) and decoding/ (shortest paths).
Everything here is derived from track/nodes/arcs only -> safe to cache per trip.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra, connected_components

from .geo import bearing


@dataclass
class RoadGraph:
    """Built once per trip from TripData."""
    xy_nodes: np.ndarray       # (n_nodes, 2) metres
    arcs: np.ndarray           # (n_arcs, 2) from/to node ids
    arc_len: np.ndarray        # (n_arcs,) metres
    arc_bearing: np.ndarray    # (n_arcs,) radians (0 = north, clockwise)
    twin: np.ndarray           # (n_arcs,) arc id of the reverse arc (to->from) or -1
    out_deg: np.ndarray        # (n_nodes,) number of arcs leaving each node
    in_deg: np.ndarray         # (n_nodes,) number of arcs entering each node
    adj: sparse.csr_matrix     # node x node, weight = arc length (directed)
    _sp_cache: dict = field(default_factory=dict, repr=False)

    @property
    def n_nodes(self) -> int:
        return int(self.xy_nodes.shape[0])

    @property
    def n_arcs(self) -> int:
        return int(self.arcs.shape[0])

    @property
    def a_xy(self) -> np.ndarray:
        return self.xy_nodes[self.arcs[:, 0]]

    @property
    def b_xy(self) -> np.ndarray:
        return self.xy_nodes[self.arcs[:, 1]]


def build_graph(trip, projector) -> RoadGraph:
    xy = projector.forward(trip.nodes[:, 0], trip.nodes[:, 1])
    arcs = trip.arcs.astype(np.int64)
    a, b = xy[arcs[:, 0]], xy[arcs[:, 1]]
    arc_len = np.sqrt(((b - a) ** 2).sum(1))
    arc_bear = bearing(a, b)
    n = xy.shape[0]

    # reverse twin lookup via a hash of (from, to)
    key = arcs[:, 0] * n + arcs[:, 1]
    rkey = arcs[:, 1] * n + arcs[:, 0]
    order = np.argsort(key)
    pos = np.searchsorted(key[order], rkey)
    pos = np.clip(pos, 0, len(key) - 1)
    hit = key[order][pos] == rkey
    twin = np.where(hit, order[pos], -1)

    out_deg = np.bincount(arcs[:, 0], minlength=n)
    in_deg = np.bincount(arcs[:, 1], minlength=n)
    w = np.maximum(arc_len, 1e-3)  # zero-length arcs would vanish from a sparse matrix
    adj = sparse.csr_matrix((w, (arcs[:, 0], arcs[:, 1])), shape=(n, n))
    return RoadGraph(xy, arcs, arc_len, arc_bear, twin, out_deg, in_deg, adj)


def component_stats(graph: RoadGraph) -> dict:
    """Weakly / strongly connected components and isolated nodes (for EDA)."""
    n_w, lab_w = connected_components(graph.adj, directed=True, connection="weak")
    n_s, _ = connected_components(graph.adj, directed=True, connection="strong")
    sizes = np.bincount(lab_w)
    isolated = int(((graph.out_deg + graph.in_deg) == 0).sum())
    return {
        "n_weak_components": int(n_w),
        "n_strong_components": int(n_s),
        "largest_weak_component_frac": float(sizes.max() / graph.n_nodes),
        "isolated_nodes": isolated,
        "twin_frac": float((graph.twin >= 0).mean()),
    }


def bounded_shortest_paths(graph: RoadGraph, sources, cutoff_m: float):
    """Network distances (m) from each source node to all nodes, capped at cutoff_m (inf beyond).

    Returns dict {source_node: dist_array}. Memoised per graph, so repeated sources are free.
    Used by the HMM transition model (M3).
    """
    sources = np.unique(np.asarray(sources, dtype=np.int64))
    todo = [s for s in sources if (s, cutoff_m) not in graph._sp_cache]
    if todo:
        d = dijkstra(graph.adj, directed=True, indices=todo, limit=cutoff_m)
        d = np.atleast_2d(d)
        for s, row in zip(todo, d):
            graph._sp_cache[(s, cutoff_m)] = row
    return {int(s): graph._sp_cache[(s, cutoff_m)] for s in sources}
