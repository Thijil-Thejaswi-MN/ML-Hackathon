"""Folium map builder.  OWNER: M3 (first version written by M1 for EDA).   Contract: README 3.12 [FROZEN signature].

Used by M1 (EDA), M3 (report) and M4 (UI). Remember Folium wants [lat, lon].
`reference` is for OFFLINE use only (train / student_test). The UI must never pass it (README R2).
Routes are given as sequences of arc ids (one per GPS point or a de-duplicated arc list - both work).
"""
from __future__ import annotations
import numpy as np

from ..geo import LocalProjector
from ..graph import build_graph

LAYER_STYLE = {
    "gps": {"color": "#1f6fd1", "label": "Raw GPS points"},
    "network": {"color": "#9a9a9a", "label": "Road network (near trip)"},
    "reference": {"color": "#1a9e3a", "label": "Reference route"},
    "classifier": {"color": "#f28e1c", "label": "Classifier-only route"},
    "hmm": {"color": "#d62728", "label": "HMM / Viterbi route"},
}


def _dedupe(seq):
    seq = [int(a) for a in np.asarray(seq).ravel() if int(a) >= 0]
    return [a for i, a in enumerate(seq) if i == 0 or a != seq[i - 1]]


def _arc_lines(trip, arc_ids):
    """List of [[lat, lon], [lat, lon]] segments for the given arcs (multi-polyline)."""
    a = trip.nodes[trip.arcs[arc_ids, 0]]
    b = trip.nodes[trip.arcs[arc_ids, 1]]
    return [[[float(p[1]), float(p[0])], [float(q[1]), float(q[0])]] for p, q in zip(a, b)]


def _legend_html(items):
    rows = "".join(
        f'<div><span style="display:inline-block;width:14px;height:4px;background:{c};margin-right:6px;'
        f'vertical-align:middle"></span>{t}</div>' for t, c in items)
    return ('<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:8px 10px;'
            'border:1px solid #888;border-radius:4px;font:12px sans-serif;box-shadow:0 1px 4px rgba(0,0,0,.3)">'
            f'<b>Legend</b>{rows}</div>')


def build_trip_map(trip, *, gps=True, network=True, reference=None, classifier_route=None,
                   hmm_route=None, bounds_buffer_m: float = 300.0, max_gps_points: int = 3000,
                   max_network_arcs: int = 20000, title: str | None = None):
    """Return a folium.Map with FeatureGroups (GPS, network, reference, classifier, HMM), legend, LayerControl, fit_bounds."""
    import folium

    lon, lat, t = trip.track[:, 0], trip.track[:, 1], trip.track[:, 2]
    m = folium.Map(location=[float(np.median(lat)), float(np.median(lon))], zoom_start=14,
                   tiles="OpenStreetMap", control_scale=True)
    legend = []

    if network:
        proj = LocalProjector.from_track(trip.track)
        g = build_graph(trip, proj)
        xy = proj.forward(lon, lat)
        lo, hi = xy.min(0) - bounds_buffer_m, xy.max(0) + bounds_buffer_m
        mid = (g.a_xy + g.b_xy) / 2
        inside = np.where(np.all((mid >= lo) & (mid <= hi), axis=1))[0]
        if len(inside) > max_network_arcs:   # keep the arcs closest to the GPS track
            from scipy.spatial import cKDTree
            d, _ = cKDTree(xy).query(mid[inside])
            inside = inside[np.argsort(d)[:max_network_arcs]]
        fg = folium.FeatureGroup(name=f"{LAYER_STYLE['network']['label']} ({len(inside)} arcs)")
        folium.PolyLine(_arc_lines(trip, inside), color=LAYER_STYLE["network"]["color"], weight=1.5,
                        opacity=0.6).add_to(fg)
        fg.add_to(m)
        legend.append((LAYER_STYLE["network"]["label"], LAYER_STYLE["network"]["color"]))

    for key, route in (("reference", reference), ("classifier", classifier_route), ("hmm", hmm_route)):
        if route is None:
            continue
        arcs = _dedupe(route)
        fg = folium.FeatureGroup(name=f"{LAYER_STYLE[key]['label']} ({len(arcs)} arcs)")
        folium.PolyLine(_arc_lines(trip, np.array(arcs, dtype=np.int64)), color=LAYER_STYLE[key]["color"],
                        weight=6 if key == "reference" else 4, opacity=0.75 if key == "reference" else 0.9).add_to(fg)
        fg.add_to(m)
        legend.append((LAYER_STYLE[key]["label"], LAYER_STYLE[key]["color"]))

    if gps:
        step = max(1, int(np.ceil(len(lon) / max_gps_points)))
        idx = np.arange(0, len(lon), step)
        name = LAYER_STYLE["gps"]["label"] + (f" (every {step}th point)" if step > 1 else "")
        fg = folium.FeatureGroup(name=name)
        for i in idx:
            folium.CircleMarker([float(lat[i]), float(lon[i])], radius=2.5, color=LAYER_STYLE["gps"]["color"],
                                fill=True, fill_opacity=0.8, weight=0,
                                tooltip=f"obs {i} | t={t[i]:.0f}s").add_to(fg)
        fg.add_to(m)
        legend.append((name, LAYER_STYLE["gps"]["color"]))
        se = folium.FeatureGroup(name="Start / end")
        folium.Marker([float(lat[0]), float(lon[0])], tooltip="Start",
                      icon=folium.Icon(color="green", icon="play")).add_to(se)
        folium.Marker([float(lat[-1]), float(lon[-1])], tooltip="End",
                      icon=folium.Icon(color="red", icon="stop")).add_to(se)
        se.add_to(m)

    if title:
        m.get_root().html.add_child(folium.Element(
            f'<div style="position:fixed;top:10px;left:60px;z-index:9999;background:white;padding:4px 10px;'
            f'border:1px solid #888;border-radius:4px;font:bold 13px sans-serif">{title}</div>'))
    m.get_root().html.add_child(folium.Element(_legend_html(legend)))
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[float(lat.min()), float(lon.min())], [float(lat.max()), float(lon.max())]])
    return m
