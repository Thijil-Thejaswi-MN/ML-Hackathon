"""Folium map builder.  OWNER: M3.   Contract: README 3.12 [FROZEN signature].

Used by M1 (EDA), M3 (report) and M4 (UI). Remember Folium wants [lat, lon].
`reference` is for OFFLINE use only (train / student_test). The UI must never pass it (README R2).
"""
from __future__ import annotations
import folium
from folium import FeatureGroup, LayerControl
import numpy as np


def build_trip_map(
    trip,
    *,
    gps: bool = True,
    network: bool = True,
    reference=None,
    classifier_route=None,
    hmm_route=None,
    bounds_buffer_m: float = 300.0,
) -> folium.Map:
    """Return a folium.Map with FeatureGroups (GPS, network, reference, classifier, HMM),
    legend, LayerControl, and fit_bounds.
    """
    track = trip.track  # (n_obs, 3): lon, lat, t
    nodes = trip.nodes  # (n_nodes, 2): lon, lat
    arcs = trip.arcs    # (n_arcs, 2): from, to

    # Center of map from track (lat, lon)
    mean_lat = float(np.mean(track[:, 1]))
    mean_lon = float(np.mean(track[:, 0]))

    m = folium.Map(location=[mean_lat, mean_lon], zoom_start=13, tiles="CartoDB positron")

    # 1. Road Network Layer (clipped near trip to keep HTML small)
    if network:
        fg_net = FeatureGroup(name="Road Network", show=False)
        min_lon, max_lon = float(np.min(track[:, 0])), float(np.max(track[:, 0]))
        min_lat, max_lat = float(np.min(track[:, 1])), float(np.max(track[:, 1]))
        buffer_deg = 0.02  # ~2km buffer

        # Cap network segments so map stays responsive
        count = 0
        for arc in arcs:
            u, v = arc[0], arc[1]
            u_lon, u_lat = nodes[u, 0], nodes[u, 1]
            v_lon, v_lat = nodes[v, 0], nodes[v, 1]
            if (min_lon - buffer_deg <= u_lon <= max_lon + buffer_deg and
                min_lat - buffer_deg <= u_lat <= max_lat + buffer_deg):
                folium.PolyLine(
                    locations=[[u_lat, u_lon], [v_lat, v_lon]],
                    color="#888888",
                    weight=1.5,
                    opacity=0.5,
                ).add_to(fg_net)
                count += 1
                if count >= 3000:
                    break
        fg_net.add_to(m)

    # 2. GPS Layer (Blue dots)
    if gps:
        fg_gps = FeatureGroup(name="Observed GPS", show=True)
        # Subsample if large
        step = max(1, len(track) // 500)
        sub_track = track[::step]
        for idx, row in enumerate(sub_track):
            lon, lat, t = row[0], row[1], row[2]
            folium.CircleMarker(
                location=[lat, lon],
                radius=3,
                color="#0066cc",
                fill=True,
                fill_color="#0066cc",
                fill_opacity=0.7,
                popup=f"Obs #{idx * step} | t={t:.1f}s",
            ).add_to(fg_gps)
        fg_gps.add_to(m)

    # 3. Reference Route (Offline only, Green)
    if reference is not None and len(reference) > 0:
        fg_ref = FeatureGroup(name="Reference Route (Truth)", show=True)
        folium.PolyLine(
            locations=reference,
            color="#28a745",
            weight=4,
            opacity=0.8,
        ).add_to(fg_ref)
        fg_ref.add_to(m)

    # 4. Classifier-only Route (Orange)
    if classifier_route is not None and len(classifier_route) > 0:
        fg_cls = FeatureGroup(name="Classifier Route", show=False)
        folium.PolyLine(
            locations=classifier_route,
            color="#fd7e14",
            weight=3,
            opacity=0.7,
        ).add_to(fg_cls)
        fg_cls.add_to(m)

    # 5. HMM Route (Red)
    if hmm_route is not None and len(hmm_route) > 0:
        fg_hmm = FeatureGroup(name="Predicted Route (HMM)", show=True)
        folium.PolyLine(
            locations=hmm_route,
            color="#dc3545",
            weight=4,
            opacity=0.9,
        ).add_to(fg_hmm)
        fg_hmm.add_to(m)

    # Start & End Markers
    start_lat, start_lon = float(track[0, 1]), float(track[0, 0])
    end_lat, end_lon = float(track[-1, 1]), float(track[-1, 0])
    folium.Marker(
        [start_lat, start_lon],
        popup="Start",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(m)
    folium.Marker(
        [end_lat, end_lon],
        popup="End",
        icon=folium.Icon(color="red", icon="stop"),
    ).add_to(m)

    # Fit bounds
    m.fit_bounds([[float(np.min(track[:, 1])), float(np.min(track[:, 0]))],
                  [float(np.max(track[:, 1])), float(np.max(track[:, 0]))]])

    LayerControl().add_to(m)
    return m
