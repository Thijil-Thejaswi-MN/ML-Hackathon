"""Folium map builder.  OWNER: M3.   Contract: README 3.12 [FROZEN signature].

Used by M1 (EDA), M3 (report) and M4 (UI). Remember Folium wants [lat, lon].
`reference` is for OFFLINE use only (train / student_test). The UI must never pass it (README R2).
"""
from __future__ import annotations


def build_trip_map(trip, *, gps=True, network=True, reference=None, classifier_route=None,
                   hmm_route=None, bounds_buffer_m: float = 300.0):
    """Return a folium.Map with FeatureGroups (GPS, network, reference, classifier, HMM), legend, LayerControl, fit_bounds."""
    raise NotImplementedError
