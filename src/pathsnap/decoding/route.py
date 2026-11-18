"""Stitch decoded arcs into a connected route.  OWNER: M3.

Shortest-path fill between consecutive distinct arcs; flag unfillable breaks; produce a [lat, lon] polyline.
"""
from __future__ import annotations


def stitch_route(arc_per_obs, graph, projector, cfg: dict):
    raise NotImplementedError
