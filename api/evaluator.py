"""Server-side scoring.  OWNER: M4.

The ONLY runtime module allowed to read the reference route (README R2). Called AFTER prediction has finished.
Uses metrics and route comparison. Returns metrics only -- never labels or the route itself.
"""
from pathlib import Path
import os
import numpy as np


def eval_dir() -> str:
    """Return configured evaluation directory or default to Student folder/student_test."""
    configured = os.environ.get("PATHSNAP_EVAL_DIR", "")
    if configured and Path(configured).exists():
        return configured

    # Try common local locations
    candidates = [
        Path("data/student_test"),
        Path("../Student folder/student_test"),
        Path("Student folder/student_test"),
        Path("/Users/uthkarsha/Documents/ML Hackaton /Student folder/student_test"),
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    return ""


def score_trip(trip_id: str, prediction) -> dict:
    """Score a finished prediction against <eval_dir>/<trip_id>/<trip_id>.route (server side only).

    Same definitions as the report (src/pathsnap/metrics.py, README 3.6):
      * Top-1 = share of GPS observations whose predicted arc equals the observation's reference arc. The per-
        observation reference arcs are derived from the .route with labels.derive_obs_labels (monotone alignment),
        exactly as for train / student_test.
      * Route continuity = share of consecutive distinct predicted arcs that connect (to-node == from-node).
      * Sequence F1 = arc-set F1 of the stitched predicted route vs the reference route.
      * Candidate miss rate = share of observations whose reference arc is not among their candidates.
    Returns metrics only -- never labels or the route itself.
    """
    import json
    from pathsnap.io import load_trip, load_route
    from pathsnap.geo import LocalProjector
    from pathsnap.graph import build_graph
    from pathsnap.labels import derive_obs_labels
    from pathsnap.candidates import generate_candidates
    from pathsnap.metrics import route_continuity, sequence_metrics

    edir = Path(eval_dir())
    route_file = edir / trip_id / f"{trip_id}.route"
    if not route_file.exists():
        # No reference available (e.g. a trip without .route): report zeros rather than invented numbers
        return {"top1_accuracy_hmm": 0.0, "top1_accuracy_classifier": 0.0, "route_continuity_hmm": 0.0,
                "route_continuity_classifier": 0.0, "sequence_f1": 0.0, "candidate_miss_rate": 0.0}

    trip = load_trip(edir.parent, edir.name, trip_id)
    route = load_route(edir.parent, edir.name, trip_id)
    projector = LocalProjector.from_track(trip.track)
    graph = build_graph(trip, projector)
    truth, _ = derive_obs_labels(trip, route, projector, graph)

    hmm = np.asarray(prediction.arc_per_obs, dtype=np.int64)
    cls = np.asarray(prediction.classifier_arc_per_obs, dtype=np.int64)
    top1_hmm = float((hmm == truth).mean()) if len(hmm) else 0.0
    top1_cls = float((cls == truth).mean()) if len(cls) else 0.0

    def cont(seq):
        v = route_continuity(seq, trip.arcs)["strict"]
        return 1.0 if np.isnan(v) else float(v)

    stitched = prediction.route_arcs if len(prediction.route_arcs) else hmm
    seq_f1 = sequence_metrics(stitched, route)["arc_f1"]

    cand_cfg = {}
    cfg_path = Path(os.environ.get("PATHSNAP_ARTIFACTS_DIR", "artifacts")) / "candidate_config.json"
    if cfg_path.exists():
        cand_cfg = json.loads(cfg_path.read_text())
    cands = generate_candidates(trip, graph, projector, cand_cfg)
    hit = np.zeros(trip.n_obs, dtype=bool)
    o, a = cands["obs_idx"].to_numpy(), cands["arc_id"].to_numpy()
    hit[o[a == truth[o]]] = True

    return {
        "top1_accuracy_hmm": round(top1_hmm, 4),
        "top1_accuracy_classifier": round(top1_cls, 4),
        "route_continuity_hmm": round(cont(hmm), 4),
        "route_continuity_classifier": round(cont(cls), 4),
        "sequence_f1": round(float(seq_f1), 4),
        "candidate_miss_rate": round(float(1.0 - hit.mean()), 4),
    }
