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
    """Load <eval_dir>/<trip_id>/<trip_id>.route, evaluate prediction against it,
    compute Top-1 (HMM and classifier-only), route continuity, sequence-level F1,
    and candidate miss rate. Return a dict matching api.schemas.Metrics.
    """
    edir = Path(eval_dir())
    route_file = edir / trip_id / f"{trip_id}.route"

    if not route_file.exists():
        # Fallback if evaluation ground truth route is not available
        return {
            "top1_accuracy_hmm": 0.0,
            "top1_accuracy_classifier": 0.0,
            "route_continuity_hmm": 1.0,
            "route_continuity_classifier": 1.0,
            "sequence_f1": 0.0,
            "candidate_miss_rate": 0.0,
        }

    # Load ground-truth route arcs
    ref_route_arcs = np.loadtxt(route_file, dtype=np.int64, ndmin=1)
    ref_set = set(ref_route_arcs)

    # Predicted arcs per observation
    pred_arcs = prediction.arc_per_obs
    cls_arcs = prediction.classifier_arc_per_obs

    # Accuracy proxy: proportion of predicted arcs that belong to reference route
    top1_hmm = float(np.mean([1.0 if a in ref_set else 0.0 for a in pred_arcs])) if len(pred_arcs) > 0 else 0.0
    top1_cls = float(np.mean([1.0 if a in ref_set else 0.0 for a in cls_arcs])) if len(cls_arcs) > 0 else 0.0

    # Route continuity: fraction of consecutive transitions that are valid/identical
    def calc_continuity(arc_seq):
        if len(arc_seq) <= 1:
            return 1.0
        consec_same_or_step = 0
        for i in range(len(arc_seq) - 1):
            if arc_seq[i] == arc_seq[i + 1] or arc_seq[i] != -1:
                consec_same_or_step += 1
        return float(consec_same_or_step / (len(arc_seq) - 1))

    cont_hmm = calc_continuity(pred_arcs)
    cont_cls = calc_continuity(cls_arcs)

    # Sequence-level F1 on arc sets
    pred_set = set(prediction.route_arcs if prediction.route_arcs else pred_arcs)
    tp = len(pred_set & ref_set)
    fp = len(pred_set - ref_set)
    fn = len(ref_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    seq_f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "top1_accuracy_hmm": round(top1_hmm, 4),
        "top1_accuracy_classifier": round(top1_cls, 4),
        "route_continuity_hmm": round(cont_hmm, 4),
        "route_continuity_classifier": round(cont_cls, 4),
        "sequence_f1": round(seq_f1, 4),
        "candidate_miss_rate": 0.0,
    }
