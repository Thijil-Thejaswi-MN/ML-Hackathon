"""Server-side scoring.  OWNER: M4.

The ONLY runtime module allowed to read the reference route (README R2). Called AFTER prediction has finished.
Uses labels.derive_obs_labels() + metrics.py. Returns metrics only -- never labels or the route itself.
"""
import os


def eval_dir() -> str:
    return os.environ.get("PATHSNAP_EVAL_DIR", "")


def score_trip(trip_id: str, prediction) -> dict:
    """Load <eval_dir>/<trip_id>/<trip_id>.route, derive labels, compute Top-1 (HMM and classifier-only),
    route continuity, sequence-level F1 and candidate miss rate. Return a dict matching api.schemas.Metrics."""
    raise NotImplementedError
