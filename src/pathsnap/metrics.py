"""All metrics, defined ONCE.  OWNER: M2 (M3 adds route metrics via PR).   Contract: README 3.6 [FROZEN].

Every script must call these functions -- never re-implement a metric elsewhere.
"""
from __future__ import annotations


def candidate_miss_rate(cands, obs_labels, n_obs_per_trip) -> float: ...
def candidate_level_metrics(y_true, p, threshold: float) -> dict:
    """precision, recall, f1_pos, macro_f1, accuracy, log_loss (+ confusion matrix)."""
    raise NotImplementedError
def obs_top1_accuracy(cands_with_p, obs_labels, n_obs_per_trip) -> dict:
    """{'top1': over ALL observations (misses count wrong), 'top1_given_hit': ...}"""
    raise NotImplementedError
def route_continuity(arc_seq, arcs, hops: int = 3) -> dict:
    """collapse consecutive duplicates; {'strict': share with to==from, 'relaxed': reachable within `hops`}"""
    raise NotImplementedError
def sequence_metrics(pred_route, ref_route) -> dict:
    """arc-set precision/recall/F1 + normalised edit distance or LCS ratio."""
    raise NotImplementedError
def pick_threshold(y_true_oof, p_oof) -> float:
    """Maximise group-CV macro-F1 on TRAIN OOF probabilities only."""
    raise NotImplementedError
