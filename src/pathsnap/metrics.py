"""All metrics, defined ONCE.  OWNER: M2 (M3 adds route metrics via PR).   Contract: README 3.6 [FROZEN].

Every script must call these functions -- never re-implement a metric elsewhere.

Inputs use the processed-table conventions:
  candidate rows: trip_id, obs_idx, arc_id, p (classifier probability), label (0/1, offline only)
  observation rows (obs_labels): trip_id, obs_idx, true_arc_id
"""
from __future__ import annotations
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, log_loss, precision_score, recall_score

EPS = 1e-6


def _obs_key(df: pd.DataFrame) -> np.ndarray:
    """Integer key per (trip_id, obs_idx); trip ids are 8-digit strings."""
    return df["trip_id"].astype(np.int64).to_numpy() * 1_000_000 + df["obs_idx"].to_numpy().astype(np.int64)


def candidate_miss_rate(cands: pd.DataFrame, obs_labels: pd.DataFrame, n_obs_per_trip=None) -> float:
    """Share of observations whose true arc is not in their candidate set (empty sets count as misses).
    `obs_labels` must contain every observation (it does - one row per GPS fix)."""
    hit = cands.loc[cands["label"] == 1, ["trip_id", "obs_idx"]].drop_duplicates()
    return float(1.0 - len(hit) / len(obs_labels))


def candidate_level_metrics(y_true, p, threshold: float) -> dict:
    """precision, recall, f1_pos, macro_f1, accuracy, log_loss (+ confusion matrix) over candidate rows."""
    y_true = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    y_hat = (p >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_hat, labels=[0, 1])
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "recall": float(recall_score(y_true, y_hat, zero_division=0)),
        "f1_pos": float(f1_score(y_true, y_hat, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_hat, average="macro", zero_division=0)),
        "accuracy": float((y_hat == y_true).mean()),
        "log_loss": float(log_loss(y_true, p, labels=[0, 1])),
        "confusion_matrix": cm.tolist(),   # [[TN, FP], [FN, TP]]
        "n_rows": int(len(y_true)),
    }


def argmax_per_obs(cands_with_p: pd.DataFrame) -> pd.DataFrame:
    """Highest-probability candidate per observation -> DataFrame[trip_id, obs_idx, arc_id, p]."""
    key = _obs_key(cands_with_p)
    p = cands_with_p["p"].to_numpy()
    order = np.lexsort((-p, key))
    first = np.ones(len(order), dtype=bool)
    first[1:] = key[order][1:] != key[order][:-1]
    sel = order[first]
    return cands_with_p.iloc[sel][["trip_id", "obs_idx", "arc_id", "p"]].reset_index(drop=True)


def obs_top1_accuracy(cands_with_p: pd.DataFrame, obs_labels: pd.DataFrame, n_obs_per_trip=None) -> dict:
    """{'top1': over ALL observations in obs_labels (misses / empty sets count wrong), 'top1_given_hit': ...}"""
    best = argmax_per_obs(cands_with_p).rename(columns={"arc_id": "pred_arc"})
    j = obs_labels[["trip_id", "obs_idx", "true_arc_id"]].merge(best[["trip_id", "obs_idx", "pred_arc"]],
                                                                 on=["trip_id", "obs_idx"], how="left")
    correct = (j["pred_arc"].to_numpy() == j["true_arc_id"].to_numpy())
    hit_keys = cands_with_p.loc[cands_with_p["label"] == 1, ["trip_id", "obs_idx"]].drop_duplicates() \
        if "label" in cands_with_p else None
    out = {"top1": float(correct.mean()), "n_obs": int(len(j))}
    if hit_keys is not None:
        h = j.merge(hit_keys.assign(_hit=True), on=["trip_id", "obs_idx"], how="left")["_hit"].fillna(False).astype(bool).to_numpy()
        out["top1_given_hit"] = float(correct[h].mean()) if h.any() else float("nan")
        out["candidate_miss_rate"] = float(1 - h.mean())
    return out


def _collapse(seq):
    seq = [int(a) for a in np.asarray(seq).ravel() if int(a) >= 0]
    return [a for i, a in enumerate(seq) if i == 0 or a != seq[i - 1]]


def route_continuity(arc_seq, arcs: np.ndarray, hops: int = 3) -> dict:
    """collapse consecutive duplicates; {'strict': share with to==from, 'relaxed': reachable within `hops` arcs}"""
    seq = _collapse(arc_seq)
    if len(seq) < 2:
        return {"strict": float("nan"), "relaxed": float("nan"), "n_pairs": 0, "n_arcs": len(seq)}
    a = np.asarray(seq)
    to_n, fr_n = arcs[a[:-1], 1], arcs[a[1:], 0]
    strict = to_n == fr_n
    # relaxed: BFS from to_node over up to `hops` arcs, looking for the next arc's from_node
    out_nbrs: dict[int, np.ndarray] = {}
    order = np.argsort(arcs[:, 0], kind="stable")
    starts = np.searchsorted(arcs[order, 0], np.arange(arcs.max() + 2))

    def nbrs(n):
        if n not in out_nbrs:
            out_nbrs[n] = arcs[order[starts[n]:starts[n + 1]], 1]
        return out_nbrs[n]

    relaxed = strict.copy()
    for k in np.where(~strict)[0]:
        frontier, seen, target = {int(to_n[k])}, {int(to_n[k])}, int(fr_n[k])
        for _ in range(hops):
            nxt = set()
            for n in frontier:
                nxt.update(int(x) for x in nbrs(n))
            if target in nxt:
                relaxed[k] = True
                break
            frontier = nxt - seen
            seen |= nxt
            if not frontier:
                break
    return {"strict": float(strict.mean()), "relaxed": float(relaxed.mean()),
            "n_pairs": int(len(strict)), "n_arcs": int(len(seq))}


def sequence_metrics(pred_route, ref_route) -> dict:
    """arc-set precision/recall/F1 + sequence similarity (difflib ratio = 2*matched / total length)."""
    p, r = _collapse(pred_route), _collapse(ref_route)
    sp, sr = set(p), set(r)
    inter = len(sp & sr)
    prec = inter / len(sp) if sp else 0.0
    rec = inter / len(sr) if sr else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    sim = SequenceMatcher(None, p, r, autojunk=False).ratio() if p and r else 0.0
    return {"arc_precision": prec, "arc_recall": rec, "arc_f1": f1, "seq_similarity": sim,
            "n_pred_arcs": len(p), "n_ref_arcs": len(r)}


def pick_threshold(y_true_oof, p_oof, grid=None) -> float:
    """Maximise group-CV macro-F1 on TRAIN OOF probabilities only."""
    y = np.asarray(y_true_oof).astype(bool)
    p = np.asarray(p_oof, dtype=np.float64)
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2) if grid is None else grid
    best_t, best = 0.5, -1.0
    n_pos, n_neg = y.sum(), (~y).sum()
    for t in grid:
        yh = p >= t
        tp = (yh & y).sum()
        fp = (yh & ~y).sum()
        fn = n_pos - tp
        tn = n_neg - fp
        f1p = 2 * tp / max(2 * tp + fp + fn, 1)
        f1n = 2 * tn / max(2 * tn + fn + fp, 1)
        m = (f1p + f1n) / 2
        if m > best:
            best, best_t = m, float(t)
    return best_t
