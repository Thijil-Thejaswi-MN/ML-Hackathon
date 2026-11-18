"""Shared offline helpers for 06_tune_decoder.py, 07_eval_hmm.py, 08_make_maps.py.  OWNER: M3.

Offline only: reads derived labels / reference routes to SCORE decoded output. The decoder itself
(src/pathsnap/decoding) never sees them.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.io import load_trip, load_route  # noqa: E402
from pathsnap.geo import LocalProjector  # noqa: E402
from pathsnap.graph import build_graph  # noqa: E402
from pathsnap.decoding.hmm import decode  # noqa: E402
from pathsnap.dataset import load_table  # noqa: E402
from pathsnap.metrics import argmax_per_obs, route_continuity, sequence_metrics  # noqa: E402

RAW = ROOT / "data/raw/Student folder"


def load_probs(split: str, tag: str) -> pd.DataFrame:
    """Candidate table of `split` with the probabilities of model run `tag`:
    train -> out-of-fold probs (results/oof/<tag>), student_test -> results/preds/student_test_<tag>."""
    cands = load_table(ROOT / f"data/processed/candidates_{split}.parquet")
    src = ROOT / ("results/oof" if split == "train" else "results/preds") / (
        f"{tag}.parquet" if split == "train" else f"student_test_{tag}.parquet")
    pr = load_table(src)
    assert len(pr) == len(cands) and (pr["arc_id"].to_numpy() == cands["arc_id"].to_numpy()).all(), \
        f"{src} does not line up with candidates_{split} - rebuild both from the same 02_build_dataset run"
    cands["p"] = pr["p"].to_numpy()
    return cands


class TripCache:
    """Trips, projectors, graphs and candidate groups loaded once (decoder grid search re-decodes many times)."""

    def __init__(self, split: str, cands: pd.DataFrame, trip_ids=None):
        self.split = split
        obs = load_table(ROOT / f"data/processed/obs_labels_{split}.parquet")
        self.items = []
        for tid, c in cands.groupby("trip_id"):
            if trip_ids is not None and tid not in trip_ids:
                continue
            trip = load_trip(RAW, split, tid)
            pj = LocalProjector.from_track(trip.track)
            o = obs[obs.trip_id == tid].sort_values("obs_idx")
            cls = np.full(trip.n_obs, -1, dtype=np.int64)
            b = argmax_per_obs(c)
            cls[b["obs_idx"].to_numpy()] = b["arc_id"].to_numpy()
            self.items.append({"trip_id": tid, "trip": trip, "proj": pj, "graph": build_graph(trip, pj), "cands": c,
                               "truth": o["true_arc_id"].to_numpy(), "reliable": o["label_reliable"].to_numpy(),
                               "route": load_route(RAW, split, tid), "cls": cls})


def evaluate_decoder(cache: TripCache, cfg: dict, keep_arcs: bool = False) -> pd.DataFrame:
    rows = []
    for it in cache.items:
        t0 = time.perf_counter()
        res = decode(it["cands"], it["trip"], it["graph"], it["proj"], cfg)
        sec = time.perf_counter() - t0
        truth, rel, cls, hmm = it["truth"], it["reliable"], it["cls"], res.arc_per_obs
        arcs = it["trip"].arcs
        r = {"trip_id": it["trip_id"], "n_obs": it["trip"].n_obs, "n_reliable": int(rel.sum()),
             "cls_top1": float((cls == truth).mean()), "hmm_top1": float((hmm == truth).mean()),
             "cls_top1_reliable": float((cls == truth)[rel].mean()) if rel.any() else np.nan,
             "hmm_top1_reliable": float((hmm == truth)[rel].mean()) if rel.any() else np.nan,
             "cls_continuity": route_continuity(cls, arcs)["strict"],
             "hmm_continuity": route_continuity(hmm, arcs)["strict"],
             "stitched_continuity": route_continuity(res.route_arcs, arcs)["strict"],
             "cls_arc_f1": sequence_metrics(cls, it["route"])["arc_f1"],
             "hmm_arc_f1": sequence_metrics(res.route_arcs, it["route"])["arc_f1"],
             "hmm_seq_similarity": sequence_metrics(res.route_arcs, it["route"])["seq_similarity"],
             "n_breaks": len(res.breaks), "unfilled_gaps": res.timings["route_unfilled_gaps"],
             "decode_seconds": round(sec, 3)}
        if keep_arcs:
            r["_hmm"], r["_route_arcs"] = hmm, res.route_arcs
        rows.append(r)
    return pd.DataFrame(rows)


def pooled(df: pd.DataFrame) -> dict:
    """Observation-weighted totals over trips."""
    w, wr = df["n_obs"], df["n_reliable"]
    return {"cls_top1": float((df.cls_top1 * w).sum() / w.sum()), "hmm_top1": float((df.hmm_top1 * w).sum() / w.sum()),
            "cls_top1_reliable": float((df.cls_top1_reliable * wr).sum() / wr.sum()),
            "hmm_top1_reliable": float((df.hmm_top1_reliable * wr).sum() / wr.sum()),
            "cls_continuity": float(df.cls_continuity.mean()), "hmm_continuity": float(df.hmm_continuity.mean()),
            "stitched_continuity": float(df.stitched_continuity.mean()),
            "cls_arc_f1": float(df.cls_arc_f1.mean()), "hmm_arc_f1": float(df.hmm_arc_f1.mean()),
            "decode_seconds_total": float(df.decode_seconds.sum()),
            "decode_ms_per_1000_obs": float(df.decode_seconds.sum() / w.sum() * 1e6)}
