"""Offline dataset assembly (train / student_test).  OWNER: M1.

Builds candidates + features + labels, writes data/processed/*.parquet, folds.json (frozen group folds),
fits the scaler on TRAIN only.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .io import list_trip_ids, load_trip, load_route
from .geo import LocalProjector
from .graph import build_graph
from .candidates import ArcIndex, generate_candidates, DEFAULT_CAND_CFG
from .features import build_features, FEATURE_COLUMNS, DEFAULT_FEAT_CFG
from .labels import derive_obs_labels_full, nearest_route_arc

LABEL_RELIABLE_M = 50.0   # obs whose derived label lies > 50 m from the GPS fix are flagged unreliable


def save_table(df: pd.DataFrame, path: Path) -> Path:
    """Parquet when pyarrow/fastparquet is installed (normal case), pickle otherwise."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
        return path
    except ImportError:
        alt = path.with_suffix(".pkl")
        df.to_pickle(alt)
        print(f"[warn] no parquet engine; wrote {alt}")
        return alt


def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.exists():
        return pd.read_parquet(path)
    return pd.read_pickle(path.with_suffix(".pkl"))


def process_trip(root, split, trip_id, cand_cfg, feat_cfg, with_labels=True):
    """Candidates + features (+ labels) for one trip. Returns (cand_df, obs_df, trip_stats)."""
    t0 = time.time()
    trip = load_trip(root, split, trip_id)
    proj = LocalProjector.from_track(trip.track)
    graph = build_graph(trip, proj)
    index = ArcIndex(graph, cand_cfg.get("sample_step_m", 10.0))
    cands = generate_candidates(trip, graph, proj, cand_cfg, index)
    feats = build_features(trip, graph, proj, cands, feat_cfg)
    feats.insert(1, "split", split)
    n = trip.n_obs
    obs = pd.DataFrame({"trip_id": trip_id, "split": split, "obs_idx": np.arange(n, dtype=np.int32),
                        "n_candidates": np.bincount(cands["obs_idx"], minlength=n).astype(np.int16)})
    stats = {"trip_id": trip_id, "split": split, "n_obs": n, "n_cand_rows": len(feats)}
    if with_labels:
        route = load_route(root, split, trip_id)
        arc, conf, dist, pos = derive_obs_labels_full(trip, route, proj, graph)
        nn = nearest_route_arc(trip, route, proj)
        obs["true_arc_id"] = arc.astype(np.int32)
        obs["label_conf"] = conf.astype(np.float32)
        obs["label_dist_m"] = dist.astype(np.float32)
        obs["route_pos"] = pos.astype(np.int32)
        obs["label_reliable"] = (dist <= LABEL_RELIABLE_M)
        feats["label"] = (feats["arc_id"].to_numpy() == arc[feats["obs_idx"].to_numpy()]).astype(np.int8)
        hit = np.zeros(n, dtype=bool)
        hit[feats.loc[feats["label"] == 1, "obs_idx"].to_numpy()] = True
        obs["cand_hit"] = hit
        r0 = feats["rank"] == 0
        nearest_ok = np.zeros(n, dtype=bool)
        nearest_ok[feats.loc[r0 & (feats["label"] == 1), "obs_idx"].to_numpy()] = True
        stats.update({
            "n_route_arcs": len(route),
            "label_median_dist_m": float(np.median(dist)),
            "label_p95_dist_m": float(np.percentile(dist, 95)),
            "label_reliable_frac": float((dist <= LABEL_RELIABLE_M).mean()),
            "label_agree_nearest_route_arc": float((nn == arc).mean()),
            "route_arcs_used_frac": float(len(np.unique(pos)) / len(route)),
            "candidate_miss_rate": float(1 - hit.mean()),
            "nearest_arc_top1": float(nearest_ok.mean()),
        })
    stats["seconds"] = round(time.time() - t0, 2)
    return feats, obs, stats


def build_split(split: str, cfg: dict):
    root = cfg["paths"]["raw_root"]
    cand_cfg = {**DEFAULT_CAND_CFG, **cfg.get("candidates", {})}
    feat_cfg = {**DEFAULT_FEAT_CFG, **cfg.get("features", {})}
    cands, obs, stats = [], [], []
    for tid in list_trip_ids(root, split):
        f, o, s = process_trip(root, split, tid, cand_cfg, feat_cfg, with_labels=True)
        cands.append(f), obs.append(o), stats.append(s)
        print(f"  {split} {tid}: obs={s['n_obs']} rows={s['n_cand_rows']} miss={s.get('candidate_miss_rate', 0):.3f} "
              f"label_med={s.get('label_median_dist_m', 0):.1f}m {s['seconds']}s", flush=True)
    return pd.concat(cands, ignore_index=True), pd.concat(obs, ignore_index=True), pd.DataFrame(stats)


def make_group_folds(trip_sizes: dict, n_splits: int, seed: int) -> dict:
    """Group-aware folds by trip, balanced by #observations (greedy largest-first into the smallest fold).
    Ties are broken by a seeded shuffle so the assignment is reproducible. Write once; everybody reuses folds.json."""
    rng = np.random.default_rng(seed)
    ids = list(trip_sizes)
    rng.shuffle(ids)
    ids.sort(key=lambda t: -trip_sizes[t])
    load = np.zeros(n_splits)
    folds = {}
    for t in ids:
        k = int(np.argmin(load))
        folds[t] = k
        load[k] += trip_sizes[t]
    return {"n_splits": n_splits, "seed": seed, "fold_of_trip": dict(sorted(folds.items())),
            "obs_per_fold": load.astype(int).tolist()}
