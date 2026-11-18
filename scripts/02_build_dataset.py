"""Labels + candidates + features + frozen folds; fit scaler on TRAIN only -> data/processed, artifacts/

OWNER: M1.  Run:  python scripts/02_build_dataset.py --config configs/global.yaml [--mini 5]

Outputs
  data/processed/candidates_{train,student_test}.parquet   one row per (obs, candidate arc) + features + label
  data/processed/obs_labels_{train,student_test}.parquet   one row per GPS observation (derived label, confidence)
  data/processed/folds.json                                frozen 5-fold group split of the 80 train trips
  artifacts/features.json, artifacts/candidate_config.json, artifacts/scaler.joblib
  results/data/trip_stats_{split}.csv, results/data/candidate_sweep.csv, results/data/build_log.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.dataset import build_split, make_group_folds, save_table, process_trip  # noqa: E402
from pathsnap.features import FEATURE_COLUMNS  # noqa: E402
from pathsnap.candidates import DEFAULT_CAND_CFG  # noqa: E402


def load_cfg(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs/global.yaml"))
    ap.add_argument("--mini", type=int, default=0, help="only the first N trips per split (quick G1a dataset)")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    import copy
    cfg_as_written = copy.deepcopy(cfg)
    for k in ("raw_root", "processed", "artifacts", "results"):
        p = Path(cfg["paths"][k])
        cfg["paths"][k] = str(p if p.is_absolute() else ROOT / p)
    proc, art, res = (Path(cfg["paths"][k]) for k in ("processed", "artifacts", "results"))
    (res / "data").mkdir(parents=True, exist_ok=True)
    art.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    if args.mini:
        import pathsnap.dataset as ds
        orig = ds.list_trip_ids
        ds.list_trip_ids = lambda root, split: orig(root, split)[: args.mini]

    out = {}
    for split in ("train", "student_test"):
        print(f"[build] {split}")
        cands, obs, stats = build_split(split, cfg)
        save_table(cands, proc / f"candidates_{split}.parquet")
        save_table(obs, proc / f"obs_labels_{split}.parquet")
        stats.to_csv(res / "data" / f"trip_stats_{split}.csv", index=False)
        out[split] = (cands, obs, stats)

    # frozen group folds over TRAIN trips only
    sizes = out["train"][2].set_index("trip_id")["n_obs"].to_dict()
    folds = make_group_folds(sizes, cfg["cv"]["n_splits"], cfg["seed"])
    (proc / "folds.json").write_text(json.dumps(folds, indent=1))

    # scaler fit on TRAIN rows only (README R4)
    from sklearn.preprocessing import StandardScaler
    import joblib
    scaler = StandardScaler().fit(out["train"][0][FEATURE_COLUMNS].to_numpy())
    joblib.dump(scaler, art / "scaler.joblib")
    (art / "features.json").write_text(json.dumps(FEATURE_COLUMNS, indent=1))
    cand_cfg = {**DEFAULT_CAND_CFG, **cfg.get("candidates", {})}
    (art / "candidate_config.json").write_text(json.dumps(cand_cfg, indent=1))

    summary = {}
    for split, (c, o, s) in out.items():
        summary[split] = {
            "trips": int(len(s)), "observations": int(len(o)), "candidate_rows": int(len(c)),
            "candidates_per_obs": round(len(c) / len(o), 2),
            "candidate_miss_rate": round(float(1 - o["cand_hit"].mean()), 4),
            "candidate_miss_rate_reliable_labels": round(float(1 - o.loc[o.label_reliable, "cand_hit"].mean()), 4),
            "nearest_arc_top1": round(float(s["nearest_arc_top1"].mul(s["n_obs"]).sum() / s["n_obs"].sum()), 4),
            "label_reliable_frac": round(float(o["label_reliable"].mean()), 4),
            "positive_rate": round(float(c["label"].mean()), 4),
        }
    log = {"config": cfg_as_written, "mini": args.mini, "wall_clock_s": round(time.time() - t_start, 1), "summary": summary,
           "n_features": len(FEATURE_COLUMNS)}
    (res / "data" / "build_log.json").write_text(json.dumps(log, indent=1, default=str))
    print(json.dumps(summary, indent=1))
    print(f"[done] {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
