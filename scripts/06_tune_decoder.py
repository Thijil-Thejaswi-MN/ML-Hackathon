"""Tune HMM parameters on train OOF probs, confirm on student-test -> results/hmm/decoder_tuning.csv, artifacts/hmm_config.json

OWNER: M3.  Run after the chosen model has OOF probabilities (03_train_model.py / 05_bayes_opt.py):
    python scripts/06_tune_decoder.py --model lightgbm_tuned
Grid search over beta (transition scale) and the emission GPS-distance weight, scored by pooled Top-1 on
RELIABLE train observations decoded with out-of-fold probabilities (never student_test). The winning config is
written to artifacts/hmm_config.json (what the API loads); student_test is decoded once with it as confirmation.
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import pandas as pd
import yaml

from hmm_eval_common import ROOT, TripCache, evaluate_decoder, load_probs, pooled

GRID = {"beta_m": [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0], "emission_distance_weight": [0.0, 1.0]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="results tag whose OOF / student-test probs to decode")
    args = ap.parse_args()
    out = ROOT / "results/hmm"
    out.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load((ROOT / "configs/hmm.yaml").read_text())
    t_all = time.time()

    cache = TripCache("train", load_probs("train", args.model))
    rows = []
    for vals in itertools.product(*GRID.values()):
        cfg = {**base, **dict(zip(GRID, vals))}
        t0 = time.time()
        r = pooled(evaluate_decoder(cache, cfg))
        rows.append({**dict(zip(GRID, vals)), **r, "seconds": round(time.time() - t0, 1)})
        print(f"  {dict(zip(GRID, vals))}: hmm_top1_reliable={r['hmm_top1_reliable']:.4f} "
              f"(classifier-only {r['cls_top1_reliable']:.4f}) continuity={r['hmm_continuity']:.3f}", flush=True)
    grid = pd.DataFrame(rows).sort_values("hmm_top1_reliable", ascending=False)
    grid.insert(0, "partition", "train OOF (80 trips)")
    grid.to_csv(out / "decoder_tuning.csv", index=False)

    best = {**base, **{k: grid.iloc[0][k] for k in GRID}}
    best = {k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v) for k, v in best.items()}
    confirm = pooled(evaluate_decoder(TripCache("student_test", load_probs("student_test", args.model)), best))
    (ROOT / "artifacts/hmm_config.json").write_text(json.dumps(best, indent=1))
    (out / "decoder_tuning_summary.json").write_text(json.dumps({
        "model_probs": args.model, "grid": GRID, "objective": "pooled Top-1 on reliable train observations (OOF probs)",
        "best_config": best, "best_train_oof": grid.iloc[0].drop("partition").to_dict(),
        "student_test_confirmation": confirm, "wall_clock_s": round(time.time() - t_all, 1)}, indent=1, default=str))
    print(f"[best] {({k: best[k] for k in GRID})}  train OOF hmm_top1_reliable={grid.iloc[0]['hmm_top1_reliable']:.4f}")
    print(f"[student_test] classifier-only top1={confirm['cls_top1']:.4f}  HMM top1={confirm['hmm_top1']:.4f}  "
          f"continuity {confirm['cls_continuity']:.3f} -> {confirm['hmm_continuity']:.3f} (stitched {confirm['stitched_continuity']:.3f})")
    print("wrote results/hmm/decoder_tuning.csv, decoder_tuning_summary.json, artifacts/hmm_config.json")


if __name__ == "__main__":
    main()
