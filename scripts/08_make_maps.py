"""Generate Folium HTML maps for chosen trips -> maps/

OWNER: M3.  Run after 06_tune_decoder.py:
    python scripts/08_make_maps.py --model lightgbm_tuned [--trip 00000077 00000056 ...] [--split student_test]
Each map has layers: raw GPS, road network, reference route (offline evidence only), classifier-only route,
HMM / Viterbi route, with legend and layer control. Default trips: best, median and worst HMM trip.
"""
import argparse
import json
import sys

import numpy as np

from hmm_eval_common import ROOT, TripCache, evaluate_decoder, load_probs

sys.path.insert(0, str(ROOT / "src"))
from pathsnap.viz.folium_maps import build_trip_map  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="student_test")
    ap.add_argument("--trip", nargs="*", help="trip ids (default: best, median, worst HMM Top-1)")
    args = ap.parse_args()
    cfg = json.loads((ROOT / "artifacts/hmm_config.json").read_text())
    cands = load_probs(args.split, args.model)
    cache = TripCache(args.split, cands, trip_ids=set(args.trip) if args.trip else None)
    df = evaluate_decoder(cache, cfg, keep_arcs=True).sort_values("hmm_top1")
    if not args.trip:
        df = df.iloc[[0, len(df) // 2, len(df) - 1]]
    (ROOT / "maps").mkdir(exist_ok=True)
    for _, r in df.iterrows():
        it = next(i for i in cache.items if i["trip_id"] == r.trip_id)
        m = build_trip_map(it["trip"], reference=it["route"], classifier_route=it["cls"], hmm_route=r["_route_arcs"],
                           title=f"{args.split} trip {r.trip_id} | Top-1 classifier {r.cls_top1:.3f} -> HMM "
                                 f"{r.hmm_top1:.3f} | continuity {r.cls_continuity:.2f} -> {r.stitched_continuity:.2f}")
        path = ROOT / "maps" / f"hmm_{args.split}_{r.trip_id}.html"
        m.save(str(path))
        print("wrote", path)


if __name__ == "__main__":
    main()
