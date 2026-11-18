"""Candidate miss rate vs (radius schedule, K) on the TRAIN partition -> results/data/candidate_sweep.csv + plot.

OWNER: M1.  Evidence for choosing configs/global.yaml:candidates (README 3.4). Run before 02_build_dataset.py.
    python scripts/02a_candidate_sweep.py
Miss rate is reported on observations whose derived label is reliable (<= 50 m from the GPS fix), because
for the offset trips (33, 54) no candidate radius can recover a label that is 120+ m away.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.io import list_trip_ids, load_trip, load_route  # noqa: E402
from pathsnap.geo import LocalProjector  # noqa: E402
from pathsnap.graph import build_graph  # noqa: E402
from pathsnap.candidates import ArcIndex, generate_candidates  # noqa: E402
from pathsnap.labels import derive_obs_labels_full  # noqa: E402

SCHEDULES = {"fixed30": [30], "fixed50": [50], "fixed100": [100], "adaptive50-100-200": [50, 100, 200],
             "adaptive30-60-120-200": [30, 60, 120, 200], "fixed200": [200]}
KS = [3, 5, 8, 10, 12, 16, 20, 25, 30, 40]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=str(ROOT / "data/raw/Student folder"))
    ap.add_argument("--k-min", type=int, default=3)
    args = ap.parse_args()
    rows, t0 = [], time.time()
    trips = []
    for tid in list_trip_ids(args.data_root, "train"):
        tr = load_trip(args.data_root, "train", tid)
        pr = LocalProjector.from_track(tr.track)
        g = build_graph(tr, pr)
        lab, _, dist, _ = derive_obs_labels_full(tr, load_route(args.data_root, "train", tid), pr, g)
        trips.append((tr, pr, g, ArcIndex(g), lab, dist <= 50.0))
    for name, radii in SCHEDULES.items():
        ts = time.time()
        true_rank, ncand = [], []
        for tr, pr, g, idx, lab, rel in trips:
            c = generate_candidates(tr, g, pr, {"radii_m": radii, "k_max": max(KS), "k_min": args.k_min}, idx)
            rk = np.full(tr.n_obs, 10 ** 6)
            hit = c[c["arc_id"].to_numpy() == lab[c["obs_idx"].to_numpy()]]
            rk[hit["obs_idx"].to_numpy()] = hit["rank"].to_numpy()
            true_rank.append(rk[rel])
            ncand.append(np.bincount(c["obs_idx"], minlength=tr.n_obs)[rel])
        true_rank, ncand = np.concatenate(true_rank), np.concatenate(ncand)
        for k in KS:
            rows.append({"schedule": name, "radii_m": str(radii), "k_max": k,
                         "miss_rate": float((true_rank >= k).mean()),
                         "mean_candidates": float(np.minimum(ncand, k).mean()),
                         "n_obs_reliable": int(len(true_rank)), "seconds": round(time.time() - ts, 1)})
        print(f"{name}: miss@12={(true_rank >= 12).mean():.4f} miss@20={(true_rank >= 20).mean():.4f}", flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "results/data"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "candidate_sweep.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, d in df.groupby("schedule", sort=False):
        ax[0].plot(d.k_max, d.miss_rate * 100, marker="o", label=name)
        ax[1].plot(d.mean_candidates, d.miss_rate * 100, marker="o", label=name)
    ax[0].set(xlabel="K (max candidates per GPS point)", ylabel="candidate miss rate (%)", yscale="log",
              title="Candidate miss rate vs K  [train, 80 trips]")
    ax[1].set(xlabel="mean candidates per GPS point (cost)", ylabel="candidate miss rate (%)", yscale="log",
              title="Miss rate vs candidate-set size  [train]")
    ax[0].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(ROOT / "results/data/candidate_sweep.png", dpi=130)
    print(f"[done] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
