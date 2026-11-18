"""EDA plots, missing-value analysis, Folium EDA maps -> figures/eda, maps/ (state the partition in every title)

OWNER: M1.  Run AFTER 00_inventory.py and 02_build_dataset.py:
    python scripts/01_eda.py [--map-trips train:00000010 train:00000033 student_test:00000000]
All figures use only the train and student_test partitions (the evaluator partition is not available to us).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.io import load_trip, load_route, list_trip_ids  # noqa: E402
from pathsnap.geo import LocalProjector  # noqa: E402
from pathsnap.graph import build_graph  # noqa: E402
from pathsnap.dataset import load_table  # noqa: E402

FIG = ROOT / "figures/eda"
SPLIT_COLORS = {"train": "#1f6fd1", "student_test": "#f28e1c"}


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", FIG / name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=str(ROOT / "data/raw/Student folder"))
    ap.add_argument("--map-trips", nargs="*",
                    default=["train:00000010", "train:00000033", "student_test:00000000"])
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    (ROOT / "maps").mkdir(exist_ok=True)
    inv = pd.read_csv(ROOT / "results/eda/inventory.csv", dtype={"trip_id": str})
    proc = ROOT / "data/processed"
    obs = pd.concat([load_table(proc / f"obs_labels_{s}.parquet") for s in ("train", "student_test")])
    cands = {s: load_table(proc / f"candidates_{s}.parquet") for s in ("train", "student_test")}

    # 1. observations per trip
    fig, ax = plt.subplots(figsize=(7, 4))
    for s, d in inv.groupby("split"):
        ax.hist(d.n_obs, bins=np.logspace(2, 4.5, 25), alpha=.7, label=f"{s} ({len(d)} trips)", color=SPLIT_COLORS[s])
    ax.set(xscale="log", xlabel="GPS observations per trip", ylabel="trips",
           title="Trip length  [train + student_test]")
    ax.legend()
    savefig(fig, "01_obs_per_trip.png")

    # 2. sampling intervals
    dts = {s: [] for s in SPLIT_COLORS}
    for _, r in inv.iterrows():
        tr = load_trip(args.data_root, r.split, r.trip_id)
        dts[r.split].append(np.diff(tr.track[:, 2]))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for s, v in dts.items():
        v = np.concatenate(v)
        vals, counts = np.unique(np.clip(v, 0, 11), return_counts=True)
        ax[0].bar(vals + (0.2 if s == "student_test" else -0.2), counts / counts.sum() * 100, width=.4,
                  label=s, color=SPLIT_COLORS[s])
        big = np.sort(v[v > 10])
        ax[1].step(big, np.arange(len(big), 0, -1), where="post", label=f"{s}: {len(big)} gaps > 10 s",
                   color=SPLIT_COLORS[s])
    ax[0].set(xlabel="time step between consecutive fixes (s, 11 = '>10')", ylabel="% of steps", yscale="log",
              title="Sampling interval  [train + student_test]")
    ax[1].set(xscale="log", yscale="log", xlabel="gap length (s)", ylabel="# gaps >= x",
              title="Large GPS gaps (> 10 s)  [train + student_test]")
    for a in ax:
        a.legend()
    savefig(fig, "02_sampling_intervals.png")

    # 3. missing values / data-quality table
    mv = inv.groupby("split")[["track_nan", "nodes_nan", "dup_points", "dup_timestamps", "t_backwards",
                               "n_gaps_gt_60s", "n_jumps_gt_70mps", "zero_length_arcs", "self_loop_arcs",
                               "isolated_nodes"]].sum()
    zc = obs.groupby("split").apply(lambda d: int((d.n_candidates == 0).sum()))
    mv["obs_with_zero_candidates"] = zc
    mv["files_missing"] = inv.groupby("split")[["has_track", "has_nodes", "has_arcs", "has_route"]] \
        .apply(lambda d: int((~d).sum().sum()))
    mv.T.to_csv(ROOT / "results/eda/missing_values.csv")
    fig, ax = plt.subplots(figsize=(9, 4))
    mv.T.plot.barh(ax=ax, color=[SPLIT_COLORS[c] for c in mv.index])
    ax.legend(loc="center right")
    ax.set(xscale="symlog", xlabel="count (symlog)", title="Missing values & data-quality issues  [train + student_test]")
    for i, (a, b) in enumerate(zip(mv.T.iloc[:, 0], mv.T.iloc[:, 1])):
        ax.text(max(a, b) + .5, i, f"{mv.index[0]}={int(a)}, {mv.index[1]}={int(b)}", va="center", fontsize=7)
    savefig(fig, "03_missing_values.png")

    # 4. GPS-to-derived-label distance per trip
    tr_obs = obs[obs.split == "train"]
    q = tr_obs.groupby("trip_id").label_dist_m.quantile([.5, .95]).unstack()
    q = q.sort_values(0.5)
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(q))
    ax.bar(x, q[0.95], color="#c9d7ee", label="p95")
    ax.bar(x, q[0.5], color="#1f6fd1", label="median")
    ax.axhline(50, color="r", ls="--", lw=1, label="50 m reliability threshold")
    for i, (tid, r) in enumerate(q.iterrows()):
        if r[0.5] > 50 or r[0.95] > 100:
            ax.text(i, r[0.95], tid[-2:], ha="center", va="bottom", fontsize=8, color="r")
    ax.set(yscale="log", xticks=[], xlabel="train trips (sorted by median)", ylabel="distance GPS -> derived label arc (m)",
           title="How far is each GPS fix from its derived reference arc?  [train]")
    ax.legend()
    savefig(fig, "04_label_distance_per_trip.png")

    # 5. road network size & connectivity
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for s, d in inv.groupby("split"):
        ax[0].scatter(d.n_nodes, d.n_arcs, s=14, label=s, color=SPLIT_COLORS[s])
        ax[1].scatter(d.n_nodes, d.n_weak_components, s=14, label=s, color=SPLIT_COLORS[s])
        ax[2].hist(d.twin_frac * 100, bins=15, alpha=.7, label=s, color=SPLIT_COLORS[s])
    ax[0].set(xscale="log", yscale="log", xlabel="nodes", ylabel="arcs", title="Network size per trip")
    ax[1].set(xscale="log", yscale="log", xlabel="nodes", ylabel="weakly connected components",
              title="Network fragmentation per trip")
    ax[2].set(xlabel="% arcs with a reverse twin (two-way roads)", ylabel="trips", title="Two-way roads")
    for a in ax:
        a.legend(fontsize=8)
    fig.suptitle("Road-network structure  [train + student_test]", y=1.02)
    savefig(fig, "05_network_stats.png")

    # 6. candidate set
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for s, c in cands.items():
        nc = obs[obs.split == s].n_candidates
        ax[0].hist(nc, bins=np.arange(0, 32), alpha=.6, density=True, label=s, color=SPLIT_COLORS[s])
        pos = c[c.label == 1]
        ax[1].hist(pos["rank"], bins=np.arange(0, 31), alpha=.6, density=True, label=s, color=SPLIT_COLORS[s])
        ax[2].hist(pos["dist_m"], bins=np.linspace(0, 60, 61), alpha=.6, density=True, label=s, color=SPLIT_COLORS[s])
    ax[0].set(xlabel="candidates per GPS point", ylabel="density", title="Candidate-set size")
    ax[1].set(xlabel="distance rank of the true arc (0 = nearest)", yscale="log", title="Where is the true arc?")
    ax[2].set(xlabel="distance GPS -> true arc (m)", title="GPS noise vs true arc")
    for a in ax:
        a.legend()
    fig.suptitle("Candidate generation (adaptive radius, K = 30)  [train + student_test]", y=1.02)
    savefig(fig, "06_candidates.png")

    # 7. trajectories over the network (metres, local projection)
    pick = [("train", "00000010"), ("train", "00000033"), ("train", "00000054"),
            ("train", "00000032"), ("student_test", "00000000"), ("student_test", "00000077")]
    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (s, tid) in zip(axs.ravel(), pick):
        tr = load_trip(args.data_root, s, tid)
        pr = LocalProjector.from_track(tr.track)
        g = build_graph(tr, pr)
        xy = pr.forward(tr.track[:, 0], tr.track[:, 1])
        lo, hi = xy.min(0) - 300, xy.max(0) + 300
        mid = (g.a_xy + g.b_xy) / 2
        ins = np.all((mid >= lo) & (mid <= hi), 1)
        A, B = g.a_xy[ins], g.b_xy[ins]
        ax.add_collection(LineCollection(np.stack([A, B], 1), colors="#bbbbbb", linewidths=.4))
        r = load_route(args.data_root, s, tid)
        ax.add_collection(LineCollection(np.stack([g.a_xy[r], g.b_xy[r]], 1), colors="#1a9e3a", linewidths=2.5))
        ax.scatter(xy[:, 0], xy[:, 1], s=1.5, color="#1f6fd1", zorder=3)
        ax.set(aspect="equal", title=f"{s} {tid}  ({tr.n_obs} fixes)", xlabel="x (m)", ylabel="y (m)")
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    axs[0, 0].plot([], [], color="#bbbbbb", label="road network")
    axs[0, 0].plot([], [], color="#1a9e3a", lw=2.5, label="reference route")
    axs[0, 0].scatter([], [], s=6, color="#1f6fd1", label="GPS fixes")
    axs[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle("GPS trajectories vs road network and reference route (local metric projection)", y=1.0)
    savefig(fig, "07_trajectories.png")

    # 8. speed distribution
    sp = []
    for _, r in inv.iterrows():
        tr = load_trip(args.data_root, r.split, r.trip_id)
        xy = LocalProjector.from_track(tr.track).forward(tr.track[:, 0], tr.track[:, 1])
        v = np.sqrt((np.diff(xy, axis=0) ** 2).sum(1)) / np.maximum(np.diff(tr.track[:, 2]), 1e-6)
        sp.append(pd.DataFrame({"split": r.split, "v": v * 3.6}))
    sp = pd.concat(sp)
    fig, ax = plt.subplots(figsize=(7, 4))
    for s, d in sp.groupby("split"):
        ax.hist(d.v.clip(0, 300), bins=100, density=True, alpha=.6, label=s, color=SPLIT_COLORS[s])
    ax.set(xlabel="implied speed between consecutive fixes (km/h, clipped at 300)", ylabel="density", yscale="log",
           title="Vehicle speed  [train + student_test]")
    ax.legend()
    savefig(fig, "08_speed.png")

    # 9. Folium maps
    try:
        from pathsnap.viz.folium_maps import build_trip_map
        for spec in args.map_trips:
            s, tid = spec.split(":")
            tr = load_trip(args.data_root, s, tid)
            o = obs[(obs.split == s) & (obs.trip_id == tid)].sort_values("obs_idx")
            m = build_trip_map(tr, reference=load_route(args.data_root, s, tid),
                               title=f"EDA - {s} trip {tid}: raw GPS, road network, reference route")
            out = ROOT / "maps" / f"eda_{s}_{tid}.html"
            m.save(str(out))
            print("  wrote", out, f"(label median dist {o.label_dist_m.median():.1f} m)")
    except ImportError:
        print("  [skip] folium not installed - pip install folium, then re-run for the maps")

    print(json.dumps({"figures": sorted(p.name for p in FIG.glob("*.png"))}, indent=1))


if __name__ == "__main__":
    main()
