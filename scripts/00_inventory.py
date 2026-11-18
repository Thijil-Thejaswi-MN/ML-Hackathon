"""Dataset inventory + data-quality table -> results/eda/inventory.csv, network_stats.csv (README 1.3 #1-3)

OWNER: M1.  Run:  python scripts/00_inventory.py
One row per trip (train + student_test): file availability and sizes, observation / node / arc / route counts,
dtypes, NaNs, duplicates, timestamp ordering and gaps, geographic extent, network connectivity, route connectivity,
and a projection check (local metres vs haversine).
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
from pathsnap.io import list_trip_ids, load_trip, load_route, trip_dir  # noqa: E402
from pathsnap.geo import LocalProjector, haversine_m  # noqa: E402
from pathsnap.graph import build_graph, component_stats  # noqa: E402

EXT = ["track", "nodes", "arcs", "route"]


def trip_row(root, split, tid):
    d = trip_dir(root, split, tid)
    row = {"trip_id": tid, "split": split}
    for e in EXT:
        f = d / f"{tid}.{e}"
        row[f"has_{e}"] = f.exists()
        row[f"{e}_bytes"] = f.stat().st_size if f.exists() else 0
    raw = (d / f"{tid}.track").read_bytes()
    row["crlf_line_endings"] = b"\r\n" in raw[:2000]
    tr = load_trip(root, split, tid)
    lon, lat, t = tr.track[:, 0], tr.track[:, 1], tr.track[:, 2]
    dt = np.diff(t)
    row.update({
        "n_obs": tr.n_obs, "n_nodes": len(tr.nodes), "n_arcs": len(tr.arcs),
        "track_nan": int(np.isnan(tr.track).sum()), "nodes_nan": int(np.isnan(tr.nodes).sum()),
        "dup_points": int(pd.DataFrame(tr.track[:, :2]).duplicated().sum()),
        "dup_timestamps": int((dt == 0).sum()),
        "t_monotonic": bool(np.all(dt > 0)), "t_backwards": int((dt < 0).sum()),
        "t_start": float(t[0]), "duration_s": float(t[-1] - t[0]),
        "frac_dt_eq_1s": float(np.mean(np.isclose(dt, 1.0))) if len(dt) else np.nan,
        "dt_median": float(np.median(dt)), "dt_max": float(dt.max()),
        "n_gaps_gt_10s": int((dt > 10).sum()), "n_gaps_gt_60s": int((dt > 60).sum()),
        "lon_min": lon.min(), "lon_max": lon.max(), "lat_min": lat.min(), "lat_max": lat.max(),
        "arc_from_out_of_range": int((tr.arcs >= len(tr.nodes)).sum() + (tr.arcs < 0).sum()),
        "self_loop_arcs": int((tr.arcs[:, 0] == tr.arcs[:, 1]).sum()),
    })
    pr = LocalProjector.from_track(tr.track)
    xy = pr.forward(lon, lat)
    ext = xy.max(0) - xy.min(0)
    step = np.sqrt((np.diff(xy, axis=0) ** 2).sum(1))
    hv = haversine_m(lon[:-1], lat[:-1], lon[1:], lat[1:])
    speed = step / np.maximum(dt, 1e-6)
    row.update({"extent_x_km": ext[0] / 1000, "extent_y_km": ext[1] / 1000, "path_length_km": step.sum() / 1000,
                "speed_p50_mps": float(np.median(speed)), "speed_p99_mps": float(np.percentile(speed, 99)),
                "n_jumps_gt_70mps": int((speed > 70).sum()),
                "proj_check_max_abs_err_m": float(np.abs(step - hv).max()) if len(hv) else 0.0})
    g = build_graph(tr, pr)
    row.update(component_stats(g))
    row["arc_len_median_m"] = float(np.median(g.arc_len))
    row["zero_length_arcs"] = int((g.arc_len < 0.01).sum())
    if row["has_route"]:
        r = load_route(root, split, tid)
        conn = tr.arcs[r[:-1], 1] == tr.arcs[r[1:], 0]
        row.update({"n_route_arcs": len(r), "route_connected_frac": float(conn.mean()) if len(conn) else 1.0,
                    "route_unique_arcs": int(len(np.unique(r))), "route_repeat_arcs": int(len(r) - len(np.unique(r))),
                    "route_length_km": float(g.arc_len[r].sum() / 1000)})
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=str(ROOT / "data/raw/Student folder"))
    args = ap.parse_args()
    t0 = time.time()
    rows = []
    for split in ("train", "student_test"):
        for tid in list_trip_ids(args.data_root, split):
            rows.append(trip_row(args.data_root, split, tid))
    inv = pd.DataFrame(rows)
    out = ROOT / "results/eda"
    out.mkdir(parents=True, exist_ok=True)
    inv.to_csv(out / "inventory.csv", index=False)
    net_cols = ["trip_id", "split", "n_nodes", "n_arcs", "n_weak_components", "n_strong_components",
                "largest_weak_component_frac", "isolated_nodes", "twin_frac", "arc_len_median_m", "zero_length_arcs",
                "self_loop_arcs", "route_connected_frac"]
    inv[net_cols].to_csv(out / "network_stats.csv", index=False)
    num = inv.select_dtypes("number").columns
    summ = inv.groupby("split")[list(num)].agg(["min", "median", "max", "sum"]).T
    summ.to_csv(out / "inventory_summary.csv")
    flags = inv[(inv.n_gaps_gt_60s > 0) | (~inv.t_monotonic) | (inv.n_jumps_gt_70mps > 0) | (inv.dup_points > 0)]
    flags[["trip_id", "split", "n_obs", "dt_max", "n_gaps_gt_60s", "t_backwards", "dup_points", "n_jumps_gt_70mps"]] \
        .to_csv(out / "anomalies.csv", index=False)
    print(inv.groupby("split").agg(trips=("trip_id", "count"), obs=("n_obs", "sum"), obs_min=("n_obs", "min"),
                                    obs_max=("n_obs", "max"), nodes_max=("n_nodes", "max"), arcs_max=("n_arcs", "max"),
                                    nan=("track_nan", "sum"), dups=("dup_points", "sum"), nonmono=("t_backwards", "sum"),
                                    dt_max=("dt_max", "max"), route_conn_min=("route_connected_frac", "min"),
                                    proj_err_max=("proj_check_max_abs_err_m", "max")).T)
    print(f"anomalous trips: {len(flags)}  -> results/eda/anomalies.csv   [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
