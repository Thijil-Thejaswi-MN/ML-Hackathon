# Data notes, quality issues and decisions  (owner: M1)

Every number below comes from our own scripts. Re-run them to reproduce:

```
python scripts/00_inventory.py          # -> results/eda/inventory.csv, network_stats.csv, anomalies.csv, inventory_summary.csv
python scripts/02a_candidate_sweep.py   # -> results/data/candidate_sweep.csv/.png
python scripts/02_build_dataset.py      # -> data/processed/*, artifacts/*, results/data/trip_stats_*.csv, build_log.json
python scripts/01_eda.py                # -> figures/eda/*.png, results/eda/missing_values.csv, maps/eda_*.html
```

Partitions used: **train (80 trips)** and **student_test (10 trips)** only. The evaluator partition is not available to us.

## 1. Inventory  (`results/eda/inventory.csv`)

| | train | student_test |
|---|---|---|
| trips | 80 | 10 |
| GPS observations | 208,417 (361 – 19,313 per trip, median 1,529) | 19,160 (262 – 3,958, median 1,420) |
| road nodes per trip | 299 – 155,003 (median 9,608) | 1,770 – 26,635 |
| road arcs per trip | 649 – 307,392 (median 19,527) | 3,577 – 56,027 |
| reference-route arcs per trip | 59 – 2,266 (median 280) | 95 – 1,323 |
| files missing | 0 | 0 |

- All four files exist for every trip. They are whitespace-separated, have no header and use CRLF line endings, and `np.loadtxt` reads them correctly.
- Trips are spread across the world (lon −100 … 137, lat −35 … 62). Each trip ships its **own local road network**, so there is no single global CRS. See section 3.
- Ordering checks pass: `.track` is (lon, lat, t), `.nodes` is (lon, lat), and all values are in valid lon/lat ranges.

## 2. Data quality  (`results/eda/missing_values.csv`, `anomalies.csv`, figure `03_missing_values.png`)

| Issue | train | student_test | Decision |
|---|---|---|---|
| NaN values (track / nodes) | 0 / 0 | 0 / 0 | nothing to impute |
| duplicated GPS points (same lon/lat) | 572 | 6 | kept; they are real stops. Heading is unreliable when stopped, so `head_valid` / `stationary` features handle it |
| duplicated or backwards timestamps | 0 / 0 | 0 / 0 | timestamps strictly increasing in every trip |
| steps exactly 1 s | median 99.1 % of steps per trip (min 53 %) | median 99.9 % (min 86 %) | nominal 1 Hz confirmed; irregular trips exist |
| gaps > 60 s | 69 (trips 78: 52,810 s, 90: 153,064 s, 69: 2,658 s, 21: 2,161 s …) | 1 (max 210 s) | `dt_prev` / `dt_next` features; HMM must break the chain at large gaps (M3) |
| implied speed > 70 m/s (252 km/h) | 149 steps | 1 | GPS glitches. Most trips with fast steps are just highways (median trip speed ≈ 41 m/s in trips 1, 32), so we do **not** delete points; speed is clipped at 80 m/s in features |
| zero-length arcs / self-loop arcs | 100 / 22 | 6 / 2 | kept; harmless for candidates (distance to a point) |
| isolated nodes | 0 | 0 | — |
| GPS points with zero candidates (after adaptive 200 m) | 108 | 0 | kept in `obs_labels` with `n_candidates = 0`; HMM must handle them |

**Road network** (`network_stats.csv`, figure `05_network_stats.png`):
- 35–100 % of arcs have a reverse twin (median 95 %). Two-way roads are stored as two directed arcs with identical geometry, so **only heading can separate them**. That is why the `cos_head*` features exist.
- Networks are fragmented: up to 421 weakly connected components (median 20). The HMM must allow for "no path" transitions.
- Consecutive reference-route arcs connect head-to-tail 98.1–100 % of the time. The route is *mostly*, not perfectly, connected.

## 3. Projection  (`src/pathsnap/geo.py`)

- **Method:** a per-trip local equirectangular projection centred on the trip's median GPS position. Scale factors come from the WGS84 radii of curvature at that latitude, and all distances are in metres.
- **Check:** distances between consecutive GPS fixes agree with haversine to **≤ 1.3 m on every trip** (`proj_check_max_abs_err_m`). Feature distances are all under 1 km.
- **Caveat:** over a whole 44 km trip the difference from haversine is about 0.3 %. Most of that is haversine's own spherical-earth error, not ours.
- **Why not pyproj:** it is not needed, and dropping it avoids a native dependency on the two demo laptops.

## 4. Per-observation labels  (`src/pathsnap/labels.py`, `results/data/trip_stats_*.csv`)

The `.route` gives the ordered arcs driven, not a label per GPS point: about 2,600 points vs about 280 arcs per trip. We derive one label per point.

**Method: monotone alignment by dynamic programming.**
- Point *i* is assigned route position *j_i*, with *j* non-decreasing over time.
- Cost of an assignment = point-to-arc distance + 15 m × (1 − cos(heading − arc bearing))/2. The heading term applies only when the vehicle moves faster than 2 m/s.
- Moving forward along the route costs 2 + 0.05 × (length of any route arcs skipped).
- The problem is solved exactly in O(n_obs × n_route_arcs) using a prefix-minimum trick, in about 3 s for the largest trip (19k points × 2.3k route arcs).
- Why monotone rather than "nearest route arc": routes revisit places (loops, U-turns, the two directions of one road). The naive rule disagrees with the DP on 0–45 % of points per trip (median 5 %; per trip in `trip_stats_*.csv`).

**Results (train):**
- The median GPS-to-label distance is 3.5–11 m for 78 of 80 trips.
- Trips **33 and 54** sit **122 m and 140 m** from their reference route along its whole length (median). That is a systematic offset of either the GPS or the route. Trip 2 has a p95 of 242 m (local outliers).
- 88.8 % of train observations and 99.9 % of student_test observations have a label within 50 m (`label_reliable`).

**Limitations to state in the report (Dataset doc §6):**
- The labels are *derived* from a reference route that is itself unverified.
- GPS points recorded before the route starts or after it ends are forced onto the first or last arc, which gives low confidence.
- On parallel or twin roads, the label inherits whatever the route says.
- `label_conf = exp(−½ (d/20 m)²)` is stored for every point.

**Proposed decision (M2 to confirm with CV evidence):**
- Train classifiers only on observations with `label_reliable == True`. This removes most of trips 33 and 54, about 11 % of train observations.
- Still *report* metrics on all observations, and separately on the reliable subset.

## 5. Candidate generation  (`src/pathsnap/candidates.py`, `results/data/candidate_sweep.csv/.png`)

- **Index:** arcs are densified every 10 m into a KD-tree. This is exact: no arc within *r* is missed, however long it is.
- **Sweep:** over six radius schedules × K ∈ {3 … 40} on the train trips. The miss rate is measured on observations with reliable labels.

| K | miss rate (adaptive 50→100→200 m) | mean candidates / point |
|---|---|---|
| 8 | 0.885 % | 6.4 |
| 12 | 0.572 % | 7.9 |
| 20 | 0.294 % | 9.6 |
| **30 (chosen)** | **0.026 %** | **10.4** |
| 40 | 0.010 % | 10.7 |

- Every miss at K = 12 happened at points with 12+ arcs within 50 m (dense junctions), so the cap, not the radius, was the bottleneck.
- K = 30 cuts misses 20× while adding only 2.5 candidates per point on average.
- The adaptive radius (50 → 100 → 200 m, only where fewer than 3 candidates are found) is kept so far-off points still get candidates for the HMM. A fixed 30 m radius has a 1.5 % floor.
- **Final miss rate** (`build_log.json`): train 0.03 % on reliable labels (4.96 % on all observations, almost entirely trips 33/54); student_test **0.10 %**.

## 6. Features  (`src/pathsnap/features.py`, 37 columns, order in `artifacts/features.json`)

Groups:
- geometry: distance, rank, gap to the best/second-best candidate, projection fraction, signed offset, arc length
- direction: cos and absolute difference between the vehicle heading and the arc bearing, at 3 window sizes
- window: mean/max distance of the ±2 and ±5 neighbouring fixes to this arc
- kinematics: speed, acceleration, time gaps, stationary flag
- context: number of candidates, twin flags, node degrees
- continuity: the arc is the same as, or linked to, the previous/next point's top-3 candidates

How features are built:
- They use only track, nodes and arcs, and the same function runs offline and in the API.
- The scaler is fit on train rows only (`artifacts/scaler.joblib`).

**Baseline for context:** picking the nearest arc gives Top-1 = 49.2 % on train and 53.2 % on student_test. This is the bar the classifiers must beat.

## 7. Anything dropped, clipped or imputed

- Nothing is dropped from the processed tables.
- Clipping applied in features: speed at 80 m/s, acceleration at ±20 m/s², time gaps at 3,600 s, neighbour distances at 500 m.
- The training-row filter (`label_reliable`) is a modelling decision, logged in section 4.
