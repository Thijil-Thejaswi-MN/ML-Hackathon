# Data notes, quality issues and decisions  (owner: M1)

Record every anomaly, assumption and decision **with the evidence file that supports it**. The report's EDA section
and the viva both draw from this file. Re-verify every "lead" below with our own code before citing it.

## Leads from the planning pass (NOT yet verified by our code)
| Lead | Where to verify | Decision / outcome (fill in) |
|------|-----------------|------------------------------|
| Trips are located all over the world; each has its own road network | `results/eda/inventory.csv` (lon/lat extents) | per-trip local projection |
| Reference route has ~10x fewer arcs than GPS points; no per-observation label is supplied | inventory | derive labels (`labels.py`) |
| Many arcs have a reverse twin (two-way roads) | `results/eda/network_stats.csv` | heading features |
| Some train trips look systematically shifted from their reference route (e.g. ids 33, 54); a few have huge outliers (e.g. 2, 19) | per-trip GPS-to-route distance table | keep / down-weight / exclude **from training only**; justify with CV evidence |
| Very large time gaps in some trips (e.g. 78, 90) and irregular sampling | time-gap table | HMM chain breaks, gap features |
| Large trips (up to ~19k points, ~300k arcs) | inventory | performance budget for the API |

## Label derivation
Method, parameters, per-trip agreement with the simple nearest-route-arc rule, known failure modes.

## Reference-route limitations (Dataset doc section 6)
What the route is and is not; how we avoid overstating label certainty.

## Candidate generation
Chosen radii, K, and the miss-rate curve that justified them.

## Anything dropped, clipped or imputed
(nothing is dropped silently)
