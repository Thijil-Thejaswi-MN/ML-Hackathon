# AI assistance disclosure

Tool: **Claude (Anthropic)**, used as a coding and writing assistant on 18 November 2026. Every experiment uses only
the provided train / student_test partitions; every reported number is read from result files produced by the code in
this repository. Data processing, training and Bayesian optimisation were run by team members on their laptops; the
runs listed as "run by the assistant" below were executed by Claude on the team's data and model outputs. The team chose the models, parameters and final configuration from the recorded evidence
and is responsible for all submitted content.

| Date | Tool | Used for | Files / sections affected | Team's own work and verification |
|------|------|----------|---------------------------|----------------------------------|
| 18-11-2026 | Claude | Reading the guidelines, planning the 4-person work split and timeline | README (planning parts) | Roles and plan agreed by the team |
| 18-11-2026 | Claude | Drafting data loading, metric projection, per-observation label alignment, candidate generation, 37 features, dataset builder, inventory / EDA / candidate-sweep scripts, Folium map builder | `src/pathsnap/{geo,graph,labels,candidates,features,dataset}.py`, `src/pathsnap/viz/folium_maps.py`, `scripts/00_inventory.py`, `01_eda.py`, `02_build_dataset.py`, `02a_candidate_sweep.py`, `docs/DATA_NOTES.md` | M1 ran all scripts on own laptop, checked outputs against the data, chose K = 30 from the sweep |
| 18-11-2026 | Claude | Drafting the Extra-Trees model wrapper | `src/pathsnap/models/trees.py`, `configs/models/extratrees.yaml` | M1 |
| 18-11-2026 | Claude | Drafting metrics, logistic regression / KNN / LightGBM wrappers, shared training protocol, comparison, Bayesian-optimisation and export scripts | `src/pathsnap/metrics.py`, `training.py`, `models/{logreg,knn,boosting}.py`, `configs/models/*.yaml`, `configs/bo.yaml`, `scripts/03,04,05,09` | M2 ran all training, the 40-trial optimisation and the export on own laptop; selected the final model |
| 18-11-2026 | Claude | Efficient implementation of M3's HMM design (emission, transition, chain breaks) on the team's graph structures; route stitching; decoder tuning / evaluation / map scripts | `src/pathsnap/decoding/{hmm,route}.py`, `scripts/06,07,08`, `scripts/hmm_eval_common.py` | HMM model designed by M3 (`notebooks/pathsnap_m3_hmm_route_matching.ipynb`). Decoder tuning and evaluation (`06`, `07` -> `results/hmm/`) **run by the assistant** using M2's out-of-fold / student-test probabilities of the final model; Folium maps (`08`) run by the team |
| 18-11-2026 | Claude | Correcting the API's server-side scoring to the report's metric definitions | `api/evaluator.py` | API, UI, pipeline and deployment built by M4 |
| 18-11-2026 | Claude | Submission clean-up: reproducible RF config, UTF-8 comparison table, EDA figure cropping fix (EDA figures first produced by M1's own run of `01_eda.py`; re-saved by the assistant with the cropping fix, data unchanged), README run order | `configs/models/rf.yaml`, `results/model_comparison_report_table.md`, `scripts/01_eda.py`, `figures/eda/`, `README.md` | Reviewed by M1 |
| 18-11-2026 | Claude | Drafting the technical report text from the team's result files | `report/PathSnap_Technical_Report.md/.pdf`, `docs/DECISIONS.md` | All numbers cross-checked against `results/`; each member to review their own section before the viva |

Statement: all metrics, figures and analysis in the report come from runs of the code in this repository on the
permitted data (by team members, or by the assistant where stated above); no results were invented or edited by hand.
