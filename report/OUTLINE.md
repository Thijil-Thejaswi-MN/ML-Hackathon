# Report outline (one consolidated PDF)  -- owner: M1 assembles

Front matter: team, repo URL, commit hash of reported results, AI-assistance disclosure.

2. Dataset understanding & EDA (M1) - preprocessing; patterns; features and why; missing-value analysis; trajectory plots; road-network plots; Folium map; partition stated per analysis.
3. Model experimentation & comparison (M2 + model owners) - 3.1 table; 3.2 per-model subsections; 3.3 visualisations; 3.4 selection rationale (primary metric + why).
4. Bayesian optimisation (M2) - six questions; untuned vs tuned table (CV macro-F1, student-test Top-1, student-test macro-F1, log loss, inference time); cost.
5. HMM + Viterbi (M3) - six questions; classifier-only vs +HMM; improvements and failures.
6. Folium visualisation (M3) - layers, legend, colours, zoom, layer control; screenshot(s); code pointers.
7. API, UI, deployment (M4) - UI screenshots; API-console screenshots on the receiving laptop; deployment steps.
Appendix - reproduction commands; member contributions; limitations.

Rules: every table/figure generated from results/ by a script; caption = partition + script; own words; peer-check numbers before freeze.
