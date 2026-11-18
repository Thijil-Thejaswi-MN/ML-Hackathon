# Decision log (owner: M4; everyone appends)

Format: date/time - decision - alternatives considered - evidence (file + numbers) - who decided.

## Organiser answers (README 1.4)
1. Phase-1 evaluation covers:
2. Evaluator data delivery / layout:
3. Macro-F1 level:
4. Missing per-model questions 3-4:
5. AI-disclosure format:
6. Laptop / network setup at demo:

## G0 - contracts frozen
## G3 - model selection (primary metric, rubric scores, chosen model, why)
2026-11-18 - **Final candidate model: LightGBM, tuned by Bayesian optimisation (`lightgbm_tuned`).** Decided by M2 (team).
- Primary metric: CV observation-level Top-1 (mean of the 5 frozen group folds, train only); tie-breakers CV macro F1, then log loss
  (the HMM consumes probabilities, so calibration matters), then inference time. Student-test was used only to report, never to choose.
- Evidence (`results/model_comparison.csv`, all five models through the identical protocol of `scripts/03_train_model.py`):

| Model | CV Top-1 | CV Macro F1 | Student-test Top-1 | Log loss | ms / 1000 obs |
|---|---|---|---|---|---|
| LightGBM | **0.9058 ± 0.028** | **0.9445** | **0.9014** | **0.0434** | 40.7 |
| ExtraTrees | 0.8967 ± 0.029 | 0.9404 | 0.8949 | 0.0501 | 56.9 |
| Random Forest | 0.8967 ± 0.029 | 0.9406 | 0.8868 | 0.0779 | 54.3 |
| Logistic regression | 0.8857 ± 0.031 | 0.9252 | 0.8801 | 0.0629 | 1.9 |
| KNN | 0.8608 ± 0.034 | 0.9132 | 0.8553 | 0.0900 | 919.7 |

- Why LightGBM: best on every selection criterion (CV Top-1, CV macro F1, log loss), small (1.3 MB untuned / 3.6 MB tuned)
  and fast enough for the live API. Alternatives: ExtraTrees / RF are 0.9 points lower on CV Top-1 with worse log loss
  (RF 0.078) and 14-98 MB on disk; logistic regression is the fastest but ~2 points lower; KNN is lowest and ~20x slower.
- Generalisation: LightGBM train (in-sample) 0.945 vs CV 0.906 vs student-test 0.901 - a moderate gap, CV agrees with test.
- Threshold tau = 0.45 (tuned model), chosen on train OOF probabilities (max macro F1); Top-1 itself is threshold-free.

## Bayesian optimisation - search space justification
2026-11-18 - Optuna TPE on LightGBM, ranges and reasons written in `configs/bo.yaml` before running (decided by M2).
- Objective: mean CV Top-1 over the 5 frozen group folds (reliable train observations); each trial fitted on a seeded 30 %
  of training observations (validation folds complete) to fit 40 trials into the event; median pruner; trial 0 = untuned values.
- Budget/result (`results/bo/lightgbm/`): 40 trials (15 complete, 25 pruned), 3,817 s. Best trial 20: n_estimators 900,
  learning_rate 0.039, num_leaves 80, min_child_samples 12, subsample 0.78, colsample_bytree 0.67, reg_alpha 1.60, reg_lambda 0.038.
- Tuned vs untuned, both refit on 100 % of training data (`results/bo/lightgbm/tuned_vs_untuned.csv`): CV Top-1 0.9058 -> 0.9074,
  CV macro F1 0.9445 -> 0.9453, student-test Top-1 0.9014 -> 0.9073, log loss 0.0434 -> 0.0423; cost: inference 40.7 -> 117.5
  ms / 1000 obs (900 trees), still far below the demo budget. Kept the tuned model: small but consistent gains on CV and test.
## Decoder - parameters chosen and why
