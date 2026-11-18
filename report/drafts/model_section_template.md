# Model <N>: <name>  (family: <family>)  - owner: <M?>

Fill from THIS model's result files (results/models/<name>/). Textbook descriptions alone are not acceptable.

## A. Model understanding
**A1. Why is this algorithm suitable for PathSnap?** (link to the candidate-classification formulation, features, imbalance, need for probabilities)

**A2. Major strengths and limitations for this task** (tie to what you saw: e.g. speed, calibration, twin-arc direction handling)

## B. Training and configuration
**B5. Important hyperparameters and their initial values** (table; seed; class-weight choice; scaling)

**B6. Loss function / optimisation objective**

## C. Experimental results
**C7. Performance on the student-testing dataset** (summary + per-trip variation)

**C8. Observation-level Top-1, Macro-F1 and Log Loss on student-test** (table; N/A + reason where not applicable)

**C9. Overfitting / underfitting?** Evidence: train vs CV vs student-test gap, fold std, loss curves ONLY if a real history exists.

Figures (actual results only): confusion matrix; loss curve if applicable.
Source files: <paths>   Partition(s) used: <CV on 80 train trips / student-test>
