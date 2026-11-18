# MLP results - not part of the reported comparison

These files were produced by M4 with a separate training/evaluation script that is not in this repository; its
cross-validation was scored on all observations (including the unreliable-label trips 33 and 54), so the numbers are
not comparable with the other models and cannot be re-derived from the repository. They are therefore excluded from
`results/model_comparison.csv` and the report (section 3.1).

To produce comparable MLP results under the shared protocol (requires PyTorch):

    python scripts/03_train_model.py --model mlp --train-obs-frac 0.3
