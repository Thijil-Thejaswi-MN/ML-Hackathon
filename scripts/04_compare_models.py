"""Build results/model_comparison.csv + all required comparison figures (README 3.7, 5.2)

OWNER: M2.  Run after training models with scripts/03_train_model.py:
    python scripts/04_compare_models.py [--tags logreg knn lightgbm extratrees rf mlp]
Reads results/models/<tag>/{config.yaml,cv.json,test.json,train_fit.json,timing.json}; every number in the table and
the figures comes from those files (README R7). Figures -> figures/models/.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_run(d: Path) -> dict:
    import yaml
    cfg = yaml.safe_load((d / "config.yaml").read_text())
    cv = json.loads((d / "cv.json").read_text())
    te = json.loads((d / "test.json").read_text())
    tr = json.loads((d / "train_fit.json").read_text()) if (d / "train_fit.json").exists() else None
    tm = json.loads((d / "timing.json").read_text())
    key = ", ".join(f"{k}={v}" for k, v in cfg["params"].items()
                    if k not in ("random_state", "n_jobs", "verbose", "device", "batch_rows"))
    if cfg.get("train_obs_frac", 1.0) < 1.0:
        key += f", train_obs_frac={cfg['train_obs_frac']}"
    return {
        "Model": d.name, "Algorithm family": cfg.get("family", cfg["model"]), "Key configuration": key,
        "CV Macro F1 (mean)": cv["macro_f1_mean"], "CV Macro F1 (std)": cv["macro_f1_std"],
        "CV Top-1 (mean)": cv["top1_mean"], "CV Top-1 (std)": cv["top1_std"],
        "Top-1 accuracy (student-test)": te["all"]["top1"],
        "Top-1 given hit (student-test)": te["all"].get("top1_given_hit"),
        "Route continuity (student-test, classifier-only)": te["classifier_only_route"]["continuity_strict"],
        "Route continuity relaxed": te["classifier_only_route"]["continuity_relaxed"],
        "Student-Test Macro F1": te["reliable"]["macro_f1"],
        "Log loss (student-test)": te["reliable"]["log_loss"],
        "Inference time (ms / 1000 obs)": tm["inference_ms_per_1000_obs"],
        "Precision (student-test)": te["reliable"]["precision"], "Recall (student-test)": te["reliable"]["recall"],
        "Threshold tau (from train OOF)": cv["threshold"],
        "Train time full fit (s)": tm["train_seconds_full"], "CV fit time per fold (s)": tm["cv_fit_seconds_mean"],
        "Model size (MB)": tm.get("model_size_mb"),
        "Train Top-1 (in-sample, 16 trips)": tr["all"]["top1"] if tr else np.nan,
        "Arc-set F1 vs reference (student-test)": te["classifier_only_route"]["arc_f1"],
    }


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(ROOT / "results/models"))
    ap.add_argument("--tags", nargs="*", help="subset / order of runs (default: all folders not starting with _)")
    args = ap.parse_args()
    base = Path(args.results)
    need = ("config.yaml", "cv.json", "test.json", "timing.json")
    if args.tags:
        dirs = [base / t for t in args.tags]
    else:
        dirs = []
        for d in sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith("_")):
            if all((d / f).exists() for f in need):
                dirs.append(d)
            else:
                print(f"[skip] {d.name}: not produced by scripts/03_train_model.py (missing {[f for f in need if not (d / f).exists()]})")
    df = pd.DataFrame([load_run(d) for d in dirs])
    df.to_csv(ROOT / "results/model_comparison.csv", index=False, encoding="utf-8")

    # the organisers' 9-column table (report section 3.1), formatted
    fmt = pd.DataFrame({
        "Model": df["Model"], "Algorithm family": df["Algorithm family"], "Key configuration": df["Key configuration"],
        "CV Macro F1-score": [f"{m:.4f} ± {s:.4f}" for m, s in zip(df["CV Macro F1 (mean)"], df["CV Macro F1 (std)"])],
        "Top-1 Accuracy": [f"{v:.4f}" for v in df["Top-1 accuracy (student-test)"]],
        "Route continuity": [f"{v:.3f}" for v in df["Route continuity (student-test, classifier-only)"]],
        "Student-Test Macro F1-score": [f"{v:.4f}" for v in df["Student-Test Macro F1"]],
        "Log Loss": [f"{v:.4f}" for v in df["Log loss (student-test)"]],
        "Inference Time (ms/1000 obs)": [f"{v:.1f}" for v in df["Inference time (ms / 1000 obs)"]],
    })
    fmt.to_csv(ROOT / "results/model_comparison_report_table.csv", index=False, encoding="utf-8")
    try:
        md = fmt.to_markdown(index=False)          # needs the optional 'tabulate' package
    except ImportError:
        md = fmt.to_string(index=False)
    (ROOT / "results/model_comparison_report_table.md").write_text(md, encoding="utf-8")

    out = ROOT / "figures/models"
    out.mkdir(parents=True, exist_ok=True)
    names = df["Model"].tolist()
    x = np.arange(len(names))

    def bar(col, err, title, ylabel, fname, fmt_="{:.3f}", lower_better=False):
        fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(names)), 4))
        vals = df[col].to_numpy(dtype=float)
        ax.bar(x, vals, yerr=df[err] if err else None, capsize=4, color="#1f6fd1")
        for i, v in enumerate(vals):
            ax.text(i, v, fmt_.format(v), ha="center", va="bottom", fontsize=8)
        ax.set(xticks=x, xticklabels=names, title=title, ylabel=ylabel)
        if not lower_better:
            lo = np.nanmin(vals - (df[err].to_numpy() if err else 0))
            ax.set_ylim(max(0, lo - 0.05), min(1.0, np.nanmax(vals) + 0.03) if np.nanmax(vals) <= 1 else None)
        ax.grid(axis="y", alpha=.3)
        fig.tight_layout(); fig.savefig(out / fname, dpi=130); plt.close(fig)

    bar("CV Macro F1 (mean)", "CV Macro F1 (std)", "CV Macro F1 (mean ± std over 5 group folds)  [train]",
        "macro F1 @ tau", "01_cv_macro_f1.png", "{:.4f}")
    bar("CV Top-1 (mean)", "CV Top-1 (std)", "CV observation-level Top-1 (mean ± std)  [train]",
        "Top-1 accuracy", "02_cv_top1.png", "{:.4f}")

    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(names)), 4))
    w = 0.38
    a = ax.bar(x - w / 2, df["Top-1 accuracy (student-test)"], w, label="Top-1 accuracy", color="#1f6fd1")
    b = ax.bar(x + w / 2, df["Student-Test Macro F1"], w, label="Macro F1", color="#f28e1c")
    ax.bar_label(a, fmt="%.3f", fontsize=7); ax.bar_label(b, fmt="%.3f", fontsize=7)
    lo = min(df["Top-1 accuracy (student-test)"].min(), df["Student-Test Macro F1"].min())
    ax.set(xticks=x, xticklabels=names, ylim=(max(0, lo - 0.05), 1.0),
           title="Student-test Top-1 accuracy and Macro F1  [student_test]")
    ax.legend(); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(out / "03_student_test_top1_macro_f1.png", dpi=130); plt.close(fig)

    bar("Log loss (student-test)", None, "Log loss (lower = better probabilities)  [student_test]", "log loss",
        "04_log_loss.png", "{:.4f}", lower_better=True)
    bar("Inference time (ms / 1000 obs)", None, "Inference time: features -> probabilities  [student_test]",
        "ms per 1000 GPS observations", "05_inference_time.png", "{:.1f}", lower_better=True)

    # over/underfitting: train (in-sample) vs CV vs student-test Top-1
    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(names)), 4))
    w = 0.27
    for i, (col, lab, c) in enumerate([("Train Top-1 (in-sample, 16 trips)", "train (in-sample)", "#9a9a9a"),
                                       ("CV Top-1 (mean)", "CV (out-of-fold)", "#1f6fd1"),
                                       ("Top-1 accuracy (student-test)", "student-test", "#f28e1c")]):
        bars = ax.bar(x + (i - 1) * w, df[col], w, label=lab, color=c)
        ax.bar_label(bars, fmt="%.3f", fontsize=6)
    ax.set(xticks=x, xticklabels=names, ylim=(max(0, df[["CV Top-1 (mean)", "Top-1 accuracy (student-test)"]].min().min() - 0.06), 1.0),
           title="Generalisation gap: train vs CV vs student-test Top-1", ylabel="Top-1 accuracy")
    ax.legend(); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(out / "06_overfitting_gap.png", dpi=130); plt.close(fig)

    bar("Route continuity (student-test, classifier-only)", None,
        "Route continuity of classifier-only predictions (before HMM)  [student_test]", "share of connected arc pairs",
        "07_route_continuity.png", "{:.3f}")

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(fmt.to_string(index=False))
    print(f"\nwrote results/model_comparison.csv, results/model_comparison_report_table.csv/.md, figures/models/*.png")


if __name__ == "__main__":
    main()
