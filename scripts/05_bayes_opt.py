"""Optuna TPE on the selected model, group-CV on train only; save every trial; tuned-vs-untuned table -> results/bo/

OWNER: M2.  Run AFTER the untuned model has been trained with 03_train_model.py and chosen with 04_compare_models.py:
    python scripts/05_bayes_opt.py --model lightgbm [--n-trials 40]
Search space, objective and budget come from configs/bo.yaml (written before running). Never uses student_test or
evaluator data for the search; student_test is used once, afterwards, to compare tuned vs untuned.

Outputs (results/bo/<model>/):
  study.db (Optuna SQLite storage, every trial)  trials.csv  best_params.json  search_config.json
  tuned_vs_untuned.csv  optimisation_history.png  param_vs_score.png
and the refitted tuned model in results/models/<model>_tuned/ (same protocol as every other model).
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


def suggest(trial, space: dict) -> dict:
    out = {}
    for name, s in space.items():
        if s["type"] == "float":
            out[name] = trial.suggest_float(name, s["low"], s["high"], log=s.get("log", False))
        elif s["type"] == "int":
            if s.get("log"):
                out[name] = trial.suggest_int(name, s["low"], s["high"], log=True)
            else:
                out[name] = trial.suggest_int(name, s["low"], s["high"], step=s.get("step", 1))
        elif s["type"] == "categorical":
            out[name] = trial.suggest_categorical(name, s["choices"])
    return out


def main() -> None:
    import optuna
    import yaml
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathsnap.training import Data, cross_validate, run_experiment

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="lightgbm")
    ap.add_argument("--n-trials", type=int)
    ap.add_argument("--config", default=str(ROOT / "configs/bo.yaml"))
    ap.add_argument("--skip-refit", action="store_true", help="only run the search")
    args = ap.parse_args()

    bo = yaml.safe_load(Path(args.config).read_text())
    space = bo["search_space"][args.model]
    n_trials = args.n_trials or bo["n_trials"]
    base_params = yaml.safe_load((ROOT / "configs/models" / f"{args.model}.yaml").read_text()) or {}
    out = ROOT / "results/bo" / args.model
    out.mkdir(parents=True, exist_ok=True)
    (out / "search_config.json").write_text(json.dumps({"model": args.model, "n_trials": n_trials, **bo,
                                                        "search_space": space, "base_params": base_params}, indent=1))

    data = Data()
    frac, seed = bo.get("train_obs_frac", 1.0), bo.get("seed", 42)

    def objective(trial):
        params = {**base_params, **suggest(trial, space)}
        t0 = time.perf_counter()
        _, folds, _, _ = cross_validate(data, args.model, params, train_obs_frac=frac, seed=seed,
                                        history_fold=None, trial=trial, verbose=False)
        scores = [f["top1"] for f in folds]
        trial.set_user_attr("fold_top1", scores)
        trial.set_user_attr("fold_log_loss", [f["log_loss"] for f in folds])
        trial.set_user_attr("seconds", round(time.perf_counter() - t0, 1))
        val = float(np.mean(scores))
        print(f"  trial {trial.number:3d}: cv_top1={val:.4f}  ({time.perf_counter() - t0:.0f}s)  {suggest_str(params, space)}",
              flush=True)
        return val

    def suggest_str(params, space):
        return " ".join(f"{k}={params[k]:.4g}" if isinstance(params[k], float) else f"{k}={params[k]}" for k in space)

    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=bo.get("n_startup_trials", 10))
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1) if bo.get("pruner") == "median" \
        else optuna.pruners.NopPruner()
    study = optuna.create_study(study_name=f"pathsnap_{args.model}", direction="maximize", sampler=sampler,
                                pruner=pruner, storage=f"sqlite:///{(out / 'study.db').as_posix()}",
                                load_if_exists=True)
    if bo.get("enqueue_untuned", True) and len(study.trials) == 0:
        start = {}
        for k, sp in space.items():      # untuned values, clipped into the search range (e.g. reg_alpha=0 on a log scale)
            if k in base_params and base_params[k] is not None:
                start[k] = type(sp["low"])(min(max(base_params[k], sp["low"]), sp["high"]))
        study.enqueue_trial(start, skip_if_exists=True)
    remaining = max(0, n_trials - len([t for t in study.trials if t.state.is_finished()]))
    t0 = time.time()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=remaining)
    search_s = time.time() - t0

    df = study.trials_dataframe()
    df.to_csv(out / "trials.csv", index=False)
    best = {**base_params, **study.best_params}
    (out / "best_params.json").write_text(json.dumps({"best_value_cv_top1": study.best_value,
                                                      "best_trial": study.best_trial.number,
                                                      "best_params": best,
                                                      "n_trials_total": len(study.trials),
                                                      "n_complete": int((df.state == "COMPLETE").sum()),
                                                      "n_pruned": int((df.state == "PRUNED").sum()),
                                                      "search_wall_clock_s_this_run": round(search_s, 1)}, indent=1))

    # plots: optimisation history + score vs each hyperparameter
    comp = df[df.state == "COMPLETE"].sort_values("number")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(comp.number, comp.value, s=18, label="trial (complete)")
    ax.plot(comp.number, comp.value.cummax(), color="r", label="best so far")
    pr = df[df.state == "PRUNED"]
    if len(pr):
        ax.scatter(pr.number, [comp.value.min()] * len(pr), marker="x", color="#999", label="pruned")
    ax.set(xlabel="trial", ylabel="CV Top-1 (mean of 5 group folds)",
           title=f"Bayesian optimisation (TPE) - {args.model}  [train, group CV]")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "optimisation_history.png", dpi=130); plt.close(fig)
    cols = [c for c in df.columns if c.startswith("params_")]
    nc = 4
    fig, axs = plt.subplots(int(np.ceil(len(cols) / nc)), nc, figsize=(14, 3.2 * int(np.ceil(len(cols) / nc))))
    for ax, c in zip(np.ravel(axs), cols):
        ax.scatter(comp[c], comp.value, s=12)
        ax.set(xlabel=c.replace("params_", ""), ylabel="CV Top-1")
        if space[c.replace("params_", "")].get("log"):
            ax.set_xscale("log")
    for ax in np.ravel(axs)[len(cols):]:
        ax.axis("off")
    fig.suptitle(f"CV Top-1 vs hyperparameter ({args.model})  [train]")
    fig.tight_layout(); fig.savefig(out / "param_vs_score.png", dpi=130); plt.close(fig)
    print(f"[bo] best cv_top1={study.best_value:.4f} at trial {study.best_trial.number}: {study.best_params}")

    if args.skip_refit:
        return
    # refit: tuned and untuned through the IDENTICAL protocol (100 % of training observations, same folds)
    untuned_dir = ROOT / "results/models" / args.model
    if not (untuned_dir / "test.json").exists():
        run_experiment(args.model, base_params, tag=args.model, data=data)
    run_experiment(args.model, best, tag=f"{args.model}_tuned", data=data,
                   notes=f"best params from results/bo/{args.model}/best_params.json")

    rows = []
    for tag, label in ((args.model, "untuned"), (f"{args.model}_tuned", "tuned")):
        d = ROOT / "results/models" / tag
        cv, te, tm = (json.loads((d / f).read_text()) for f in ("cv.json", "test.json", "timing.json"))
        rows.append({"model": label, "tag": tag,
                     "CV Macro F1 (mean)": cv["macro_f1_mean"], "CV Macro F1 (std)": cv["macro_f1_std"],
                     "CV Top-1 (mean)": cv["top1_mean"],
                     "Student-test Top-1": te["all"]["top1"], "Student-test Macro F1": te["reliable"]["macro_f1"],
                     "Log loss (student-test)": te["reliable"]["log_loss"],
                     "Inference ms / 1000 obs": tm["inference_ms_per_1000_obs"],
                     "Train time full fit (s)": tm["train_seconds_full"]})
    tv = pd.DataFrame(rows)
    tv.to_csv(out / "tuned_vs_untuned.csv", index=False)
    print(tv.to_string(index=False))


if __name__ == "__main__":
    main()
