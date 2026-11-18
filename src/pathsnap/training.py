"""Shared experiment runner for every candidate model.  OWNER: M2.

Used by scripts/03_train_model.py (one model) and scripts/05_bayes_opt.py (objective + tuned refit) so that every
model - tuned or not - is trained and scored with exactly the same data, folds, filters and metrics.

Protocol (README 3.6-3.9):
  * features: the 37 columns of artifacts/features.json, standardised with artifacts/scaler.joblib (fit on the
    80 train trips only). The API applies the same scaler, so training and serving see identical inputs
    (tree models are unaffected by scaling).
  * training rows: candidates of observations whose derived label is reliable (<= 50 m), see docs/DATA_NOTES.md.
  * CV: the frozen 5 group folds in data/processed/folds.json (whole trips per fold).
  * metrics: computed on fold/test observations, reported both on 'reliable' observations (primary) and 'all'.
  * threshold tau: maximises macro-F1 on train OOF probabilities; never chosen on student_test.
"""
from __future__ import annotations
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .dataset import load_table, save_table
from .metrics import (argmax_per_obs, candidate_level_metrics, obs_top1_accuracy, pick_threshold,
                      route_continuity, sequence_metrics)
from .models.registry import get_model

ROOT = Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------------------------------------- data
class Data:
    """All tables needed for training / evaluation, loaded once."""

    def __init__(self, root: Path = ROOT, include_test: bool = True):
        proc, art = root / "data/processed", root / "artifacts"
        self.root = root
        self.features = json.loads((art / "features.json").read_text())
        self.scaler = joblib.load(art / "scaler.joblib")
        folds = json.loads((proc / "folds.json").read_text())
        self.fold_of_trip = folds["fold_of_trip"]
        self.n_splits = folds["n_splits"]
        self.tr = load_table(proc / "candidates_train.parquet")
        self.obs_tr = load_table(proc / "obs_labels_train.parquet")
        self.tr["fold"] = self.tr["trip_id"].map(self.fold_of_trip).astype(np.int8)
        self.obs_tr["fold"] = self.obs_tr["trip_id"].map(self.fold_of_trip).astype(np.int8)
        rel = self.obs_tr.loc[self.obs_tr.label_reliable, ["trip_id", "obs_idx"]].assign(_rel=True)
        self.tr = self.tr.merge(rel, on=["trip_id", "obs_idx"], how="left")
        self.tr["_rel"] = self.tr["_rel"].fillna(False).astype(bool)
        self.X_tr = self._X(self.tr)
        self.y_tr = self.tr["label"].to_numpy().astype(np.int8)
        if include_test:
            self.te = load_table(proc / "candidates_student_test.parquet")
            self.obs_te = load_table(proc / "obs_labels_student_test.parquet")
            self.X_te = self._X(self.te)
            self.y_te = self.te["label"].to_numpy().astype(np.int8)

    def _X(self, df):
        return self.scaler.transform(df[self.features].to_numpy()).astype(np.float32)


def _obs_subsample_mask(df: pd.DataFrame, frac: float, seed: int) -> np.ndarray:
    """Keep whole observations (all their candidate rows) with probability frac."""
    if frac >= 1.0:
        return np.ones(len(df), dtype=bool)
    key = df["trip_id"].astype(np.int64).to_numpy() * 1_000_000 + df["obs_idx"].to_numpy()
    u = np.unique(key)
    keep = np.random.default_rng(seed).random(len(u)) < frac
    return np.isin(key, u[keep])


# ------------------------------------------------------------------------------------------ evaluate
def evaluate(cands: pd.DataFrame, p: np.ndarray, obs: pd.DataFrame, threshold: float) -> dict:
    """Candidate-level + observation-level metrics on 'reliable' and 'all' observations."""
    c = cands[["trip_id", "obs_idx", "arc_id", "label"]].copy()
    c["p"] = p
    out = {}
    for name, o in (("reliable", obs[obs.label_reliable]), ("all", obs)):
        keys = o[["trip_id", "obs_idx"]]
        cc = c.merge(keys, on=["trip_id", "obs_idx"])
        m = candidate_level_metrics(cc["label"], cc["p"], threshold)
        m.update(obs_top1_accuracy(cc, o))
        out[name] = m
    return out


def classifier_route_metrics(cands: pd.DataFrame, p: np.ndarray, split: str) -> dict:
    """Classifier-only (argmax per observation, no HMM) route continuity + sequence metrics per trip."""
    from .io import load_trip, load_route
    raw = ROOT / "data/raw/Student folder"
    c = cands[["trip_id", "obs_idx", "arc_id"]].copy()
    c["p"] = p
    best = argmax_per_obs(c).sort_values(["trip_id", "obs_idx"])
    per_trip = []
    for tid, g in best.groupby("trip_id"):
        trip = load_trip(raw, split, tid)
        route = load_route(raw, split, tid)
        rc = route_continuity(g["arc_id"].to_numpy(), trip.arcs)
        sq = sequence_metrics(g["arc_id"].to_numpy(), route)
        per_trip.append({"trip_id": tid, "continuity_strict": rc["strict"], "continuity_relaxed": rc["relaxed"],
                         **{k: v for k, v in sq.items()}})
    df = pd.DataFrame(per_trip)
    return {"per_trip": df.to_dict(orient="records"),
            "mean": df.drop(columns="trip_id").mean().to_dict()}


# ------------------------------------------------------------------------------------------------ run
def cross_validate(data: Data, model_name: str, params: dict, train_obs_frac: float = 1.0, seed: int = 42,
                   history_fold: int | None = 0, folds=None, trial=None, verbose=True):
    """Group-CV on the frozen folds. Returns (oof_p over ALL train rows, per-fold metrics, history, fold_seconds)."""
    oof = np.full(len(data.tr), np.nan, dtype=np.float32)
    fold_metrics, fold_secs, history = [], [], None
    folds = range(data.n_splits) if folds is None else folds
    for k in folds:
        tr_mask = (data.tr["fold"].to_numpy() != k) & data.tr["_rel"].to_numpy()
        tr_mask &= _obs_subsample_mask(data.tr, train_obs_frac, seed + k)
        va_mask = data.tr["fold"].to_numpy() == k
        model = get_model(model_name, **params)
        t0 = time.perf_counter()
        kw = {}
        if history_fold == k and getattr(model, "accepts_eval_set", False):
            rel_va = va_mask & data.tr["_rel"].to_numpy()
            kw["eval_set"] = (data.X_tr[rel_va], data.y_tr[rel_va])
        model.fit(data.X_tr[tr_mask], data.y_tr[tr_mask], **kw)
        fit_s = time.perf_counter() - t0
        oof[va_mask] = model.predict_proba(data.X_tr[va_mask])
        if kw and getattr(model, "history", None):
            history = model.history
        fold_secs.append(fit_s)
        # per-fold metrics at tau = 0.5 (tau is chosen afterwards on the pooled OOF)
        ev = evaluate(data.tr.loc[va_mask], oof[va_mask], data.obs_tr[data.obs_tr.fold == k], 0.5)
        fm = {"fold": int(k), "fit_seconds": round(fit_s, 2), "n_train_rows": int(tr_mask.sum()),
              "top1": ev["reliable"]["top1"], "top1_all": ev["all"]["top1"],
              "macro_f1@0.5": ev["reliable"]["macro_f1"], "log_loss": ev["reliable"]["log_loss"]}
        fold_metrics.append(fm)
        if verbose:
            print(f"    fold {k}: top1={fm['top1']:.4f} macroF1@.5={fm['macro_f1@0.5']:.4f} "
                  f"logloss={fm['log_loss']:.4f} fit={fit_s:.1f}s", flush=True)
        if trial is not None:   # Optuna pruning on the running mean
            trial.report(float(np.mean([f["top1"] for f in fold_metrics])), step=len(fold_metrics) - 1)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()
    return oof, fold_metrics, history, fold_secs


def _versions():
    import sklearn
    v = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
         "scikit-learn": sklearn.__version__}
    for mod in ("lightgbm", "torch", "optuna"):
        try:
            v[mod] = __import__(mod).__version__
        except ImportError:
            pass
    return v


def run_experiment(model_name: str, params: dict, tag: str | None = None, data: Data | None = None,
                   train_obs_frac: float = 1.0, seed: int = 42, save_model: bool = True, notes: str = "") -> dict:
    """Full protocol for one model configuration; writes results/models/<tag>/ and returns the summary."""
    import yaml
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tag = tag or model_name
    data = data or Data()
    out = ROOT / "results/models" / tag
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()
    print(f"[{tag}] model={model_name} params={params} train_obs_frac={train_obs_frac}", flush=True)

    # 1. group-CV
    oof, folds, history, fold_secs = cross_validate(data, model_name, params, train_obs_frac, seed)
    rel = data.tr["_rel"].to_numpy()
    tau = pick_threshold(data.y_tr[rel], oof[rel])
    cv_eval = evaluate(data.tr, oof, data.obs_tr, tau)
    fold_macro = []
    for k in range(data.n_splits):          # per-fold macro-F1 at the chosen tau (for mean +- std)
        m = (data.tr["fold"].to_numpy() == k) & rel
        fold_macro.append(candidate_level_metrics(data.y_tr[m], oof[m], tau)["macro_f1"])
        folds[k]["macro_f1@tau"] = fold_macro[-1]
    top1s = [f["top1"] for f in folds]
    cv = {"partition": "train (80 trips), frozen 5-fold group CV", "threshold": tau, "folds": folds,
          "top1_mean": float(np.mean(top1s)), "top1_std": float(np.std(top1s)),
          "macro_f1_mean": float(np.mean(fold_macro)), "macro_f1_std": float(np.std(fold_macro)),
          "pooled": cv_eval}

    # 2. full fit on all reliable train rows (+ optional obs subsample, same as CV)
    model = get_model(model_name, **params)
    m_full = rel & _obs_subsample_mask(data.tr, train_obs_frac, seed + 99)
    t0 = time.perf_counter()
    model.fit(data.X_tr[m_full], data.y_tr[m_full])
    train_s = time.perf_counter() - t0

    # 3. student_test (evaluated once) + inference timing
    t0 = time.perf_counter()
    p_te = model.predict_proba(data.X_te)
    infer_s = time.perf_counter() - t0
    test_eval = evaluate(data.te, p_te, data.obs_te, tau)
    route = classifier_route_metrics(data.te, p_te, "student_test")

    # 4. training-set fit on 16 seeded train trips (overfitting check: train vs CV vs test gap)
    rng = np.random.default_rng(seed)
    trips = sorted(data.fold_of_trip)
    sample_trips = set(rng.choice(trips, size=16, replace=False))
    ms = data.tr["trip_id"].isin(sample_trips).to_numpy()
    train_eval = evaluate(data.tr.loc[ms], model.predict_proba(data.X_tr[ms]),
                          data.obs_tr[data.obs_tr.trip_id.isin(sample_trips)], tau)

    # 5. save everything
    size_mb = None
    if save_model:
        joblib.dump(model, out / "model.joblib", compress=3)
        size_mb = round((out / "model.joblib").stat().st_size / 1e6, 2)
    n_obs_te = len(data.obs_te)
    timing = {"cv_fit_seconds_per_fold": [round(s, 2) for s in fold_secs],
              "cv_fit_seconds_mean": round(float(np.mean(fold_secs)), 2),
              "train_seconds_full": round(train_s, 2),
              "inference_seconds_student_test": round(infer_s, 3),
              "inference_ms_per_1000_obs": round(infer_s / n_obs_te * 1000 * 1000, 2),
              "student_test_candidate_rows": int(len(data.te)), "student_test_obs": int(n_obs_te),
              "model_size_mb": size_mb, "wall_clock_total_s": round(time.perf_counter() - t_all, 1)}
    test = {"partition": "student_test (10 trips), evaluated once", "threshold": tau, **test_eval,
            "classifier_only_route": route["mean"], "per_trip_route": route["per_trip"]}
    cfg = {"model": model_name, "tag": tag, "family": getattr(model, "family", ""), "params": params,
           "seed": seed, "train_obs_frac": train_obs_frac, "training_rows": "label_reliable observations only",
           "features": len(data.features), "scaling": "artifacts/scaler.joblib (fit on train)",
           "n_train_rows_full": int(m_full.sum()), "versions": _versions(), "notes": notes}
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out / "cv.json").write_text(json.dumps(cv, indent=1))
    (out / "test.json").write_text(json.dumps(test, indent=1))
    (out / "train_fit.json").write_text(json.dumps({"partition": "16 seeded train trips (in-sample)", **train_eval},
                                                   indent=1))
    (out / "timing.json").write_text(json.dumps(timing, indent=1))
    if history:
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        h = pd.DataFrame(history)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(h.iteration, h.train_logloss, label="train (fold-0 training folds)")
        ax.plot(h.iteration, h.val_logloss, label="validation (fold 0)")
        ax.set(xlabel="boosting iteration", ylabel="log loss", title=f"{tag}: training history  [train, CV fold 0]")
        ax.legend(); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(out / "history.png", dpi=130); plt.close(fig)
    cm = np.array(test_eval["reliable"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["pred 0", "pred 1"], yticklabels=["true 0", "true 1"],
           title=f"{tag} @ tau={tau:.2f}\n[student_test, candidate rows]")
    fig.tight_layout(); fig.savefig(out / "confusion_matrix.png", dpi=130); plt.close(fig)
    (ROOT / "results/oof").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/preds").mkdir(parents=True, exist_ok=True)
    save_table(data.tr[["trip_id", "obs_idx", "arc_id", "fold"]].assign(p=oof, label=data.y_tr),
               ROOT / "results/oof" / f"{tag}.parquet")
    save_table(data.te[["trip_id", "obs_idx", "arc_id"]].assign(p=p_te),
               ROOT / "results/preds" / f"student_test_{tag}.parquet")

    summary = {"tag": tag, "model": model_name, "cv_top1": cv["top1_mean"], "cv_top1_std": cv["top1_std"],
               "cv_macro_f1": cv["macro_f1_mean"], "test_top1": test_eval["all"]["top1"],
               "test_top1_reliable": test_eval["reliable"]["top1"], "test_macro_f1": test_eval["reliable"]["macro_f1"],
               "test_log_loss": test_eval["reliable"]["log_loss"], "continuity": route["mean"]["continuity_strict"],
               **{k: timing[k] for k in ("train_seconds_full", "inference_ms_per_1000_obs", "model_size_mb")}}
    print(f"[{tag}] CV top1={cv['top1_mean']:.4f}+-{cv['top1_std']:.4f}  CV macroF1={cv['macro_f1_mean']:.4f}  "
          f"TEST top1={summary['test_top1']:.4f}  macroF1={summary['test_macro_f1']:.4f}  "
          f"logloss={summary['test_log_loss']:.4f}  continuity={summary['continuity']:.3f}  "
          f"({timing['wall_clock_total_s']}s)", flush=True)
    return summary
