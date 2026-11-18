"""Gradient-boosting candidate models (LightGBM, sklearn HistGradientBoosting).  OWNER: M2.  Contract: README 3.7.

Algorithm family: Boosting.
Why it suits PathSnap: trees added one at a time, each correcting the previous ensemble's errors, capture the
non-linear feature interactions (distance x heading x speed) with strong accuracy on tabular data, train fast
on ~2M rows, and output log-loss-optimised probabilities that the HMM can use as emissions.
Limitations: more hyperparameters to tune (learning rate vs number of trees vs leaves), can overfit with too
many leaves / iterations; no notion of sequence (the HMM adds that).

Both models record a REAL per-iteration training history (train / validation log loss) when the training
script passes eval_set=(X_val, y_val); the validation set is only monitored, never used to stop or select.
"""
from __future__ import annotations
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

from pathsnap.models.base import CandidateModel


class LightGBMModel(CandidateModel):
    name: str = "lightgbm"
    family: str = "Boosting (LightGBM, leaf-wise GBDT)"
    accepts_eval_set = True

    def __init__(self, n_estimators: int = 400, learning_rate: float = 0.05, num_leaves: int = 63,
                 max_depth: int = -1, min_child_samples: int = 50, subsample: float = 0.8, subsample_freq: int = 1,
                 colsample_bytree: float = 0.8, reg_alpha: float = 0.0, reg_lambda: float = 1.0,
                 is_unbalance: bool = False, random_state: int = 42, n_jobs: int = -1, **kwargs):
        try:
            import lightgbm  # noqa: F401
        except ImportError as e:
            raise ImportError("lightgbm is not installed: pip install lightgbm (or use the 'hgb' model)") from e
        self.params = {"n_estimators": n_estimators, "learning_rate": learning_rate, "num_leaves": num_leaves,
                       "max_depth": max_depth, "min_child_samples": min_child_samples, "subsample": subsample,
                       "subsample_freq": subsample_freq, "colsample_bytree": colsample_bytree,
                       "reg_alpha": reg_alpha, "reg_lambda": reg_lambda, "is_unbalance": is_unbalance,
                       "random_state": random_state, "n_jobs": n_jobs, **kwargs}
        from lightgbm import LGBMClassifier
        self.clf = LGBMClassifier(objective="binary", verbose=-1, **self.params)
        self.history: Optional[dict] = None

    def fit(self, X, y, groups=None, eval_set=None, **kw) -> "LightGBMModel":
        import lightgbm as lgb
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        if eval_set is not None:
            rec: dict = {}
            Xv, yv = eval_set
            self.clf.fit(X, y, eval_set=[(X, y), (np.asarray(Xv, dtype=np.float32), np.asarray(yv))],
                         eval_names=["train", "val"], eval_metric="binary_logloss",
                         callbacks=[lgb.record_evaluation(rec)])
            n = len(rec["train"]["binary_logloss"])
            self.history = {"iteration": list(range(1, n + 1)),
                            "train_logloss": list(rec["train"]["binary_logloss"]),
                            "val_logloss": list(rec["val"]["binary_logloss"])}
        else:
            self.clf.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.clf.predict_proba(np.asarray(X, dtype=np.float32))[:, 1]

    def feature_importances(self, feature_names, kind: str = "gain") -> dict:
        imp = self.clf.booster_.feature_importance(importance_type=kind)
        return dict(sorted(zip(feature_names, imp.tolist()), key=lambda kv: -kv[1]))

    def save(self, path: str) -> None:
        joblib.dump({"params": self.params, "clf": self.clf, "history": self.history}, path)

    @classmethod
    def load(cls, path: str) -> "LightGBMModel":
        d = joblib.load(path)
        inst = cls(**d["params"])
        inst.clf, inst.history = d["clf"], d.get("history")
        return inst


class HGBModel(CandidateModel):
    """sklearn HistGradientBoosting - same idea as LightGBM (histogram GBDT), no extra dependency."""
    name: str = "hgb"
    family: str = "Boosting (sklearn HistGradientBoosting)"
    accepts_eval_set = True

    def __init__(self, max_iter: int = 300, learning_rate: float = 0.08, max_leaf_nodes: int = 63,
                 max_depth: Optional[int] = None, min_samples_leaf: int = 50, l2_regularization: float = 0.0,
                 max_features: float = 1.0, class_weight: Optional[str] = None, random_state: int = 42,
                 history_every: int = 10, history_max_rows: int = 200000, **kwargs):
        self.params = {"max_iter": max_iter, "learning_rate": learning_rate, "max_leaf_nodes": max_leaf_nodes,
                       "max_depth": max_depth, "min_samples_leaf": min_samples_leaf,
                       "l2_regularization": l2_regularization, "max_features": max_features,
                       "class_weight": class_weight, "random_state": random_state, **kwargs}
        self.hist_cfg = {"history_every": history_every, "history_max_rows": history_max_rows}
        self.clf = HistGradientBoostingClassifier(early_stopping=False, **self.params)
        self.history: Optional[dict] = None

    def fit(self, X, y, groups=None, eval_set=None, **kw) -> "HGBModel":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.clf.fit(X, y)
        if eval_set is not None:
            # real per-iteration losses, computed from the fitted stages (monitoring only)
            rng = np.random.default_rng(0)
            m = self.hist_cfg["history_max_rows"]
            it = rng.choice(len(X), size=min(m, len(X)), replace=False)
            Xv, yv = np.asarray(eval_set[0], dtype=np.float32), np.asarray(eval_set[1])
            iv = rng.choice(len(Xv), size=min(m, len(Xv)), replace=False)
            every = self.hist_cfg["history_every"]
            h = {"iteration": [], "train_logloss": [], "val_logloss": []}
            for k, (pt, pv) in enumerate(zip(self.clf.staged_predict_proba(X[it]),
                                             self.clf.staged_predict_proba(Xv[iv])), start=1):
                if k % every == 0 or k == self.clf.n_iter_:
                    h["iteration"].append(k)
                    h["train_logloss"].append(float(log_loss(y[it], pt[:, 1], labels=[0, 1])))
                    h["val_logloss"].append(float(log_loss(yv[iv], pv[:, 1], labels=[0, 1])))
            self.history = h
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.clf.predict_proba(np.asarray(X, dtype=np.float32))[:, 1]

    def save(self, path: str) -> None:
        joblib.dump({"params": self.params, "hist_cfg": self.hist_cfg, "clf": self.clf, "history": self.history}, path)

    @classmethod
    def load(cls, path: str) -> "HGBModel":
        d = joblib.load(path)
        inst = cls(**d["params"], **d.get("hist_cfg", {}))
        inst.clf, inst.history = d["clf"], d.get("history")
        return inst
