"""Distance-weighted KNN candidate model.  OWNER: M2.  Contract: README 3.7.

Algorithm family: Instance-based.
Why it suits PathSnap: "candidates that look like this were usually the true arc" - no assumption about the
shape of the decision boundary, so it is a useful non-parametric contrast to the linear and tree models.
Limitations: prediction cost grows with the stored training set (every query compares against every stored
row), so we store a seeded random SUBSAMPLE of training rows (max_train_rows); probabilities are
neighbour-vote fractions (coarse, weakly calibrated); sensitive to feature scaling (inputs are standardised
by artifacts/scaler.joblib) and to irrelevant features (all 37 count equally in the distance).
"""
from __future__ import annotations
from typing import Optional

import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from pathsnap.models.base import CandidateModel


class KNNModel(CandidateModel):
    name: str = "knn"
    family: str = "Instance-based (k-nearest neighbours)"
    history: Optional[dict] = None

    def __init__(self, n_neighbors: int = 25, weights: str = "distance", p: int = 2,
                 max_train_rows: int = 60000, batch_rows: int = 50000, random_state: int = 42,
                 n_jobs: int = -1, **kwargs):
        self.params = {"n_neighbors": n_neighbors, "weights": weights, "p": p, "max_train_rows": max_train_rows,
                       "batch_rows": batch_rows, "random_state": random_state, "n_jobs": n_jobs, **kwargs}
        self.clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p, n_jobs=n_jobs,
                                        algorithm="brute", **kwargs)
        self.n_train_rows_used = None

    def fit(self, X, y, groups=None, **kw) -> "KNNModel":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        m = self.params["max_train_rows"]
        if m and len(X) > m:
            # stratified by class so the ~8 % positives keep their share
            rng = np.random.default_rng(self.params["random_state"])
            idx = []
            for c in (0, 1):
                ic = np.where(y == c)[0]
                k = int(round(m * len(ic) / len(y)))
                idx.append(rng.choice(ic, size=min(k, len(ic)), replace=False))
            idx = np.sort(np.concatenate(idx))
            X, y = X[idx], y[idx]
        self.n_train_rows_used = int(len(X))
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        b = self.params["batch_rows"]
        out = np.empty(len(X), dtype=np.float32)
        col = list(self.clf.classes_).index(1) if 1 in self.clf.classes_ else None
        for s in range(0, len(X), b):
            pr = self.clf.predict_proba(X[s:s + b])
            out[s:s + b] = pr[:, col] if col is not None else 0.0
        return out

    def save(self, path: str) -> None:
        joblib.dump({"params": self.params, "clf": self.clf, "n": self.n_train_rows_used}, path, compress=3)

    @classmethod
    def load(cls, path: str) -> "KNNModel":
        d = joblib.load(path)
        inst = cls(**d["params"])
        inst.clf, inst.n_train_rows_used = d["clf"], d["n"]
        return inst
