"""Logistic Regression candidate model.  OWNER: M2.  Contract: README 3.7.

Algorithm family: Linear / probabilistic.
Why it suits PathSnap: a fast, well-understood baseline whose output is a genuine probability
(sigmoid of a linear score) - exactly what the HMM emission needs. Coefficients are directly interpretable
(e.g. how much heading agreement outweighs distance).
Limitations: decision boundary is linear in the features, so interactions such as "distance matters only
when the heading agrees" must be handed to it as features; it cannot learn them itself.
L2-regularised (sklearn default; strength 1/C). Inputs are already standardised by artifacts/scaler.joblib (fit on train only).
"""
from __future__ import annotations
from typing import Optional

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from pathsnap.models.base import CandidateModel


class LogRegModel(CandidateModel):
    name: str = "logreg"
    family: str = "Linear (logistic regression)"
    history: Optional[dict] = None  # lbfgs iterations are not an epoch history -> no curves

    def __init__(self, C: float = 1.0, class_weight: Optional[str] = None,
                 max_iter: int = 300, solver: str = "lbfgs", random_state: int = 42, **kwargs):
        self.params = {"C": C, "class_weight": class_weight, "max_iter": max_iter,
                       "solver": solver, "random_state": random_state, **kwargs}
        self.clf = LogisticRegression(**self.params)

    def fit(self, X, y, groups=None, **kw) -> "LogRegModel":
        self.clf.fit(np.asarray(X, dtype=np.float64), np.asarray(y))
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.clf.predict_proba(np.asarray(X, dtype=np.float64))[:, 1]

    def coefficients(self, feature_names) -> dict:
        return dict(sorted(zip(feature_names, self.clf.coef_[0].tolist()), key=lambda kv: -abs(kv[1])))

    def save(self, path: str) -> None:
        joblib.dump({"params": self.params, "clf": self.clf}, path)

    @classmethod
    def load(cls, path: str) -> "LogRegModel":
        d = joblib.load(path)
        inst = cls(**d["params"])
        inst.clf = d["clf"]
        return inst
