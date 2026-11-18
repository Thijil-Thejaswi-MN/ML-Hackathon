"""Random Forest Candidate Model.  OWNER: M4.  Contract: README 3.7.

Algorithm family: Ensemble (bagging).
Suitable for PathSnap: Handles non-linear feature interactions (spatial distance, bearing delta, speed),
robust to outliers, requires no feature scaling, produces smooth probability estimates via tree voting.
"""
from __future__ import annotations
from typing import Optional
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from pathsnap.models.base import CandidateModel


class RandomForestCandidateModel(CandidateModel):
    name: str = "rf"
    family: str = "Ensemble"
    history: Optional[dict] = None  # Random forest has no epoch history

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 14,
        min_samples_split: int = 5,
        min_samples_leaf: int = 4,
        max_samples: Optional[float] = 0.25,
        bootstrap: bool = True,
        class_weight: Optional[str] = None,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs,
    ):
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_samples": max_samples,
            "bootstrap": bootstrap,
            "class_weight": class_weight,
            "random_state": random_state,
            "n_jobs": n_jobs,
            **kwargs,
        }
        self.clf = RandomForestClassifier(**self.params)

    def fit(self, X, y, groups=None, **kw) -> "RandomForestCandidateModel":
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Return 1D array of probabilities for class 1 (P(is_true_arc == 1))."""
        probs = self.clf.predict_proba(X)
        if probs.shape[1] == 2:
            return probs[:, 1]
        elif probs.shape[1] == 1:
            classes = list(self.clf.classes_)
            if 1 in classes:
                return np.ones(len(X), dtype=np.float32)
            return np.zeros(len(X), dtype=np.float32)
        return probs.ravel()

    def save(self, path: str) -> None:
        joblib.dump({"params": self.params, "clf": self.clf}, path)

    @classmethod
    def load(cls, path: str) -> "RandomForestCandidateModel":
        data = joblib.load(path)
        instance = cls(**data["params"])
        instance.clf = data["clf"]
        return instance
