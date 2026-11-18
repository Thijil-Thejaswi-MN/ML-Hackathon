"""Extra-Trees candidate model.  OWNER: M1.  Contract: README 3.7.

Algorithm family: Tree-based ensemble (extremely randomised trees).
Why it suits PathSnap: the decision "is this candidate the true arc?" depends on non-linear interactions
(distance only matters when heading agrees; heading only matters when the vehicle is moving; twin arcs differ
only in heading). Trees capture these without scaling. Compared with a Random Forest (M4), Extra-Trees also
randomises the split THRESHOLDS, which lowers variance and trains faster - useful with ~2.4M candidate rows.
Limitations: probabilities are vote fractions (not calibrated - check log loss); large forests are big on disk;
no notion of sequence (the HMM adds that).
"""
from __future__ import annotations
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from pathsnap.models.base import CandidateModel


class ExtraTreesModel(CandidateModel):
    name: str = "extratrees"
    family: str = "Tree-based ensemble (randomised)"
    history: Optional[dict] = None  # no epoch history -> no training curves (README R7)

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = 18,
        min_samples_leaf: int = 5,
        max_features: float | str = "sqrt",
        class_weight: Optional[str] = None,
        max_samples: Optional[float] = None,
        bootstrap: bool = False,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs,
    ):
        if max_samples is not None:
            bootstrap = True  # sklearn only allows max_samples with bootstrap
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "class_weight": class_weight,
            "max_samples": max_samples,
            "bootstrap": bootstrap,
            "random_state": random_state,
            "n_jobs": n_jobs,
            **kwargs,
        }
        self.clf = ExtraTreesClassifier(**self.params)

    def fit(self, X, y, groups=None, **kw) -> "ExtraTreesModel":
        self.clf.fit(np.asarray(X, dtype=np.float32), np.asarray(y))
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Shape (n,) = P(candidate is the true arc)."""
        p = self.clf.predict_proba(np.asarray(X, dtype=np.float32))
        if p.shape[1] == 2:
            return p[:, 1]
        return np.full(len(p), float(self.clf.classes_[0] == 1), dtype=np.float32)

    def feature_importances(self, feature_names) -> dict:
        """Impurity-based importances (train data). Use permutation importance on held-out data for the report."""
        return dict(sorted(zip(feature_names, self.clf.feature_importances_.tolist()), key=lambda kv: -kv[1]))

    def save(self, path: str) -> None:
        joblib.dump({"params": self.params, "clf": self.clf}, path, compress=3)

    @classmethod
    def load(cls, path: str) -> "ExtraTreesModel":
        data = joblib.load(path)
        inst = cls(**data["params"])
        inst.clf = data["clf"]
        return inst
