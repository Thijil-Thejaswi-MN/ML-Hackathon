"""Shared model interface.  OWNER: M2.   Contract: README 3.7 [FROZEN]."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class CandidateModel(ABC):
    name: str = "unnamed"
    family: str = "unknown"
    history: Optional[dict] = None   # ONLY real epoch/iteration histories (train/val loss). Otherwise leave None.

    @abstractmethod
    def fit(self, X, y, groups=None, **kw) -> "CandidateModel": ...

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        """Return shape (n,) = P(label == 1)."""

    @abstractmethod
    def save(self, path: str) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "CandidateModel": ...
