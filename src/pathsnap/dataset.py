"""Offline dataset assembly (train / student_test).  OWNER: M1.

Builds candidates + features + labels, writes data/processed/*.parquet, folds.json (frozen group folds), fits scaler on TRAIN only.
"""
from __future__ import annotations


def build_split(split: str, cfg: dict):
    raise NotImplementedError


def make_group_folds(trip_ids: list[str], n_splits: int, seed: int) -> dict:
    """Group-aware folds by trip (balance by trip size). Write once; everybody reuses folds.json."""
    raise NotImplementedError
