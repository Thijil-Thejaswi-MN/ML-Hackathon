"""Artifact bundle I/O.  OWNER: M4.   Contract: README 3.11 [FROZEN names].

ModelBundle = model + scaler + feature list + candidate config + threshold + (calibrator) + hmm config + manifest.
Refuse to load when manifest/feature hash mismatches.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelBundle:
    model: Any
    scaler: Any
    feature_columns: list
    candidate_cfg: dict
    threshold: float
    hmm_cfg: dict
    calibrator: Any = None
    manifest: dict = field(default_factory=dict)


def load_bundle(artifacts_dir: str = "artifacts") -> ModelBundle:
    raise NotImplementedError


def save_bundle(bundle: ModelBundle, artifacts_dir: str = "artifacts") -> None:
    raise NotImplementedError
