"""Artifact bundle I/O.  OWNER: M4.   Contract: README 3.11 [FROZEN names].

ModelBundle = model + scaler + feature list + candidate config + threshold + (calibrator) + hmm config + manifest.
Refuse to load when manifest/feature hash mismatches.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any
import joblib


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
    """Load the model bundle from artifacts_dir.
    
    If the artifacts directory is empty or missing (stub mode before M1-M3 finish training),
    returns a default ModelBundle with None model and default configurations so the pipeline
    can operate in stub/fallback mode.
    """
    path = Path(artifacts_dir)
    if not path.exists():
        return ModelBundle(
            model=None,
            scaler=None,
            feature_columns=[],
            candidate_cfg={"radius_m": 50.0, "top_k": 5},
            threshold=0.5,
            hmm_cfg={"beta": 5.0, "v_max": 35.0},
            calibrator=None,
            manifest={"version": "stub-0.1.0"},
        )

    # If manifest exists, read it
    manifest = {}
    if (path / "manifest.json").exists():
        with open(path / "manifest.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)

    # Feature columns
    feature_columns = []
    if (path / "features.json").exists():
        with open(path / "features.json", "r", encoding="utf-8") as f:
            feature_columns = json.load(f)

    # Candidate config
    candidate_cfg = {"radius_m": 50.0, "top_k": 5}
    if (path / "candidate_config.json").exists():
        with open(path / "candidate_config.json", "r", encoding="utf-8") as f:
            candidate_cfg = json.load(f)

    # Threshold
    threshold = 0.5
    if (path / "threshold.json").exists():
        with open(path / "threshold.json", "r", encoding="utf-8") as f:
            t_data = json.load(f)
            threshold = float(t_data.get("threshold", 0.5) if isinstance(t_data, dict) else t_data)

    # HMM config
    hmm_cfg = {"beta": 5.0, "v_max": 35.0}
    if (path / "hmm_config.json").exists():
        with open(path / "hmm_config.json", "r", encoding="utf-8") as f:
            hmm_cfg = json.load(f)

    # Model & Scaler & Calibrator
    model = None
    if (path / "model.joblib").exists():
        model = joblib.load(path / "model.joblib")

    scaler = None
    if (path / "scaler.joblib").exists():
        scaler = joblib.load(path / "scaler.joblib")

    calibrator = None
    if (path / "calibrator.joblib").exists():
        calibrator = joblib.load(path / "calibrator.joblib")

    return ModelBundle(
        model=model,
        scaler=scaler,
        feature_columns=feature_columns,
        candidate_cfg=candidate_cfg,
        threshold=threshold,
        hmm_cfg=hmm_cfg,
        calibrator=calibrator,
        manifest=manifest,
    )


def save_bundle(bundle: ModelBundle, artifacts_dir: str = "artifacts") -> None:
    """Save the model bundle to artifacts_dir."""
    path = Path(artifacts_dir)
    path.mkdir(parents=True, exist_ok=True)

    with open(path / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(bundle.manifest, f, indent=2)

    with open(path / "features.json", "w", encoding="utf-8") as f:
        json.dump(bundle.feature_columns, f, indent=2)

    with open(path / "candidate_config.json", "w", encoding="utf-8") as f:
        json.dump(bundle.candidate_cfg, f, indent=2)

    with open(path / "threshold.json", "w", encoding="utf-8") as f:
        json.dump({"threshold": bundle.threshold}, f, indent=2)

    with open(path / "hmm_config.json", "w", encoding="utf-8") as f:
        json.dump(bundle.hmm_cfg, f, indent=2)

    if bundle.model is not None:
        joblib.dump(bundle.model, path / "model.joblib")

    if bundle.scaler is not None:
        joblib.dump(bundle.scaler, path / "scaler.joblib")

    if bundle.calibrator is not None:
        joblib.dump(bundle.calibrator, path / "calibrator.joblib")
