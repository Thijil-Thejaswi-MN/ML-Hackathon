"""Model registry.  OWNER: M2.  Each owner adds their model module + one line here.

Implemented: logreg, knn, lightgbm, hgb (M2); extratrees (M1); rf, mlp (M4). (A GNB model was planned but not built.)
"""
from __future__ import annotations
import importlib
from typing import Any

REGISTRY: dict[str, str] = {
    # M4 models
    "rf": "pathsnap.models.rf:RandomForestCandidateModel",
    "mlp": "pathsnap.models.mlp:MLPCandidateModel",
    # M1 models
    "extratrees": "pathsnap.models.trees:ExtraTreesModel",
    # M2 models
    "logreg": "pathsnap.models.logreg:LogRegModel",
    "knn": "pathsnap.models.knn:KNNModel",
    "lightgbm": "pathsnap.models.boosting:LightGBMModel",
    "hgb": "pathsnap.models.boosting:HGBModel",          # sklearn histogram GBDT (no extra dependency)
    # M3 models
}


def get_model(name: str, **kw) -> Any:
    """Resolve model class from REGISTRY and instantiate with given kwargs."""
    if name not in REGISTRY:
        raise KeyError(f"Model '{name}' not found in registry. Available: {list(REGISTRY.keys())}")
    
    target = REGISTRY[name]
    module_path, class_name = target.split(":")
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(**kw)
