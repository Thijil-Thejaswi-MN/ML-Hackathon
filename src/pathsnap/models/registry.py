"""Model registry.  OWNER: M2.  Each owner adds their model module + one line here.

Planned (README 3.7): logreg(M2) gnb(M3) knn(M2) extratrees(M1) rf(M4) hgb|lightgbm(M2) mlp(M4)
"""
from __future__ import annotations
import importlib
from typing import Any

REGISTRY: dict[str, str] = {
    # M4 models
    "rf": "pathsnap.models.rf:RandomForestCandidateModel",
    "mlp": "pathsnap.models.mlp:MLPCandidateModel",
    # M2 models
    "logreg": "pathsnap.models.logreg:LogRegModel",
    "knn": "pathsnap.models.knn:KNNModel",
    "lightgbm": "pathsnap.models.boosting:LightGBMModel",
    # M3 models
    "gnb": "pathsnap.models.gnb:GNBModel",
    # M1 models
    "extratrees": "pathsnap.models.trees:ExtraTreesModel",
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
