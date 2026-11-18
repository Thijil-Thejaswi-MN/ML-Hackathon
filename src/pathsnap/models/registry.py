"""Model registry.  OWNER: M2.  Each owner adds their model module + one line here.

Planned (README 3.7): logreg(M2) gnb(M3) knn(M2) extratrees(M1) rf(M4) hgb|lightgbm(M2) mlp(M4)
"""
from __future__ import annotations

REGISTRY: dict = {
    # "logreg": "pathsnap.models.logreg:LogRegModel",
}


def get_model(name: str, **kw):
    raise NotImplementedError("resolve REGISTRY[name] and instantiate with configs/models/<name>.yaml")
