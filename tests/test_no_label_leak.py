"""Leakage / hardcode guard (README R2, R3).  OWNER: M4.  Run before every gate: pytest -q

The reference route and labels may be touched ONLY by labels.py, dataset.py, io.py (definition), scripts/*,
and api/evaluator.py. Everything on the prediction path must be free of them.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = re.compile(
    r"\.route\b|load_route|derive_obs_labels|obs_labels|true_arc|reference_route|ground_truth|\.route\"",
    re.IGNORECASE,
)
GUARDED = [
    "src/pathsnap/pipeline.py", "src/pathsnap/candidates.py", "src/pathsnap/features.py", "src/pathsnap/graph.py",
    "src/pathsnap/geo.py", "src/pathsnap/artifacts.py", "src/pathsnap/decoding", "ui", "api/main.py", "api/schemas.py",
]


def _files():
    for g in GUARDED:
        p = ROOT / g
        if p.is_dir():
            yield from p.rglob("*.py")
        elif p.exists():
            yield p


def scan(text: str) -> list[str]:
    return [m.group(0) for m in FORBIDDEN.finditer(text)]


def test_scanner_detects_leak():          # positive control: the scanner itself works
    assert scan("r = load_route(root, split, tid)")
    assert scan("open(f'{d}/{t}.route')")
    assert not scan("arcs = np.loadtxt(path_arcs)")


def test_prediction_path_does_not_touch_ground_truth():
    offenders = {}
    for f in _files():
        hits = scan(f.read_text(encoding="utf-8", errors="ignore"))
        if hits:
            offenders[str(f.relative_to(ROOT))] = hits
    assert not offenders, f"ground-truth references on the prediction path: {offenders}"


def test_no_hardcoded_trip_ids_in_api_or_ui():
    ids = re.compile(r"[\"']0000\d{4}[\"']")
    bad = [str(f.relative_to(ROOT)) for f in _files() if ids.search(f.read_text(encoding="utf-8", errors="ignore"))]
    assert not bad, f"hardcoded trip ids found in: {bad}"
