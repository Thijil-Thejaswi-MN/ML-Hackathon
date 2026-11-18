"""Loader smoke test; skipped automatically when raw data is not present. OWNER: M1."""
from pathlib import Path
import pytest
from pathsnap.io import list_trip_ids, load_trip

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "Student folder"


@pytest.mark.skipif(not RAW.exists(), reason="raw data not present")
def test_load_all_student_test_trips():
    for tid in list_trip_ids(RAW, "student_test"):
        t = load_trip(RAW, "student_test", tid)
        assert t.n_obs > 0 and t.arcs.max() < len(t.nodes)
