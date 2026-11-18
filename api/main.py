"""FastAPI backend (Laptop 1).  OWNER: M4.   Contract: README 3.13 [FROZEN].

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000 --log-level info
Env:  PATHSNAP_EVAL_DIR = folder with the ten evaluation trip folders.

Requirements: load the bundle ONCE at startup; list trips from disk (no hardcoded ids); run the FULL pipeline
per request; no prediction caching; print one readable log block per request (time, client, trip, stage timings, top-1, status).
"""
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

# Add src to sys.path so pathsnap modules can be imported directly
src_dir = str(Path(__file__).resolve().parents[1] / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

from api.schemas import PredictRequest, PredictResponse, Metrics
from api.evaluator import eval_dir, score_trip
from pathsnap.artifacts import load_bundle
from pathsnap.io import load_trip
from pathsnap.pipeline import predict_trip

# Set up logging format
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pathsnap.api")

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load bundle ONCE at startup
    artifacts_dir = os.environ.get("PATHSNAP_ARTIFACTS_DIR", "artifacts")
    STATE["bundle"] = load_bundle(artifacts_dir)
    log.info(f"Loaded bundle from {artifacts_dir} (manifest: {STATE['bundle'].manifest})")
    yield


app = FastAPI(title="PathSnap API", lifespan=lifespan)

# Allow CORS for UI access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    edir = eval_dir()
    n_trips = 0
    if edir and Path(edir).exists():
        n_trips = len([p for p in Path(edir).iterdir() if p.is_dir()])
    bundle_loaded = "bundle" in STATE and STATE["bundle"] is not None
    manifest = STATE.get("bundle").manifest if bundle_loaded else {}
    return {
        "status": "ok",
        "model_loaded": bundle_loaded,
        "manifest": manifest,
        "available_trips": n_trips,
        "eval_dir": edir,
    }


@app.get("/trips")
def trips():
    """Ids + number of GPS points for every trip folder in PATHSNAP_EVAL_DIR. No labels, no route."""
    edir = eval_dir()
    if not edir or not Path(edir).exists():
        raise HTTPException(
            status_code=404,
            detail=f"Evaluation directory not found. Please set PATHSNAP_EVAL_DIR.",
        )

    trip_list = []
    for p in sorted(Path(edir).iterdir()):
        if p.is_dir() and not p.name.startswith("."):
            track_file = p / f"{p.name}.track"
            if track_file.exists():
                try:
                    # Quickly count lines in the track file to determine observation count
                    with open(track_file, "r") as f:
                        n_points = sum(1 for line in f if line.strip())
                    trip_list.append({"trip_id": p.name, "n_points": n_points})
                except Exception as e:
                    log.warning(f"Failed to read track for {p.name}: {e}")
                    trip_list.append({"trip_id": p.name, "n_points": -1})

    return trip_list


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request):
    """load trip (track/nodes/arcs) -> pathsnap.pipeline.predict_trip -> api.evaluator.score_trip -> JSON; log the request."""
    t0 = time.time()
    trip_id = req.trip_id
    edir = eval_dir()

    if not edir or not Path(edir).exists():
        raise HTTPException(status_code=500, detail="PATHSNAP_EVAL_DIR is not configured.")

    trip_folder = Path(edir) / trip_id
    if not trip_folder.exists() or not trip_folder.is_dir():
        raise HTTPException(status_code=404, detail=f"Trip {trip_id} not found in {edir}.")

    try:
        # Load trip (track/nodes/arcs ONLY - never the route!)
        trip = load_trip(Path(edir).parent, Path(edir).name, trip_id)
    except Exception as e:
        log.error(f"Error loading trip {trip_id}: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to load trip files: {str(e)}")

    # Run the full pipeline
    bundle = STATE.get("bundle")
    try:
        prediction = predict_trip(trip, bundle)
    except Exception as e:
        log.error(f"Prediction failed for {trip_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    # Server-side scoring (allowed in evaluator.py only)
    t_score_0 = time.perf_counter()
    metrics_dict = score_trip(trip_id, prediction)
    timings = dict(prediction.timings_ms)
    timings["scoring"] = round((time.perf_counter() - t_score_0) * 1000.0, 2)
    timings["total"] = round(timings.get("total", 0.0) + timings["scoring"], 2)

    # Convert GPS track for response [lat, lon, t]
    gps_coords = []
    # Subsample GPS if very large for network transfer
    step = max(1, trip.n_obs // 1000)
    for row in trip.track[::step]:
        gps_coords.append([float(row[1]), float(row[0]), float(row[2])])

    # Log readable block on console
    client_ip = request.client.host if request.client else "unknown"
    top1 = metrics_dict.get("top1_accuracy_hmm", 0.0)
    log.info(
        f"\n{'='*50}\n"
        f"PathSnap Request: Trip={trip_id} | Client={client_ip}\n"
        f"Timings: {timings}\n"
        f"Top-1 Accuracy: {top1*100:.2f}% | Sequence F1: {metrics_dict.get('sequence_f1', 0):.4f}\n"
        f"Status: 200 OK | Duration: {(time.time() - t0)*1000:.1f}ms\n"
        f"{'='*50}"
    )

    return PredictResponse(
        trip_id=trip_id,
        n_observations=trip.n_obs,
        gps=gps_coords,
        predicted_route=prediction.route_polyline,
        classifier_route=prediction.classifier_polyline,
        predicted_arcs=[int(a) for a in prediction.arc_per_obs],
        metrics=Metrics(**metrics_dict),
        timings_ms=timings,
    )
