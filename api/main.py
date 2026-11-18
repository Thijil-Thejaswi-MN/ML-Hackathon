"""FastAPI backend (Laptop 1).  OWNER: M4.   Contract: README 3.13 [FROZEN].

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000 --log-level info
Env:  PATHSNAP_EVAL_DIR = folder with the ten evaluation trip folders.

Requirements: load the bundle ONCE at startup; list trips from disk (no hardcoded ids); run the FULL pipeline
per request; no prediction caching; print one readable log block per request (time, client, trip, stage timings, top-1, status).
"""
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from api.schemas import PredictRequest, PredictResponse

log = logging.getLogger("pathsnap.api")
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # TODO(M4): STATE["bundle"] = pathsnap.artifacts.load_bundle("artifacts")
    yield


app = FastAPI(title="PathSnap API", lifespan=lifespan)


@app.get("/health")
def health():
    # TODO(M4): model loaded?, manifest summary, number of trips
    return {"status": "ok", "model_loaded": "bundle" in STATE}


@app.get("/trips")
def trips():
    """Ids + number of GPS points for every trip folder in PATHSNAP_EVAL_DIR. No labels, no route."""
    raise HTTPException(status_code=501, detail="M4: implement")


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request):
    """load trip (track/nodes/arcs) -> pathsnap.pipeline.predict_trip -> api.evaluator.score_trip -> JSON; log the request."""
    t0 = time.time()
    raise HTTPException(status_code=501, detail="M4: implement")
