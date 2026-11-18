# Section 7: API, UI, Deployment & System Integration
**Author:** M4 (Platform & Deployment Lead)

---

## 7.1 System Architecture & Distributed Deployment

To ensure a robust, live demonstration that satisfies the hackathon's two-laptop requirement without ground-truth leakage, PathSnap is split cleanly across two nodes:

```
┌──────────────────────────────────────────────┐        HTTP (REST)        ┌──────────────────────────────────────────────┐
│                  LAPTOP 2                    │ ───────────────────────►  │                  LAPTOP 1                    │
│             Streamlit Web UI                 │                           │            FastAPI Backend Engine            │
│               (Evaluator)                    │ ◄───────────────────────  │              (Model + Decoder)               │
│                                              │      JSON Response        │                                              │
│  • Dynamic trip dropdown from GET /trips     │   • Top-1 Acc (HMM & Cls) │  • In-memory ModelBundle loaded once         │
│  • POST /predict with {"trip_id": "..."}     │   • Route Continuity      │  • Full predict_trip() inference execution   │
│  • Interactive Folium Multi-layer Map        │   • Sequence-level F1     │  • Ground-truth scoring in evaluator.py only │
│  • Live metric cards & stage timing table    │   • Latency breakdown     │  • Structured console audit logging          │
└──────────────────────────────────────────────┘                           └──────────────────────────────────────────────┘
```

### Key Architectural Safeguards:
1. **Red Line R2 (Zero Ground-Truth Leakage):**
   - The UI and prediction pipeline **never** have access to the `.route` ground truth files.
   - The API only accesses `.route` inside [`api/evaluator.py`](file:///Users/uthkarsha/Documents/ML%20Hackaton%20/pathsnap/api/evaluator.py) *after* predictions are generated, returning aggregate metrics only.
   - Guarded by automated CI test [`tests/test_no_label_leak.py`](file:///Users/uthkarsha/Documents/ML%20Hackaton%20/pathsnap/tests/test_no_label_leak.py).
2. **Red Line R3 (No Hardcoding or Prediction Caching):**
   - Every trip requested triggers the full end-to-end pipeline: spatial index $\to$ candidates $\to$ features $\to$ classifier $\to$ HMM Viterbi decoding $\to$ route polyline stitching.
   - All evaluation trips are discovered dynamically from `PATHSNAP_EVAL_DIR` on disk.

---

## 7.2 API Specification & Live Logging

The backend runs on **Laptop 1** via Uvicorn:
```bash
export PATHSNAP_EVAL_DIR="/path/to/evaluation_data"
uvicorn api.main:app --host 0.0.0.0 --port 8000 --log-level info
```

### Console Audit Logging (Models §6 Requirement)
For every incoming request, the server prints a standardized diagnostic block containing timestamps, client IP, stage breakdown, and accuracy metrics:

```text
==================================================
PathSnap Request: Trip=00000040 | Client=192.168.1.105
Timings: {'load': 3.1ms, 'candidates': 1.2ms, 'features': 2.4ms, 'model': 4.1ms, 'hmm': 5.8ms, 'route': 1.1ms, 'scoring': 0.8ms, 'total': 18.5ms}
Top-1 Accuracy: 78.40% | Sequence F1: 0.8120
Status: 200 OK | Duration: 19.3ms
==================================================
```

---

## 7.3 Streamlit Evaluator Interface (Laptop 2)

The evaluator UI runs on **Laptop 2**:
```bash
export PATHSNAP_API_URL="http://<Laptop-1-IP>:8000"
streamlit run ui/app.py
```

### Features Demonstrated:
- **Server Connectivity Pill:** Displays connection status, model state, and trip count.
- **Dynamic Trip Dropdown:** Fetches available evaluation trips directly from Laptop 1.
- **Folium Interactive Map:**
  - Raw GPS trajectory (blue circular markers with observation metadata popups).
  - Predicted route polyline (red).
  - Optional toggleable classifier-only route overlay (orange).
  - Green Start and Red End marker pins.
- **Stage Timings Table:** Transparent latency monitoring across all pipeline stages.

---

## 7.4 Verification & Stress Testing

The deployment has been validated against all 10 student testing trips:
- **Average end-to-end latency:** < 1.0s per trip.
- **Failure rate:** 0% across all 10 trips.
- **Automated test suite:** 100% pass rate on leak detection tests.
