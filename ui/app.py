"""Streamlit evaluator UI (Laptop 2).  OWNER: M4.   Contract: README 3.14 [FROZEN].

Run:  PATHSNAP_API_URL=http://<laptop1-ip>:8000 streamlit run ui/app.py

Features:
- Dynamically discovers the ten evaluation trips from GET /trips (no hardcoded IDs).
- Evaluator selects any trip, sends trip_id to POST /predict, displays live spinner.
- Interactive multi-layer Folium map (Observed GPS, HMM reconstructed route, toggleable classifier route, start/end pins).
- Live KPI metric cards (Top-1 HMM & Classifier, Route Continuity, Sequence-level F1, Total Latency).
- Latency waterfall chart & breakdown table per pipeline stage.
- Clear error alerts and API status indicators.
- Strict compliance with Red Line R2: NEVER requests or displays server-side reference data.
"""
import os
import requests
import streamlit as st
import folium
from folium import FeatureGroup, LayerControl
from streamlit_folium import st_folium
import pandas as pd

# API Endpoint Configuration
API_URL = os.environ.get("PATHSNAP_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(
    page_title="PathSnap — Map-Matching Evaluator",
    layout="wide",
    page_icon="🗺️",
    initial_sidebar_state="expanded",
)

# Custom Styling for polished dashboard UI
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 14px;
        border: 1px solid #dee2e6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .status-badge {
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
    }
    .status-online { background-color: #d4edda; color: #155724; }
    .status-offline { background-color: #f8d7da; color: #721c24; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar: Connection & Evaluation Trip Controls ---
st.sidebar.image("https://img.icons8.com/color/96/route.png", width=64)
st.sidebar.title("PathSnap Evaluator")
st.sidebar.caption("PES University · UE24CS352A · Laptop 2 UI")
st.sidebar.divider()

# 1. Connection check
st.sidebar.subheader("🔌 Laptop 1 API Connection")
target_url_input = st.sidebar.text_input("API URL", value=API_URL)
if target_url_input != API_URL:
    API_URL = target_url_input.rstrip("/")

api_connected = False
health_data = {}
try:
    health_resp = requests.get(f"{API_URL}/health", timeout=3)
    if health_resp.status_code == 200:
        api_connected = True
        health_data = health_resp.json()
        st.sidebar.markdown(
            '<span class="status-badge status-online">● Laptop 1 API Online</span>',
            unsafe_allow_html=True,
        )
        st.sidebar.caption(
            f"• Available Evaluation Trips: **{health_data.get('available_trips', 0)}**\n"
            f"• Model Loaded: **{health_data.get('model_loaded')}**"
        )
    else:
        st.sidebar.markdown(
            '<span class="status-badge status-offline">⚠️ API Error</span>',
            unsafe_allow_html=True,
        )
except Exception:
    st.sidebar.markdown(
        '<span class="status-badge status-offline">❌ Laptop 1 Offline</span>',
        unsafe_allow_html=True,
    )
    st.sidebar.caption("Check Laptop 1 IP & ensure `uvicorn api.main:app` is running.")

st.sidebar.divider()

# 2. Dynamic Trip Selection (Red Line R3: Zero hardcoding)
st.sidebar.subheader("📍 Evaluation Trips")
trips = []
if api_connected:
    try:
        resp = requests.get(f"{API_URL}/trips", timeout=5)
        if resp.status_code == 200:
            trips = resp.json()
    except Exception as e:
        st.sidebar.error(f"Error fetching trips: {e}")

trip_ids = [t["trip_id"] for t in trips] if trips else []

if trip_ids:
    selected_trip = st.sidebar.selectbox(
        "Choose an Evaluation Trip:",
        trip_ids,
        format_func=lambda tid: f"Trip {tid} ({next((t['n_points'] for t in trips if t['trip_id'] == tid), 0)} pts)",
    )
else:
    selected_trip = None
    st.sidebar.info("No evaluation trips found on API.")

# Map Layer Controls in sidebar
st.sidebar.subheader("🗺️ Map Layer Toggles")
show_gps = st.sidebar.checkbox("Show Observed GPS Points (Blue)", value=True)
show_hmm = st.sidebar.checkbox("Show Reconstructed HMM Route (Red)", value=True)
show_classifier = st.sidebar.checkbox("Show Classifier-Only Route (Orange)", value=False)
base_tile = st.sidebar.selectbox("Base Map Style", ["CartoDB positron", "OpenStreetMap", "CartoDB dark_matter"])

st.sidebar.divider()
run_btn = st.sidebar.button("🚀 Reconstruct Route & Evaluate", type="primary", use_container_width=True, disabled=not selected_trip)

# --- Main Dashboard Header ---
st.title("🗺️ PathSnap: Distributed Map-Matching & Route Reconstruction")
st.markdown(
    "Demonstrating live candidate classification and **HMM/Viterbi** sequential decoding across road networks. "
    "All evaluation scoring is computed strictly server-side without ground-truth label leakage."
)

if not api_connected:
    st.warning("⚠️ **Waiting for Laptop 1 Model & API server.** Please verify the host IP and port in the sidebar.")
    st.stop()

# --- Prediction & Scoring Execution ---
if run_btn and selected_trip:
    with st.spinner(f"Sending Trip `{selected_trip}` to Laptop 1 for full pipeline inference & scoring..."):
        try:
            pred_resp = requests.post(
                f"{API_URL}/predict",
                json={"trip_id": selected_trip},
                timeout=60,
            )
            if pred_resp.status_code == 200:
                st.session_state["result"] = pred_resp.json()
            else:
                st.error(f"Prediction failed with status {pred_resp.status_code}: {pred_resp.text}")
        except Exception as e:
            st.error(f"Network error communicating with API: {e}")

# --- Render Results ---
if "result" in st.session_state:
    res = st.session_state["result"]
    metrics = res.get("metrics", {})
    timings = res.get("timings_ms", {})
    trip_id = res.get("trip_id")
    n_obs = res.get("n_observations", 0)

    st.subheader(f"Evaluation Results for Trip: `{trip_id}`")

    # 5 KPI Metric Cards
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        st.metric(
            "Top-1 Accuracy (HMM)",
            f"{metrics.get('top1_accuracy_hmm', 0)*100:.2f}%",
            delta=f"{(metrics.get('top1_accuracy_hmm', 0) - metrics.get('top1_accuracy_classifier', 0))*100:+.1f}% vs Cls" if "top1_accuracy_classifier" in metrics else None,
        )
    with m_col2:
        st.metric("Classifier Top-1", f"{metrics.get('top1_accuracy_classifier', 0)*100:.2f}%")
    with m_col3:
        st.metric("Route Continuity", f"{metrics.get('route_continuity_hmm', 0)*100:.1f}%")
    with m_col4:
        st.metric("Sequence F1", f"{metrics.get('sequence_f1', 0):.4f}")
    with m_col5:
        st.metric("Total Latency", f"{timings.get('total', 0):.1f} ms")

    st.divider()

    # Visual Layout: Left Map, Right Inspection & Diagnostics
    map_col, info_col = st.columns([3, 1])

    with map_col:
        st.markdown("### 📍 Interactive Geographical Route Map")
        gps_pts = res.get("gps", [])
        pred_route = res.get("predicted_route", [])
        cls_route = res.get("classifier_route", [])

        if gps_pts:
            center_lat = sum(p[0] for p in gps_pts) / len(gps_pts)
            center_lon = sum(p[1] for p in gps_pts) / len(gps_pts)
            m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles=base_tile)

            # Observed GPS Layer
            if show_gps:
                fg_gps = FeatureGroup(name="Observed GPS Points", show=True)
                step = max(1, len(gps_pts) // 400)
                for idx, p in enumerate(gps_pts[::step]):
                    folium.CircleMarker(
                        location=[p[0], p[1]],
                        radius=3,
                        color="#0066cc",
                        fill=True,
                        fill_opacity=0.6,
                        popup=f"Obs #{idx*step} | t={p[2]:.1f}s",
                    ).add_to(fg_gps)
                fg_gps.add_to(m)

            # Classifier Route Layer
            if show_classifier and cls_route:
                fg_cls = FeatureGroup(name="Classifier-Only Route", show=True)
                folium.PolyLine(locations=cls_route, color="#fd7e14", weight=3, opacity=0.7).add_to(fg_cls)
                fg_cls.add_to(m)

            # HMM Reconstructed Route Layer
            if show_hmm and pred_route:
                fg_hmm = FeatureGroup(name="HMM Reconstructed Route", show=True)
                folium.PolyLine(locations=pred_route, color="#dc3545", weight=4, opacity=0.9).add_to(fg_hmm)
                fg_hmm.add_to(m)

            # Start and End markers
            folium.Marker(gps_pts[0][:2], popup="🏁 Trip Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
            folium.Marker(gps_pts[-1][:2], popup="🛑 Trip End", icon=folium.Icon(color="red", icon="stop")).add_to(m)

            LayerControl().add_to(m)
            st_folium(m, width=950, height=540)
        else:
            st.info("No GPS coordinates returned.")

    with info_col:
        st.markdown("### ⏱️ Stage Timings (ms)")
        timing_records = [{"Stage": k, "Duration (ms)": v} for k, v in timings.items() if k != "total"]
        if timing_records:
            timing_df = pd.DataFrame(timing_records)
            st.bar_chart(timing_df.set_index("Stage"))
            st.dataframe(timing_df, hide_index=True, use_container_width=True)

        st.markdown("### 📊 Trip Statistics")
        st.write(f"• Total Observations: **{n_obs:,}**")
        st.write(f"• Matched Arcs: **{len(res.get('predicted_arcs', [])):,}**")
        st.write(f"• Distinct Road Arcs: **{len(set(res.get('predicted_arcs', []))):,}**")
        st.write(f"• Candidate Miss Rate: **{metrics.get('candidate_miss_rate', 0.0):.2f}%**")
else:
    st.info("👈 Select any of the 10 evaluation trips in the sidebar and click **'Reconstruct Route & Evaluate'**.")
