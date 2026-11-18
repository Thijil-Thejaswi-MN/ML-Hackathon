"""Streamlit evaluator UI (Laptop 2).  OWNER: M4.   Contract: README 3.14 [FROZEN].

Run:  PATHSNAP_API_URL=http://<laptop1-ip>:8000 streamlit run ui/app.py

Must: list the ten trips from GET /trips, send only the trip id to POST /predict, show a spinner, then draw a Folium map
(observed GPS + HMM route, optional classifier-only layer) and the metrics returned by the API. Show API status and clear errors.
Must NOT request or display any ground-truth data.
"""
import os
import requests
import streamlit as st
import folium
from streamlit_folium import st_folium

API_URL = os.environ.get("PATHSNAP_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="PathSnap Evaluator", layout="wide", page_icon="🗺️")

st.title("🗺️ PathSnap — Map-Matching & Route Reconstruction")
st.caption("PES University · UE24CS352A Machine Learning · Live Evaluator Dashboard (Laptop 2)")

# Sidebar for connection and trip selection
st.sidebar.header("🔌 Server Connection")
st.sidebar.text(f"API Target:\n{API_URL}")

# Check API health
try:
    health_resp = requests.get(f"{API_URL}/health", timeout=3)
    if health_resp.status_code == 200:
        health_data = health_resp.json()
        st.sidebar.success("● Connected to Laptop 1 API")
        st.sidebar.caption(
            f"Model Loaded: {health_data.get('model_loaded')}\n"
            f"Available Trips: {health_data.get('available_trips')}"
        )
    else:
        st.sidebar.error("⚠️ API error response")
except Exception as e:
    st.sidebar.error(f"❌ Cannot connect to {API_URL}")
    st.sidebar.caption("Make sure Laptop 1 backend is running!")

st.sidebar.divider()
st.sidebar.header("📍 Select Trip")

trips = []
try:
    resp = requests.get(f"{API_URL}/trips", timeout=5)
    if resp.status_code == 200:
        trips = resp.json()
except Exception as e:
    st.sidebar.warning(f"Could not fetch trip list: {e}")

trip_options = [t["trip_id"] for t in trips] if trips else []
selected_trip = st.sidebar.selectbox("Evaluation Trips", trip_options if trip_options else ["None found"])

show_classifier_layer = st.sidebar.checkbox("Overlay Classifier-Only Route", value=False)
run_btn = st.sidebar.button("🚀 Reconstruct Route & Score", type="primary", use_container_width=True)

if not trip_options:
    st.warning("No trips found. Ensure the API server is running with `PATHSNAP_EVAL_DIR` pointing to the evaluation data.")
    st.stop()

if run_btn:
    with st.spinner(f"Running full prediction pipeline on Trip {selected_trip}..."):
        try:
            pred_resp = requests.post(
                f"{API_URL}/predict",
                json={"trip_id": selected_trip},
                timeout=60,
            )
            if pred_resp.status_code == 200:
                data = pred_resp.json()
                st.session_state["last_result"] = data
            else:
                st.error(f"Prediction failed with status {pred_resp.status_code}: {pred_resp.text}")
        except Exception as e:
            st.error(f"Error communicating with API: {e}")

if "last_result" in st.session_state:
    res = st.session_state["last_result"]
    metrics = res.get("metrics", {})
    timings = res.get("timings_ms", {})
    n_obs = res.get("n_observations", 0)

    st.subheader(f"Results for Trip: `{res.get('trip_id')}` ({n_obs} Observations)")

    # Metric Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Top-1 Accuracy (HMM)", f"{metrics.get('top1_accuracy_hmm', 0)*100:.1f}%")
    with col2:
        st.metric("Top-1 Accuracy (Classifier)", f"{metrics.get('top1_accuracy_classifier', 0)*100:.1f}%")
    with col3:
        st.metric("Route Continuity (HMM)", f"{metrics.get('route_continuity_hmm', 0)*100:.1f}%")
    with col4:
        st.metric("Sequence F1", f"{metrics.get('sequence_f1', 0):.3f}")
    with col5:
        st.metric("Total Latency", f"{timings.get('total', 0):.0f} ms")

    st.divider()

    # Visual Layout: Left Map, Right Stage Timings & Metrics
    map_col, info_col = st.columns([3, 1])

    with map_col:
        st.markdown("### Interactive Route Map")
        gps_pts = res.get("gps", [])
        pred_route = res.get("predicted_route", [])
        cls_route = res.get("classifier_route", [])

        if gps_pts:
            center_lat = sum(p[0] for p in gps_pts) / len(gps_pts)
            center_lon = sum(p[1] for p in gps_pts) / len(gps_pts)
            m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="CartoDB positron")

            # GPS Points
            fg_gps = folium.FeatureGroup(name="Observed GPS Points", show=True)
            for p in gps_pts:
                folium.CircleMarker(
                    location=[p[0], p[1]],
                    radius=3,
                    color="#0066cc",
                    fill=True,
                    fill_opacity=0.6,
                ).add_to(fg_gps)
            fg_gps.add_to(m)

            # Classifier Route if toggled
            if show_classifier_layer and cls_route:
                fg_cls = folium.FeatureGroup(name="Classifier Route", show=True)
                folium.PolyLine(locations=cls_route, color="#fd7e14", weight=3, opacity=0.8).add_to(fg_cls)
                fg_cls.add_to(m)

            # Predicted HMM Route
            if pred_route:
                fg_hmm = folium.FeatureGroup(name="Predicted Route (HMM)", show=True)
                folium.PolyLine(locations=pred_route, color="#dc3545", weight=4, opacity=0.9).add_to(fg_hmm)
                fg_hmm.add_to(m)

            # Start and End markers
            folium.Marker(gps_pts[0][:2], popup="Trip Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
            folium.Marker(gps_pts[-1][:2], popup="Trip End", icon=folium.Icon(color="red", icon="stop")).add_to(m)

            folium.LayerControl().add_to(m)
            st_folium(m, width=900, height=520)
        else:
            st.info("No GPS data to render.")

    with info_col:
        st.markdown("### Stage Timings (ms)")
        st.dataframe(
            [{"Stage": k, "Duration (ms)": v} for k, v in timings.items()],
            hide_index=True,
            use_container_width=True,
        )

        st.markdown("### Route Details")
        st.write(f"• Total GPS Points: **{n_obs}**")
        st.write(f"• Predicted Arcs: **{len(res.get('predicted_arcs', []))}**")
        st.write(f"• Distinct Arcs: **{len(set(res.get('predicted_arcs', [])))}**")
