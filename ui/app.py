"""Streamlit evaluator UI (Laptop 2).  OWNER: M4.   Contract: README 3.14 [FROZEN].

Run:  PATHSNAP_API_URL=http://<laptop1-ip>:8000 streamlit run ui/app.py

Must: list the ten trips from GET /trips, send only the trip id to POST /predict, show a spinner, then draw a Folium map
(observed GPS + HMM route, optional classifier-only layer) and the metrics returned by the API. Show API status and clear errors.
Must NOT request or display any ground-truth data.
"""
import os

import streamlit as st

API_URL = os.environ.get("PATHSNAP_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="PathSnap", layout="wide")
st.title("PathSnap - map-matching evaluation")
st.info("M4: implement trip selector, run button, map and metrics.")
