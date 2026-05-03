import json
import time
from pathlib import Path

import cv2
import streamlit as st

from config import Config


st.set_page_config(page_title="ResQ Link Dashboard", layout="wide")
st.title("ResQ Link Mission Dashboard")
st.caption("Passive mission metrics only. Live video remains in the OpenCV window.")

metrics_placeholder = st.empty()
heatmap_placeholder = st.empty()


def load_state():
    state_path = Path(Config.DASHBOARD_STATE_PATH)
    if not state_path.exists():
        return None
    return json.loads(state_path.read_text(encoding="utf-8"))


while True:
    state = load_state()

    with metrics_placeholder.container():
        if state is None:
            st.info("Waiting for mission state from main.py ...")
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Victim Count", state["victim_count"])
            col2.metric("Fire Detected", "YES" if state["fire_detected"] else "NO")
            col3.metric("Explored Cells", state["explored_cells"])
            st.caption(f"Mode: {state['mode']} | State: {state['state']}")

    with heatmap_placeholder.container():
        heatmap_path = Path(Config.DASHBOARD_HEATMAP_PATH)
        if heatmap_path.exists():
            heatmap = cv2.imread(str(heatmap_path))
            if heatmap is not None:
                st.image(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), caption="Coverage Heatmap")
        else:
            st.info("Heatmap will appear once the mission starts.")

    time.sleep(1.0)
