#preprocessing overview

import streamlit as st

st.set_page_config(
    page_title="PNG Viewer",
    layout="centered"
)

st.title("The Preprocessing overview")

# Bild laden
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[3]

img_path = BASE_DIR / "src/streamlit/static/Preprocessing_overview.png"
st.image(
    img_path,
    caption="Overview",
    use_container_width=True
)