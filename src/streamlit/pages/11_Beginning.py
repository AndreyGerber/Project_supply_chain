import streamlit as st
import os
from PIL import Image  # WICHTIG: Das hat oben gefehlt

# 1. Titel und Einleitung
st.title("🤖 Text Analysis") # Kleiner Tipp: Analysis schreibt man im Englischen mit 'y'


bilder = [
    "src/streamlit/static/Scrapping_data.png",
    "src/streamlit/static/Scrapping_data_2.png",
    "src/streamlit/static/Duplicates.png",
    "src/streamlit/static/Proportion.png"
]

for pfad in bilder:
    if os.path.exists(pfad):
        st.image(Image.open(pfad), use_container_width=True)
    else:
        st.warning(f"Datei nicht gefunden: {pfad}")
