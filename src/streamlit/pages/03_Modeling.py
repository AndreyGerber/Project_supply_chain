import streamlit as st
from PIL import Image
import os

# Die gewünschte Überschrift für das Bild
st.header("Fake-News")

# Pfad zu deinem Bild in der VM
img_path = "src/streamlit/static/"Analysis.png"  

if os.path.exists(img_path):
    image = Image.open(img_path)
    # Bild anzeigen
    st.image(image, use_container_width=True)
else:
    st.error(f"Bild nicht gefunden unter: {img_path}")
    st.info("Bitte stelle sicher, dass das Bild im Ordner 'src/static/' liegt.")
