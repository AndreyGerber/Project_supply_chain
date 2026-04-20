import streamlit as st
from PIL import Image
import os

st.title("Modeling Phase")

# Pfad zum Bild (Passe den Namen an, z.B. maria_note.jpg)
img_path = "src/static/maria_note.jpg"

if os.path.exists(img_path):
    image = Image.open(img_path)
    # Bild mit Bildunterschrift anzeigen
    st.image(image, caption="Feedback von Maria zur Modeling-Phase", use_container_width=True)
else:
    st.warning("Das Notiz-Bild wurde im Ordner 'src/static/' nicht gefunden.")

st.write("---")
st.write("Basierend auf diesem Feedback werden wir nun verschiedene Modelle testen.")
