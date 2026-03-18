import streamlit as st
import pandas as pd
import plotly.express as px
import os
from pathlib import Path

@st.cache_data
def load_data():
    # Sucht im selben Ordner, in dem app.py liegt
    current_dir = Path(__file__).parent
    file_path = current_dir / "reviews_clean.csv"

    if not file_path.exists():
        st.error(f"Datei '{file_path.name}' nicht in {current_dir} gefunden!")
        # Zeige zur Sicherheit an, welche Dateien GitHub in diesem Ordner sieht:
        st.write("Dateien in diesem Ordner:", os.listdir(current_dir))
        return pd.DataFrame()

    df = pd.read_csv(file_path)
    # ... Rest deines Codes (Date-Konvertierung etc.)
    return df

df = load_data()

if not df.empty:
    st.success("✅ Daten erfolgreich geladen!")
    st.dataframe(df.head())

# Nur anzeigen, wenn Daten geladen wurden
if not df.empty:
    st.title("📊 Analyse der Autodoc Kundenbewertungen")
    st.dataframe(df.head(10))