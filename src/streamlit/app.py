import streamlit as st
import pandas as pd
import plotly.express as px
import os
from pathlib import Path

@st.cache_data
def load_data():
    # Wir bauen den Pfad ausgehend vom Hauptverzeichnis zusammen
    base_path = Path(__file__).resolve().parent.parent.parent # Geht von app.py hoch zu 'project_supply_chain'
    file_path = base_path / "src" / "data" / "clean" / "reviews_clean.csv"

    # Falls der Pfad oben nicht stimmt, probieren wir es direkt (relativ zum Root)
    if not file_path.exists():
        file_path = Path("src/data/clean/reviews_clean.csv")

    if not file_path.exists():
        st.error(f"❌ Datei immer noch nicht gefunden!")
        st.info(f"Gesuchter Pfad: {file_path.absolute()}")
        # Checke, was IM 'src' Ordner ist
        if os.path.exists("src"):
            st.write("Inhalt von 'src':", os.listdir("src"))
        return pd.DataFrame()

    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df['rating_numeric'] = df['rating_svg'].str.extract('(\d+)').astype(float)
    return df

df = load_data()

if not df.empty:
    st.success("✅ Daten erfolgreich geladen!")
    st.dataframe(df.head())

# Nur anzeigen, wenn Daten geladen wurden
if not df.empty:
    st.title("📊 Analyse der Autodoc Kundenbewertungen")
    st.dataframe(df.head(10))