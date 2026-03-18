import streamlit as st
import pandas as pd
import plotly.express as px
import os
from pathlib import Path

@st.cache_data
def load_data():
    # Wir suchen direkt ab dem Hauptverzeichnis des Repos
    # Streamlit Cloud setzt das Arbeitsverzeichnis meist auf das Root-Level
    file_path = Path("src/data/clean/reviews_clean.csv")
    
    # Falls das nicht klappt, probieren wir den Pfad ohne 'src/' am Anfang
    if not file_path.exists():
        file_path = Path("data/clean/reviews_clean.csv")

    if not file_path.exists():
        # Zeige alle Dateien im aktuellen Verzeichnis an, um den Fehler zu finden
        st.error(f"Datei nicht gefunden! Aktuelles Verzeichnis: {os.getcwd()}")
        st.write("Vorhandene Dateien:", os.listdir("."))
        return pd.DataFrame()

    df = pd.read_csv(file_path)
    # ... Rest deines Codes
    return df

# Danach nicht vergessen, die Funktion auch aufzurufen:
df = load_data()

# Nur anzeigen, wenn Daten geladen wurden
if not df.empty:
    st.title("📊 Analyse der Autodoc Kundenbewertungen")
    st.dataframe(df.head(10))