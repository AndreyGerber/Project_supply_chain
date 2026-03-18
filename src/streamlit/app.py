import streamlit as st
import pandas as pd
import plotly.express as px
import os
from pathlib import Path

@st.cache_data
def load_data():
    # 1. Wir ermitteln den absoluten Pfad zu diesem Skript (app.py)
    current_dir = Path(__file__).parent 
    
    # 2. Wir gehen einen Ordner höher (zu 'src/') und dann in 'data/clean/'
    # Struktur: src/streamlit/app.py -> src/data/clean/reviews_clean.csv
    file_path = current_dir.parent / "data" / "clean" / "reviews_clean.csv"
    
    # Check, ob die Datei wirklich da ist (hilft beim Debuggen)
    if not file_path.exists():
        st.error(f"Datei nicht gefunden! Gesucht unter: {file_path}")
        return pd.DataFrame() # Gibt leeres Set zurück, damit App nicht abstürzt

    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    # Rating aus SVG-Namen extrahieren
    df['rating_numeric'] = df['rating_svg'].str.extract('(\d+)').astype(float)
    return df

# Danach nicht vergessen, die Funktion auch aufzurufen:
df = load_data()

# Nur anzeigen, wenn Daten geladen wurden
if not df.empty:
    st.title("📊 Analyse der Autodoc Kundenbewertungen")
    st.dataframe(df.head(10))