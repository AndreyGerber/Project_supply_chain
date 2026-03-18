import streamlit as st
import pandas as pd
from pathlib import Path

# ... (Konfiguration bleibt oben)

@st.cache_data
def load_data():
    file_path = Path("src/data/clean/reviews_clean.csv")
    
    df = pd.read_csv(file_path)
    
    # Lösung 1: 'errors=coerce' wandelt ungültige Daten in 'NaT' (Not a Time) um,
    # statt das Programm mit einem Fehler abzubrechen.
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Zeilen löschen, bei denen das Datum nicht gelesen werden konnte
    df = df.dropna(subset=['date'])
    
    # Rating extrahieren
    df['rating_numeric'] = df['rating_svg'].str.extract('(\d+)').astype(float)
    return df

# Daten laden
df = load_data()

# Nur anzeigen, wenn Daten vorhanden sind
if not df.empty:
    st.title("📊 Analyse der Autodoc Kundenbewertungen")
    st.markdown("---")
    
    # Erste Metriken anzeigen
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Gesamtzahl Bewertungen", len(df))
    with col2:
        st.metric("Durchschnittliche Sterne", round(df['rating_numeric'].mean(), 2))

    st.subheader("Daten-Vorschau")
    st.dataframe(df.head(10))