import streamlit as st
import pandas as pd
import plotly.express as px

# 1. Konfiguration (Muss ganz oben stehen)
st.set_page_config(page_title="Autodoc Review Analyse", layout="wide")

# 2. Daten laden
@st.cache_data
def load_data():
    # Pfad angepasst auf deine Struktur
    df = pd.read_csv("src/data/clean/reviews_clean.csv")
    df['date'] = pd.to_datetime(df['date'])
    # Rating aus SVG-Namen extrahieren
    df['rating_numeric'] = df['rating_svg'].str.extract('(\d+)').astype(float)
    return df

# Daten tatsächlich in Variable laden
try:
    df = load_data()
except Exception as e:
    st.error(f"Fehler beim Laden der Daten: {e}")
    st.stop()

# 3. Titel & Header
st.title("📊 Analyse der Autodoc Kundenbewertungen")
st.markdown("---")
