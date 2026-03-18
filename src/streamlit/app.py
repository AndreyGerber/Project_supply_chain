import streamlit as st
import pandas as pd
import plotly.express as px
import os
from pathlib import Path


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


df = load_data() 

# 3. Titel & Header
st.title("📊 Analyse der Autodoc Kundenbewertungen")
st.markdown("---")

# Nutze st.dataframe statt print, um es in der App zu sehen
st.subheader("Vorschau der Daten")
st.dataframe(df.head(10))
