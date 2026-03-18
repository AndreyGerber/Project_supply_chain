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
    df = pd.read_csv("src/data/clean/reviews_clean.csv")
    # ... restliche Logik in der Funktion
    return df

# --- DIESE ZEILE IST DER SCHLÜSSEL ---
df = load_data() 

st.title("📊 Analyse der Autodoc Kundenbewertungen")
st.markdown("---")

st.subheader("Vorschau der Daten")
# Jetzt kennt Streamlit 'df' und kann es anzeigen:
st.dataframe(df.head(10)) 
