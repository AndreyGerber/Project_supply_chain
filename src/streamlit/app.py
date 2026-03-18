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
    # Ermittelt das Verzeichnis, in dem app.py liegt
    base_path = Path(__file__).parent.parent # Geht von 'streamlit/' ein Verzeichnis hoch zu 'src/'
    file_path = base_path / "data" / "clean" / "reviews_clean.csv"
    
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df['rating_numeric'] = df['rating_svg'].str.extract('(\d+)').astype(float)
    return df

df = load_data()


st.title("📊 Analyse der Autodoc Kundenbewertungen")
st.markdown("---")

st.subheader("Vorschau der Daten")
# Jetzt kennt Streamlit 'df' und kann es anzeigen:
st.dataframe(df.head(10)) 
