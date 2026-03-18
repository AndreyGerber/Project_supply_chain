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


# 3. Titel & Header
st.title("📊 Analyse der Autodoc Kundenbewertungen")
st.markdown("---")

# 4. Sidebar
st.sidebar.header("Filter-Optionen")
selected_rating = st.sidebar.multiselect("Bewertung wählen", options=[1,2,3,4,5], default=[1,2,3,4,5])

# 5. Metriken berechnen
durchschnitt = df['rating_numeric'].mean()
antwort_rate = df['supplier_response'].notna().mean() * 100

col1, col2, col3 = st.columns(3)
col1.metric("Gesamtanzahl Reviews", len(df))
col2.metric("Durchschnittliches Rating", f"{durchschnitt:.1f}")
col3.metric("Antwort-Rate", f"{antwort_rate:.1f}%")

st.markdown("---")
