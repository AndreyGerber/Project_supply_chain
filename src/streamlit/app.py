import streamlit as st
import pandas as pd
import plotly.express as px
import os

# 1. Konfiguration (muss ganz oben stehen)
st.set_page_config(page_title="Autodoc Review Analyse", layout="wide")

# 2. Daten laden
@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(__file__)
    file_path = os.path.join(BASE_DIR, "src/data/clean/reviews_clean.csv")

    df = pd.read_csv(file_path)

    # Spalten prüfen & umwandeln
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    if 'rating_svg' in df.columns:
        df['rating_numeric'] = df['rating_svg'].str.extract(r'(\d+)').astype(float)

    return df

# Daten laden mit Fehlerbehandlung
try:
    df = load_data()
except Exception as e:
    st.error(f"❌ Fehler beim Laden der Daten: {e}")
    st.stop()

# 3. Titel
st.title("📊 Analyse der Autodoc Kundenbewertungen")
st.markdown("---")

# 4. Sidebar Filter
st.sidebar.header("🔧 Filter")

if 'rating_numeric' in df.columns:
    min_rating = st.sidebar.slider("Mindestbewertung", 1, 5, 1)
    df_filtered = df[df['rating_numeric'] >= min_rating]
else:
    df_filtered = df.copy()

# 5. Datenüberblick
st.subheader("🔍 Datenüberblick")
st.write(df_filtered.head())

col1, col2 = st.columns(2)
with col1:
    st.metric("Anzahl Bewertungen", len(df_filtered))
with col2:
    if 'rating_numeric' in df_filtered.columns:
        st.metric("Ø Bewertung", round(df_filtered['rating_numeric'].mean(), 2))

# 6. Fehlende Werte
st.subheader("🧹 Fehlende Werte")
st.write(df_filtered.isna().sum())

# 7. Verteilung der Bewertungen
if 'rating_numeric' in df_filtered.columns:
    st.subheader("⭐ Verteilung der Bewertungen")

    fig = px.histogram(
        df_filtered,
        x="rating_numeric",
        nbins=5,
        title="Bewertungsverteilung"
    )
    st.plotly_chart(fig, use_container_width=True)

# 8. Zeitverlauf der Bewertungen
if 'date' in df_filtered.columns and 'rating_numeric' in df_filtered.columns:
    st.subheader("📈 Bewertungen im Zeitverlauf")

    df_time = df_filtered.dropna(subset=['date'])
    df_time = df_time.sort_values('date')

    fig_time = px.line(
        df_time,
        x='date',
        y='rating_numeric',
        title="Bewertungen über Zeit"
    )
    st.plotly_chart(fig_time, use_container_width=True)

# 9. Rohdaten anzeigen (optional)
with st.expander("📄 Rohdaten anzeigen"):
    st.dataframe(df_filtered)