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

# 6. Tabs Layout
tab1, tab2, tab3 = st.tabs(["📈 Trends", "💬 Kommentare", "📍 Support"])

with tab1:
    st.subheader("Verteilung der Sterne-Bewertungen")
    fig = px.histogram(df, x="rating_numeric", color="rating_numeric", title="Häufigkeit der Sterne")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("💬 Einblick in die Kundenkommentare")
    search = st.text_input("Suche nach Stichworten", "")
    
    if search:
        results = df[df['review_text'].str.contains(search, case=False, na=False)]
        st.success(f"Gefundene Kommentare: {len(results)}")
        st.dataframe(results[['rating_numeric', 'review_text', 'date']], use_container_width=True)

with tab3:
    st.subheader("📍 Support-Antworten")
    support_df = df[df['supplier_response'].notna()]
    st.dataframe(support_df[['rating_numeric', 'review_text', 'supplier_response', 'date']], use_container_width=True)

