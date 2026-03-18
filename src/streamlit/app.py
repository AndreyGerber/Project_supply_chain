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


# 3. Titel & Header
st.title("📊 Analyse der Autodoc Kundenbewertungen")
st.markdown("---")

# --- 4. Sidebar & Dynamische Filterung ---
st.sidebar.header("Filter-Optionen")

# Multiselect für die Sterne (nutzt die bereits extrahierte 'rating_numeric')
selected_rating = st.sidebar.multiselect(
    "Bewertung wählen", 
    options=sorted(df['rating_numeric'].unique()), 
    default=sorted(df['rating_numeric'].unique())
)

# WICHTIG: Hier erstellen wir das gefilterte Dataframe 'df_selection'
# Alle folgenden Grafiken nutzen ab jetzt 'df_selection' statt 'df'
df_selection = df[df['rating_numeric'].isin(selected_rating)]

# --- 5. Metriken berechnen (basierend auf Auswahl) ---
if not df_selection.empty:
    durchschnitt = df_selection['rating_numeric'].mean()
    # Wir prüfen, ob 'supplier_response' vorhanden ist (Support-Antwort)
    antwort_rate = df_selection['supplier_response'].notna().mean() * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Reviews (Auswahl)", len(df_selection))
    col2.metric("Ø Rating", f"{durchschnitt:.1f} ⭐")
    col3.metric("Antwort-Rate", f"{antwort_rate:.1f}%")
else:
    st.warning("Bitte wähle mindestens eine Bewertungsstufe in der Sidebar aus.")

st.markdown("---")

# --- 6. Tabs Layout (Visualisierungen) ---
tab1, tab2, tab3 = st.tabs(["📈 Trends", "💬 Kommentare", "📍 Support"])

with tab1:
    st.subheader("Verteilung der Sterne-Bewertungen")
    # Histogramm zeigt jetzt nur die gefilterten Daten
    fig = px.histogram(
        df_selection, 
        x="rating_numeric", 
        color="rating_numeric", 
        title="Häufigkeit der Sterne in der Auswahl",
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("💬 Einblick in die Kundenkommentare")
    search = st.text_input("Suche nach Stichworten (z.B. 'Lieferung' oder 'Preis')", "")
    
    # Filterung der Kommentare innerhalb der Auswahl
    if search:
        results = df_selection[df_selection['review_text'].str.contains(search, case=False, na=False)]
        st.success(f"Gefundene Treffer: {len(results)}")
        st.dataframe(results[['date', 'rating_numeric', 'review_text']], use_container_width=True)
    else:
        st.dataframe(df_selection[['date', 'rating_numeric', 'review_text']].head(10), use_container_width=True)

with tab3:
    st.header("📍 Support- & Standort-Analyse")
    col_a, col_b = st.columns(2)
    
    with col_a:
        # Woher kommen die Kunden in dieser Auswahl?
        top_loc = df_selection['location'].value_counts().head(10)
        fig_loc = px.pie(values=top_loc.values, names=top_loc.index, title="Top 10 Standorte", hole=0.4)
        st.plotly_chart(fig_loc, use_container_width=True)
        
    with col_b:
        # Wie oft wurde auf diese spezifischen Reviews geantwortet?
        df_selection['hat_antwort'] = df_selection['supplier_response'].notna()
        antwort_counts = df_selection['hat_antwort'].value_counts().rename({True: 'Beantwortet', False: 'Offen'})
        fig_resp = px.bar(x=antwort_counts.index, y=antwort_counts.values, title="Antwortstatus der Auswahl", labels={'x': 'Status', 'y': 'Anzahl'})
        st.plotly_chart(fig_resp, use_container_width=True)

# --- 7. Abschluss ---
st.markdown("---")
# Dein persönlicher Gruß bleibt erhalten