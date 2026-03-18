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
    
    st.markdown("---")
    st.subheader("🔝 Häufigste Begriffe")
    all_text = " ".join(df['review_text'].fillna("").astype(str)).lower()
    words = pd.Series(all_text.split())
    stopwords = ['ich', 'die', 'der', 'und', 'ist', 'das', 'für', 'zu', 'mit', 'auf', 'von', 'man']
    top_words = words[~words.isin(stopwords)].value_counts().head(10)
    
    fig_words = px.bar(top_words, x=top_words.values, y=top_words.index, orientation='h', title="Top 10 Schlagworte")
    st.plotly_chart(fig_words, use_container_width=True)

with tab3:
    st.header("📍 Support-Analyse")
    col_a, col_b = st.columns(2)
    with col_a:
        top_loc = df['location'].value_counts().head(10)
        fig_loc = px.pie(values=top_loc.values, names=top_loc.index, title="Top Standorte", hole=0.4)
        st.plotly_chart(fig_loc, use_container_width=True)
    with col_b:
        df['hat_antwort'] = df['supplier_response'].notna()
        antwort_counts = df['hat_antwort'].value_counts().rename({True: 'Beantwortet', False: 'Offen'})
        fig_resp = px.bar(x=antwort_counts.index, y=antwort_counts.values, title="Antwortstatus")
        st.plotly_chart(fig_resp, use_container_width=True)

# 7. Dein persönlicher Gruß (unten fixiert)
st.markdown("---")
olga_robert_style = '<span style="font-weight: bold; color: #ff4b4b; font-size: 1.2em;">'
st.markdown(f'''
Liebe {olga_robert_style}Olga</span> und cooler {olga_robert_style}Robert</span>. 
Das ist unsere Streamlit Oberfläche für die Supply Chain Analyse!
''', unsafe_allow_html=True)