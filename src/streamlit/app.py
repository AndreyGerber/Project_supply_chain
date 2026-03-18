import streamlit as st

st.title("Hallo! Meine Supply Chain App läuft.")
st.write("Wenn du das hier siehst, hat das Deployment geklappt!")

# Definieren des Textes mit HTML-Styling für die Namenimport streamlit as st
import pandas as pd
import plotly.express as px

# Konfiguration der Seite
st.set_page_config(page_title="Autodoc Review Analyse", layout="wide")

# Beispiel-Ladefunktion (Pfade anpassen)
@st.cache_data
def load_data():
    df = pd.read_csv("src/data/clean/reviews_clean.csv")
    # Hier deinen Datensatz laden, z.B. pd.read_csv("deine_datei.csv")
    # Wichtig: Datum in datetime umwandeln
    df['date'] = pd.to_datetime(df['date'])
    return df 

# Titel der Präsentation
st.title("📊 Analyse der Autodoc Kundenbewertungen")
st.markdown("---")

# Sidebar für Filter
st.sidebar.header("Filter-Optionen")
selected_rating = st.sidebar.multiselect("Bewertung wählen", options=[1,2,3,4,5], default=[1,2,3,4,5])

# Aus dem df neue Spalte mit Rating erzeugen (aus "rating_svg" z.B. 5 Sterne = 5, 4 Sterne = 4, etc.)
# 1. Zahl aus dem Text extrahieren (z.B. die 5 aus 'stars-5.svg')
df['rating_numeric'] = df['rating_svg'].str.extract('(\d+)').astype(float)
# 2. Durchschnitt berechnen
durchschnitt = df['rating_numeric'].mean()

# Metriken in der Übersicht
col1, col2, col3 = st.columns(3)
col1.metric("Gesamtanzahl Reviews", "621")
col2.metric("Durchschnittliches Rating", f"{durchschnitt:.1f}")

antwort_rate = df['supplier_response'].notna().mean() * 100
col3.metric("Antwort-Rate", f"{antwort_rate:.1f}%")

st.markdown("---")

# Layout mit Tabs für die Präsentation
tab1, tab2, tab3 = st.tabs()

with tab1:
    st.subheader("Verteilung der Sterne-Bewertungen")
    # Beispiel-Plot: Balkendiagramm der Ratings
    fig = px.histogram(df, x="rating", color="rating", title="Häufigkeit der Sterne")
    st.plotly_chart(fig, use_container_width=True)
    st.info("Hier kannst du zeigen, wie viele 5-Sterne vs. 1-Sterne Bewertungen existieren.")

with tab2:
    st.subheader("💬 Einblick in die Kundenkommentare")
    
    # Suchfeld
    search = st.text_input("Suche nach Stichworten (z.B. 'Versand', 'Teile', 'Preis')", "")
    
    if search:
        # Filtern (wir behandeln NaN Werte in review_text mit na=False)
        results = df[df['review_text'].str.contains(search, case=False, na=False)]
        
        st.success(f"Gefundene Kommentare: {len(results)}")
        
        # Zeige nur die relevanten Spalten an
        st.dataframe(results[['rating', 'review_text', 'date']], use_container_width=True)
    else:
        st.info("Gib oben ein Wort ein, um die Bewertungen zu filtern.")

 st.markdown("---")
    st.subheader("🔝 Häufigste Begriffe")

    # Einfaches Zählen der Wörter (ohne Füllwörter wie 'der', 'die', 'und')
    # Wir nehmen die Top 10
    all_text = " ".join(df['review_text'].fillna("").astype(str)).lower()
    words = pd.Series(all_text.split())
    
    # Filter für unwichtige Wörter (Stopwords) - erweiterbar
    stopwords = ['ich', 'die', 'der', 'und', 'ist', 'das', 'für', 'zu', 'mit', 'auf', 'von']
    top_words = words[~words.isin(stopwords)].value_counts().head(10)
    
    fig_words = px.bar(top_words, x=top_words.values, y=top_words.index, 
                       orientation='h', labels={'x': 'Anzahl', 'y': 'Wort'},
                       title="Top 10 Schlagworte in den Reviews")
    st.plotly_chart(fig_words, use_container_width=True)

  st.markdown("---")
    st.subheader("🎭 Stimmung vs. Sterne")
    
    # Vergleich: Kurze vs. Lange Texte
    df['text_length'] = df['review_text'].str.len()
    fig_len = px.box(df, x='rating', y='text_length', 
                     title="Schreiben unzufriedene Kunden längere Texte?")
    st.plotly_chart(fig_len, use_container_width=True)



with tab3:
    st.header("📍 Geografische Herkunft & Support-Analyse")

    # Spalten-Layout für die Übersicht
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Top Standorte")
        # Zähle die häufigsten Standorte (NaN ausschließen)
        top_locations = df['location'].value_counts().head(10)
        
        if not top_locations.empty:
            fig_loc = px.pie(
                values=top_locations.values, 
                names=top_locations.index, 
                title="Top 10 Herkunftsländer/Städte",
                hole=0.4 # Macht ein Donut-Diagramm daraus
            )
            st.plotly_chart(fig_loc, use_container_width=True)
        else:
            st.info("Keine Standortdaten verfügbar (viele NaN-Werte).")

    with col_b:
        st.subheader("Support-Antwortrate")
        # Berechne, ob eine Antwort existiert (True/False)
        df['hat_antwort'] = df['supplier_response'].notna()
        antwort_counts = df['hat_antwort'].value_counts().rename({True: 'Beantwortet', False: 'Offen'})
        
        fig_resp = px.bar(
            x=antwort_counts.index, 
            y=antwort_counts.values,
            color=antwort_counts.index,
            color_discrete_map={'Beantwortet': '#2ECC71', 'Offen': '#E74C3C'},
            title="Wurde auf die Bewertung reagiert?"
        )
        st.plotly_chart(fig_resp, use_container_width=True)

    st.markdown("---")

    # Vertiefende Analyse: Reagiert der Support eher auf schlechte Bewertungen?
    st.subheader("🎯 Strategie: Worauf antwortet der Support?")
    
    # Gruppiere nach Rating und schaue, wie hoch der Anteil an Antworten ist
    response_strategy = df.groupby('rating')['hat_antwort'].mean() * 100
    
    fig_strat = px.line(
        x=response_strategy.index, 
        y=response_strategy.values,
        markers=True,
        labels={'x': 'Sterne-Rating', 'y': 'Antwort-Quote in %'},
        title="Antwort-Quote nach Sterne-Kategorie"
    )
    st.plotly_chart(fig_strat, use_container_width=True)
    
    st.write("""
    **Analyse-Tipp:** Wenn die Linie bei 1-Sterne-Bewertungen höher ist als bei 5-Sterne-Bewertungen, 
    bedeutet das, dass der Support aktiv am **Beschwerdemanagement** arbeitet.
    """)
# Button zum Download der gefilterten Daten
st.sidebar.download_button("Daten exportieren", data="...", file_name="export.csv



olga_robert_style = '<span style="font-weight: bold; font-size: 1.2em;">' # "1.2em" ist ca. 2 Nummern größer

st.markdown(f'''
Liebe {olga_robert_style}Olga</span> und cooler noch junger (aber auch nicht mehr so jung, wie es mal war) 
{olga_robert_style}Robert</span>. 
Das ist unsere Streamlit Oberfläche. Hier können wir unsere Daten visualisieren und interaktiv mit ihnen arbeiten. 
Ich freue mich schon darauf, gemeinsam mit dir die nächsten Schritte zu gehen und unsere App weiterzuentwickeln!
''', unsafe_allow_html=True)