import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import re

# 1. NLTK Ressourcen laden (für Stopwords)
@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords')
    nltk.download('punkt')

download_nltk_data()

# 1. Seite konfigurieren (wie bei den anderen Seiten)
st.set_page_config(page_title="Phase 2: Preprocessing", layout="wide")

st.title("🧹 Phase 2: Natural Language Preprocessing (NLP)" )

st.markdown("""
In this step, we prepare our raw review texts for Machine Learning. 
First, let's verify that we have access to the same dataset from the previous phase.
""")

# 2. Daten aus dem "Gedächtnis" (Session State) abrufen
if 'raw_data' in st.session_state:
    # Wir laden die Daten in eine lokale Variable 'df'
    df = st.session_state['raw_data']
    
    st.success(f"✅ Successfully linked to the dataset! ({len(df)} rows loaded)")

    # 3. Rohdaten-Vorschau (Exakt wie auf der Vorseite zur Kontrolle)
    with st.expander("🔍 View Raw Data Columns"):
        st.write("### 📋 Dataset Column Overview")

        # 1. Daten vorbereiten (wie bisher)
        column_info = []
        for col in df.columns:
            column_info.append({
                "Column Name": col,
                "Unique Values (nunique)": df[col].nunique()
            })

        df_info = pd.DataFrame(column_info)

        # 2. Darstellung mit Spaltenkonfiguration
        st.dataframe(
            df_info,
            use_container_width=True,
            hide_index=True,  # Entfernt die ID-Spalte (0, 1, 2...) ganz links
            column_config={
                "Column Name": st.column_config.TextColumn(
                    "Column Name",
                    width="large",   # Hier kannst du "small", "medium" oder "large" wählen
                    help="The name of the feature in the dataset"
                ),
                "Unique Values (nunique)": st.column_config.NumberColumn(
                    "Unique Values (nunique)",
                    width="medium",  # Breite für die Zahlen-Spalte
                    format="%d",     # Zeigt ganze Zahlen ohne Komma an
                    help="Number of unique entries in this column"
                )
            }
        )

        # 3. Zentrierung der Zahlen (CSS-Hack)
        st.markdown("""
            <style>
            /* Richtet alle Zellen in der zweiten Spalte (Zahlen) zentriert aus */
            [data-testid="stTable"] td:nth-child(2), 
            [data-testid="stDataFrame"] td:nth-child(2) {
                text-align: center !important;
            }
            </style>
            """, unsafe_allow_html=True)

    # --- AB HIER GEHT ES GLEICH WEITER MIT DEM CLEANING ---
    st.divider()

else:
    # Falls jemand direkt auf diese Seite surft, ohne die Daten zu laden
    st.error("⚠️ No data found in memory!")
    st.info("Please go back to the **Data Exploration** page to initialize the dataset.")
    
    if st.button("Go to Data Exploration"):
        st.switch_page("pages/01_Data_Exploration.py") # Prüfe den exakten Dateinamen!