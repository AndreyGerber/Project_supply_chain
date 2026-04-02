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
        st.write("Current columns in our dataset:")
        st.write("Current columns in your dataset:")
        # Erstellt eine vertikale Liste mit Markdown-Stichpunkten
        for col in df.columns:
            st.markdown(f"- **{col}**")
        
        st.subheader("Data Preview (First 10 rows)")
        st.dataframe(df.head(10), use_container_width=True)

    # --- AB HIER GEHT ES GLEICH WEITER MIT DEM CLEANING ---
    st.divider()

else:
    # Falls jemand direkt auf diese Seite surft, ohne die Daten zu laden
    st.error("⚠️ No data found in memory!")
    st.info("Please go back to the **Data Exploration** page to initialize the dataset.")
    
    if st.button("Go to Data Exploration"):
        st.switch_page("pages/01_Data_Exploration.py") # Prüfe den exakten Dateinamen!