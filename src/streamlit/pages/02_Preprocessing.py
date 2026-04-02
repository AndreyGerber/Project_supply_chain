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

st.set_page_config(page_title="Phase 2: Preprocessing", layout="wide")
st.title("🧹 Phase 2: Natural Language Preprocessing (NLP)")





# 2. Daten aus dem Session State laden
if 'raw_data' in st.session_state:
    df = st.session_state['raw_data'].copy()
    
    st.info(f"Loaded {len(df)} reviews for processing.")

    # --- POSITION 1: RAW DATA PREVIEW ---
        st.subheader("📄 Raw Data Preview")
        st.info("Direct preview of the filtered dataset:")

        # Wir setzen die Höhe auf 550 Pixel. 
        # Das reicht bei Standard-Schriftgröße für ca. 15 Zeilen + Header + Menüleiste.
        st.dataframe(
            df_filtered.head(76),   # so sieht man bis zur Zeile 75 (inklusive). 
            use_container_width=True,
            height=550 
        )
        
        # Fügt eine Leerzeile ein
        st.markdown("<br>", unsafe_allow_html=True)


    # --- SCHRITT 1: TEXT CLEANING ---
    st.subheader("1. Text Cleaning & Tokenization")
    
    def clean_text(text):
        if not isinstance(text, str): return ""
        text = text.lower() # Kleinschreibung
        text = re.sub(r'[^a-z\s]', '', text) # Nur Buchstaben behalten
        words = text.split()
        # Stopwords entfernen (Englisch)
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if w not in stop_words]
        return " ".join(words)

    if st.button("Start Cleaning Process"):
        with st.spinner("Cleaning text and removing stopwords..."):
            # Wir erstellen eine neue Spalte für den bereinigten Text
            df['clean_review'] = df['review_text'].apply(clean_text)
            df['word_count'] = df['clean_review'].apply(lambda x: len(x.split()))
            st.session_state['processed_data'] = df
            st.success("Cleaning complete!")

    if 'processed_data' in st.session_state:
        df = st.session_state['processed_data']
        st.write(df[['review_text', 'clean_review', 'word_count']].head(10))

        # --- SCHRITT 2: HEATMAP & CORRELATIONS ---
        st.divider()
        st.subheader("2. Correlation Analysis")
        
        # Korrelation zwischen Rating, Textlänge etc.
        # Falls du andere numerische Spalten hast, füge sie hier hinzu
        corr_cols = ['rating', 'word_count', 'verified'] 
        corr_matrix = df[corr_cols].corr()

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)
        
        st.caption("This heatmap shows the relationship between star ratings and text length.")

else:
    st.error("No data found! Please go to 'Data Exploration' first and upload your file.")