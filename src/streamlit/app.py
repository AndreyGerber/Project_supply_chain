import streamlit as st
import pandas as pd
import re

# --- SEITEN-KONFIGURATION ---
st.set_page_config(page_title="Supply Chain Analytics", page_icon="📊", layout="wide")

# --- NAVIGATION IN DER SIDEBAR ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Gehe zu:", ["Startseite", "Phase 1: Data Exploration", "Phase 2: Preprocessing"])

# --- STARTSEITE (Dein bisheriger Code) ---
if page == "Startseite":
    st.title("🤖 Supply Chain Analytics")
    st.subheader("From Data Scraping to Star Predictions Using Machine Learning")
    
    # ... (Dein Text und die Spalten-Visualisierung hier einfügen) ...
    st.info("Nutze die Sidebar links, um zu Phase 1 zu gelangen.")

# --- PHASE 1: DATA EXPLORATION ---
elif page == "Phase 1: Data Exploration":
    st.title("📊 Phase 1: Data Exploration")
    
    # Beispiel-Daten laden (Ersetze dies durch st.file_uploader oder pd.read_csv)
    # WICHTIG: Wir laden die "schmutzige" Datei!
    raw_data = {
        "Review": ["Toller Service! 😍🚚", "Lieferung zu spät... 😡", "Top Qualität! ⭐⭐⭐"],
        "Stars": [5, 1, 5],
        "Unnötige_Spalte": ["X", "Y", "Z"]
    }
    df_raw = pd.DataFrame(raw_data)

    st.subheader("1. Rohdaten (Original)")
    st.write("So sieht die Datei aus, bevor wir sie anfassen:")
    st.dataframe(df_raw)

    # --- DER REINIGUNGSSCHRITT (Im Code sichtbar!) ---
    st.markdown("---")
    st.subheader("2. Automatisierte Bereinigung")
    
    # Wir kopieren die Daten, um das Original nicht zu verlieren
    df_cleaned = df_raw.copy()
    
    # Schritt A: Emojis entfernen
    df_cleaned['Review'] = df_cleaned['Review'].apply(lambda x: re.sub(r'[^\x00-\x7F]+', '', x))
    
    # Schritt B: Unnötige Spalten löschen
    df_cleaned = df_cleaned.drop(columns=["Unnötige_Spalte"])

    col1, col2 = st.columns(2)
    with col1:
        st.caption("Vorher (mit Emojis/Müll)")
        st.dataframe(df_raw["Review"].head())
    with col2:
        st.caption("Nachher (Clean für ML)")
        st.dataframe(df_cleaned["Review"].head())

    st.success("✅ Die Vorgehensweise ist nun im Code dokumentiert und für den Nutzer nachvollziehbar!")
