import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# --- 1. SEITEN-KONFIGURATION & RESSOURCEN ---
st.set_page_config(page_title="Phase 2: Preprocessing", layout="wide")

@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

download_nltk_data()

# --- 2. HEADER & EINLEITUNG ---
st.title("🧹 Phase 2: Natural Language Preprocessing (NLP)")
st.markdown("""
In diesem Schritt bereiten wir die Review-Texte für das Machine Learning vor.
Zuerst verifizieren wir den Datensatz und identifizieren Rauschen (System-Antworten & Duplikate).
""")

# --- 3. DATENABRUF AUS SESSION STATE ---
if 'raw_data' in st.session_state:
    df = st.session_state['raw_data']
    st.success(f"✅ Verbindung zum Datensatz hergestellt! ({len(df)} Zeilen geladen)")

    # --- 4. DATASET OVERVIEW (HTML TABELLE) ---
    st.write("### 📋 Dataset Spalten-Übersicht")
    
    html_style = """
    <style>
        .custom-table { width: 100%; border-collapse: collapse; font-family: sans-serif; color: #31333F; }
        .custom-table th, .custom-table td { border: 1px solid #e6e9ef; padding: 12px; }
        .custom-table th { background-color: #f0f2f6; font-weight: bold; text-align: left; }
        .col-name { width: 70%; text-align: left; }
        .col-count { width: 30%; text-align: center; }
    </style>
    """

    table_rows = "".join([
        f"<tr><td class='col-name'>{col}</td><td class='col-count'>{df[col].nunique()}</td></tr>" 
        for col in df.columns
    ])

    st.markdown(f"{html_style}<table class='custom-table'><thead><tr><th>Column Name</th><th>Unique Values</th></tr></thead>"
                f"<tbody>{table_rows}</tbody></table><br>", unsafe_allow_html=True)

    # --- 5. ANALYSE: SYSTEM ANTWORTEN & DUPLIKATE ---
    st.subheader("🔍 Deep Dive: Warum gibt es Duplikate?")
    
    # Logik-Definition
    system_mask = df['review_text'].str.contains(r"^Reply from", na=False, case=False, regex=True)
    df_system = df[system_mask]
    df_no_system = df[~system_mask]
    
    # Echte Text-Duplikate im Rest berechnen
    extra_rows_count = len(df_no_system) - df_no_system['review_text'].nunique()
    total_to_remove = len(df_system) + extra_rows_count

    # A. System-Antworten Zusammenfassung
    st.write(f"**A. System-Antworten:** {len(df_system)} Zeilen sind reine Firmen-Antworten.")
    
    if not df_system.empty:
        company_summary = df_system.groupby('company')['review_text'].agg(['count', 'first']).reset_index()
        company_summary.columns = ['Company', 'Count', 'Example Text']
        st.dataframe(company_summary.sort_values('Count', ascending=False), use_container_width=True, hide_index=True)

    # B. Echte User-Duplikate
    st.write(f"**B. Echte Kommentar-Duplikate:** {extra_rows_count} überflüssige Kopien von Kunden-Phrasen gefunden.")
    
    # Top 5 Duplikate anzeigen
    top_duplicates = df_no_system['review_text'].value_counts().head(5).reset_index()
    top_duplicates.columns = ['Review Content', 'Occurrence Count']
    
    st.dataframe(
        top_duplicates,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Review Content": st.column_config.TextColumn("Review Content", width="large"),
            "Occurrence Count": st.column_config.NumberColumn("Anzahl")
        }
    )

    # Fazit Box
    st.info(f"""
        💡 **Fazit:** Wir haben insgesamt **{total_to_remove}** Einträge identifiziert, die entfernt werden sollten:
        * **{len(df_system)}** automatisierte System-Antworten.
        * **{extra_rows_count}** identische Kunden-Kommentare.
        
        Ziel-Datensatz nach Bereinigung: **{df['review_text'].nunique()}** einzigartige Reviews.
    """)

else:
    st.warning("⚠️ Keine Daten im Session State gefunden. Bitte gehen Sie zurück zu Phase 1.")

# --- 6. AUSBLICK ---
st.markdown("---")
st.write("Möchten Sie nun mit der **Text-Bereinigung** (Stopwords, Sonderzeichen) fortfahren?")
