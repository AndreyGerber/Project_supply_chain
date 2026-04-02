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
        st.code(list(df.columns))
        
        st.subheader("Data Preview (First 10 rows)")
        st.dataframe(df.head(10), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)


    st.write("### 📋 Dataset Column Overview")

    # 1. HTML & CSS Definition (Volle Kontrolle über Style)
    html_style = """
    <style>
        .custom-table {
            width: 100%;
            border-collapse: collapse;
            font-family: sans-serif;
            color: #31333F;
        }
        .custom-table th, .custom-table td {
            border: 1px solid #e6e9ef;
            padding: 12px;
        }
        /* Header Styling */
        .custom-table th {
            background-color: #f0f2f6;
            font-weight: bold;
        }
        /* ANFORDERUNG 2: Column Name & Daten LINKSBÜNDIG (Breite 70%) */
        .custom-table td:nth-child(1), .custom-table th:nth-child(1) {
            text-align: left;
            width: 70%;
        }
        /* ANFORDERUNG 3: Unique Values & Daten ZENTRIERT (Breite 30%) */
        .custom-table td:nth-child(2), .custom-table th:nth-child(2) {
            text-align: center;
            width: 30%;
        }
        /* Optional: Zeilen-Highlighting beim Drüberfahren */
        .custom-table tr:hover {
            background-color: #f8f9fb;
        }
    </style>
    """

    # 2. Tabellen-Körper dynamisch aufbauen (ANFORDERUNG 1: KEINE ID-SPALTE)
    table_rows = ""
    for col in df.columns:
        unique_count = df[col].nunique()
        table_rows += f"<tr><td>{col}</td><td>{unique_count}</td></tr>"

    # 3. Alles zusammenfügen
    full_html = f"""
    {html_style}
    <table class="custom-table">
        <thead>
            <tr>
                <th>Column Name</th>
                <th>Unique Values (nunique)</th>
            </tr>
        </thead>
        <tbody>
            {table_rows}
        </tbody>
    </table>
    """

    # 4. In Streamlit anzeigen
    st.markdown(full_html, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)




    st.subheader("🔍 Deep Dive: Why are there duplicates?")

    # 1. Identifiziere die System-Antworten (Die "Reply from" Zeilen)
    system_replies = df[df['review_text'].str.contains("Reply from", na=False, case=False)]
    st.write(f"**A. System Replies:** Found {len(system_replies)} rows that are just company responses.")

    if not system_replies.empty:
        st.write("#### 🏢 Companies with System Replies & Examples")
        
        # Gruppieren nach Firma, Zählen und den ersten gefundenen Text nehmen
        company_summary = system_replies.groupby('company')['review_text'].agg(
            Count='count',
            Example='first'
        ).reset_index().sort_values(by='Count', ascending=False)

        # Spalten umbenennen für die Anzeige
        company_summary.columns =
        
        # Darstellung als HTML-Tabelle für die Zentrierung und ohne ID-Spalte
        html_summary = f"""
        <style>
            .reply-table {{ width: 100%; border-collapse: collapse; }}
            .reply-table th, .reply-table td {{ border: 1px solid #e6e9ef; padding: 10px; }}
            .reply-table th {{ background-color: #f0f2f6; }}
            .reply-table td:nth-child(1) {{ width: 25%; text-align: left; }}
            .reply-table td:nth-child(2) {{ width: 15%; text-align: center; }}
            .reply-table td:nth-child(3) {{ width: 60%; text-align: left; font-style: italic; color: #555; }}
        </style>
        <table class="reply-table">
            <thead>
                <tr>
                    <th>Company Name</th>
                    <th>Count</th>
                    <th>Example Text Content</th>
                </tr>
            </thead>
            <tbody>
        """
        for _, row in company_summary.iterrows():
            html_summary += f"<tr><td>{row['Company Name']}</td><td>{row['System Reply Count']}</td><td>{row}</td></tr>"
        
        html_summary += "</tbody></table>"
        
        st.markdown(html_summary, unsafe_allow_html=True)
        
        st.info("💡 **Insight:** These comments clearly show technical headers instead of customer feedback. We will remove them now.")

    
    # 2. Identifiziere echte Text-Duplikate (identische Kommentare)
    # Wir schauen uns nur die Texte an, die KEINE System-Antworten sind
    df_no_replies = df[~df['review_text'].str.contains("Reply from", na=False, case=False)]
    text_counts = df_no_replies['review_text'].value_counts()
    real_duplicates = text_counts[text_counts > 1]

    st.write(f"**B. Genuine Comment Duplicates:** Found {len(real_duplicates)} unique texts that appear multiple times.")

    # 3. Darstellung der echten Duplikate in einer Tabelle
    if not real_duplicates.empty:
        dup_df = real_duplicates.reset_index()
        dup_df.columns = ['Review Content', 'Occurrence Count']
        
        st.dataframe(
            dup_df.head(10), 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "Comment Text": st.column_config.TextColumn("Review Content", width="large"),
                "Occurrence Count": st.column_config.NumberColumn("Count", width="small")
            }
        )

    st.info(f"💡 **Conclusion:** Out of the missing {len(df) - df['review_text'].nunique()} entries, "
            f"{len(system_replies)} are system replies and the rest are repeated customer phrases.")




    st.subheader("🔍 Analysis of Duplicate Comments")
    st.code("""df = df[~df['review_text'].str.contains("Reply from", na=False, case=False)]""", language="python")
   


    # Die Top 5 der am häufigsten vorkommenden identischen Texte
    st.write("Most frequent identical comments:")

    # 1. Daten vorbereiten (Top 5 Duplikate finden)
    # Wir wandeln das Ergebnis von value_counts() direkt in ein DataFrame um
    duplicates_df = df['review_text'].value_counts().head(5).reset_index()
    duplicates_df.columns = ['review_text', 'count']

    # 2. Saubere Darstellung in zwei Spalten
    st.dataframe(
        duplicates_df,
        use_container_width=True,
        hide_index=True,  # Entfernt die ID-Spalte links
        column_config={
            "review_text": st.column_config.TextColumn(
                "Review Content", 
                width="large"
            ),
            "count": st.column_config.NumberColumn(
                "Occurrence (count)", 
                width="small",
                format="%d"
            )
        }
    )


    st.subheader("🧹 Data Cleaning: Removing System Replies")

    # Wir filtern alle Zeilen heraus, die mit "Reply from" beginnen
    initial_count = len(df)
    df = df[~df['review_text'].str.contains("Reply from", na=False, case=False)]
    removed_count = initial_count - len(df)

    st.warning(f"Removed {removed_count} rows containing company replies instead of customer comments.")
    st.session_state['raw_data'] = df # Speicher das gesäuberte DF wieder ab



    st.write("### 📋 Preprocessing Status Overview")

    # 1. Daten für die Tabelle vorbereiten
    column_info = []

    # Liste der Spalten, die wir als "cleaned" markieren wollen
    # (Du kannst diese Liste erweitern, wenn du mehr Spalten bearbeitest)
    cleaned_columns = ['review_text', 'review_text_clean', 'review_text_clean_advanced']

    for col in df.columns:
        # Check, ob die Spalte in unserer Liste der bereinigten Spalten ist
        is_cleaned = "✅" if col in cleaned_columns else "❌"
        
        column_info.append({
            "name": col,
            "count": df[col].nunique(),
            "status": is_cleaned
        })

    # 2. HTML & CSS (Anforderung: 3 Spalten, Zentrierung, Breiten)
    html_table = """
    <style>
        .status-table {
            width: 100%;
            border-collapse: collapse;
            font-family: sans-serif;
            color: #31333F;
        }
        .status-table th, .status-table td {
            border: 1px solid #e6e9ef;
            padding: 12px;
        }
        .status-table th {
            background-color: #f0f2f6;
        }
        /* Spalte 1: Name (links) - 50% */
        .status-table td:nth-child(1), .status-table th:nth-child(1) {
            text-align: left;
            width: 50%;
        }
        /* Spalte 2: Unique (zentriert) - 25% */
        .status-table td:nth-child(2), .status-table th:nth-child(2) {
            text-align: center;
            width: 25%;
        }
        /* Spalte 3: Cleaned (zentriert) - 25% */
        .status-table td:nth-child(3), .status-table th:nth-child(3) {
            text-align: center;
            width: 25%;
        }
    </style>

    <table class="status-table">
        <thead>
            <tr>
                <th>Column Name</th>
                <th>Unique Values</th>
                <th>Cleaned</th>
            </tr>
        </thead>
        <tbody>
    """

    for item in column_info:
        html_table += f"<tr><td>{item['name']}</td><td>{item['count']}</td><td>{item['status']}</td></tr>"

    html_table += "</tbody></table>"

    # 3. Anzeige
    st.markdown(html_table, unsafe_allow_html=True)






    # --- AB HIER GEHT ES GLEICH WEITER MIT DEM CLEANING ---
    st.divider()

else:
    # Falls jemand direkt auf diese Seite surft, ohne die Daten zu laden
    st.error("⚠️ No data found in memory!")
    st.info("Please go back to the **Data Exploration** page to initialize the dataset.")
    
    if st.button("Go to Data Exploration"):
        st.switch_page("pages/01_Data_Exploration.py") # Prüfe den exakten Dateinamen!