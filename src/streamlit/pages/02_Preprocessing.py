import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import re
import streamlit as st
import os

# 1. NLTK Ressourcen laden (für Stopwords)
@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords')
    nltk.download('punkt')

download_nltk_data()

# 2. Seite konfigurieren (wie bei den anderen Seiten)
st.set_page_config(page_title="Phase 2: Preprocessing", layout="wide")

st.title("🧹 Phase 2: Natural Language Preprocessing (NLP)" )

st.markdown("""
In this step, we prepare our raw review texts for Machine Learning. 
First, let's verify that we have access to the same dataset from the previous phase.
""")

# 3. Daten aus dem "Gedächtnis" (Session State) abrufen
if 'raw_data' in st.session_state:
    # Wir laden die Daten in eine lokale Variable 'df'
    df = st.session_state['raw_data']
    
        
    st.success(f"✅ Successfully linked to the dataset! ({len(df)} rows loaded)")

    #  Rohdaten-Vorschau (Exakt wie auf der Vorseite zur Kontrolle)
    with st.expander("🔍 View Raw Data Columns"):
        st.write("Current columns in our dataset:")
        st.code(list(df.columns))
        
        st.subheader("Data Preview (First 10 rows)")
        st.dataframe(df.head(10), use_container_width=True)

else:
    # Falls jemand direkt auf diese Seite surft, ohne die Daten zu laden
    st.error("⚠️ No data found in memory!")
    st.info("Please go back to the **Data Exploration** page to initialize the dataset.")
    
    if st.button("Go to Data Exploration"):
        st.switch_page("pages/01_Data_Exploration.py") # Prüfe den exakten Dateinamen!



st.markdown("<br>", unsafe_allow_html=True)


# 4. Übersicht über die Spalten und deren Einzigartigkeit (nunique)
st.write("### 📋 Dataset Column Overview")

# HTML & CSS Definition (Volle Kontrolle über Style)
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

# Tabellen-Körper dynamisch aufbauen (KEINE ID-SPALTE)
table_rows = ""
for col in df.columns:
    unique_count = df[col].nunique()
    table_rows += f"<tr><td>{col}</td><td>{unique_count}</td></tr>"

# Alles zusammenfügen
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

# In Streamlit anzeigen
st.markdown(full_html, unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)



# 5. Erklärung der Duplikate (Warum haben wir mehr Zeilen als einzigartige Kommentare?)
st.subheader("🔍 Deep Dive: Why are there duplicates in 'review_text'?")

st.code(""" Successfully linked to the dataset! (6443 rows loaded). But only 5471 in "review_text" """, language="python")


#Bild laden

current_dir = os.path.dirname(__file__)

# Falls dieses Skript im Ordner 'pages/' liegt, gehe eine Ebene höher
if "pages" in current_dir:
    parent_dir = os.path.dirname(current_dir)
else:
    parent_dir = current_dir

logo_path = os.path.join(parent_dir, "static", "what_is_it.png")

# 2. Layout mit Spalten (Columns)
# Wir erstellen 3 Spalten. Die mittlere (col2) enthält das Bild.
# Du kannst die Zahlen (1, 2, 1) anpassen, um die Breite der Mitte zu ändern.
col1, col2, col3 = st.columns([1, 2, 1]) 

with col2:
    if os.path.exists(logo_path):
        # Das Bild füllt nun nur die Breite von col2 aus, nicht die ganze Seite
        st.image(
            logo_path, 
            #caption="Project Overview", 
            use_container_width=True
        )
    else:
        st.error(f"❌ File not found at: {logo_path}")

# Optional: Text unter dem Bild (außerhalb der Spalten für volle Breite)
#Bild laden. Abschnitt zum Ende




# Identifiziere die System-Antworten (Die "Reply from" Zeilen)
# Wir suchen nur nach Texten, die mit "Reply from" STARTEN (^)
system_replies = df[df['review_text'].str.contains(r"^Reply from", na=False, case=False, regex=True)]

st.write(f"**A. System Replies:** Found {len(system_replies)} rows that are just company responses.")
st.code("""system_replies = df[df['review_text'].str.contains(r"^Reply from", na=False, case=False, regex=True)]""", language="python")

if not system_replies.empty:
    st.markdown("#### 🏢 Summary of System Replies by Company")

    # 1. Daten aggregieren
    company_summary = system_replies.groupby('company')['review_text'].agg(
        Count='count',
        Example='first'
    ).reset_index().sort_values(by='Count', ascending=False)

    # 2. Das HTML-Gerüst bauen (WICHTIG: Alles muss in EINER Variable sein)
    html_table = """
    <style>
        .summary-table { width: 100%; border-collapse: collapse; font-family: sans-serif; }
        .summary-table th, .summary-table td { border: 1px solid #e6e9ef; padding: 10px; }
        .summary-table th { background-color: #f0f2f6; text-align: left; }
        .summary-table td:nth-child(2) { text-align: center; } /* Count zentriert */
    </style>
    <table class="summary-table">
        <thead>
            <tr>
                <th>Company</th>
                <th>Count</th>
                <th>Example Text Content</th>
            </tr>
        </thead>
        <tbody>
    """

    # 3. Die Zeilen zur Tabelle hinzufügen
    for _, row in company_summary.iterrows():
        html_table += f"<tr><td>{row['company']}</td><td>{row['Count']}</td><td>{row['Example']}</td></tr>"

    # 4. Die Tabelle schließen
    html_table += "</tbody></table>"

    # 5. WICHTIG: Mit unsafe_allow_html=True rendern
    st.markdown(html_table, unsafe_allow_html=True)
    
    st.info(f"💡 **Insight:** Instead of showing all {len(system_replies)} rows, we summarized them by company.")


# --- 1. SCHRITT: SYSTEM-ANTWORTEN ISOLIEREN ---
# Wir suchen alles, was mit "Reply from" beginnt
system_mask = df['review_text'].str.contains(r"^Reply from", na=False, case=False, regex=True)
df_system = df[system_mask]
sys_count = len(df_system) # Das sind deine 503 Zeilen

# --- 2. SCHRITT: ECHTE USER-DUPLIKATE BERECHNEN ---
# Wir nehmen nur die Zeilen, die KEINE System-Antworten sind
df_no_system = df[~system_mask]

# Die Anzahl der "extra Kopien" ist: (Alle User-Zeilen) minus (Einzigartige User-Texte)
unique_user_count = df_no_system['review_text'].nunique()
extra_rows = len(df_no_system) - unique_user_count # Das sind die restlichen Duplikate (z.B. 469)

# Gesamtsumme der zu entfernenden Zeilen (972)
total_identified = sys_count + extra_rows

# --- 3. DARSTELLUNG ABSCHNITT B ---
st.write(f"**B. Genuine Comment Duplicates:** Identified {extra_rows} extra copies of customer phrases.")

# Top 10 Liste der echten Duplikate (ohne "Reply from")
# Wir zählen, wie oft jeder Text im gefilterten df_no_system vorkommt
text_counts = df_no_system['review_text'].value_counts()
real_duplicates = text_counts[text_counts > 1].reset_index()

if not real_duplicates.empty:
    real_duplicates.columns = ['Review Content', 'Occurrence Count']
    st.dataframe(real_duplicates.head(10), use_container_width=True, hide_index=True)

# --- 4. DIE FINALE KORREKTE CONCLUSION ---
st.info(f"""
    💡 **Conclusion:** We have identified all **{total_identified}** redundant entries:
    * **{sys_count}** are automated system replies (starting with 'Reply from').
    * **{extra_rows}** are extra copies of common customer phrases.
    
    Total: {sys_count} + {extra_rows} = **{total_identified}**.
    This explains why we have {len(df)} total rows but only **{unique_user_count}** unique customer comments.
""")



st.subheader("🧹 Data Cleaning: Removing System Replies")

# Wir filtern alle Zeilen heraus, die mit "Reply from" beginnen
initial_count = len(df)
df = df[~df['review_text'].str.contains("Reply from", na=False, case=False)]
removed_count = initial_count - len(df)

st.warning(f"Removed {removed_count} rows containing company replies instead of customer comments.")
st.session_state['raw_data'] = df # Speicher das gesäuberte DF wieder ab



st.markdown("<br>", unsafe_allow_html=True)

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




st.markdown("<br><br>", unsafe_allow_html=True)


#Ab hier die Spalte "date" bearbeiten, um neue Features zu erstellen (Jahr, Monat, Wochentag, Saison, Tageszeit)

# 1. Sicherstellen, dass Daten im Session State vorhanden sind
if 'raw_data' in st.session_state:
    # Schritt A: Echte Kopie erstellen (Original bleibt unberührt)
    df_processed = st.session_state['raw_data'].copy()

    # Schritt B: 'date' in Datetime umwandeln (für die Extraktion)
    df_processed['date'] = pd.to_datetime(df_processed['date'], utc=True)

    # Schritt C: Neue Spalten hinzufügen (Englische Begriffe)
    df_processed['year'] = df_processed['date'].dt.year
    df_processed['month_name'] = df_processed['date'].dt.month_name()
    df_processed['weekday'] = df_processed['date'].dt.day_name()

    # Saison-Logik (Englisch)
    def get_season(month):
        if month in [12, 1, 2]: return 'Winter'
        elif month in [3, 4, 5]: return 'Spring'
        elif month in [6, 7, 8]: return 'Summer'
        else: return 'Autumn'
    
    df_processed['season'] = df_processed['date'].dt.month.apply(get_season)

    # Tageszeit-Logik (Englisch)
    def get_day_period(hour):
        if 5 <= hour < 12: return 'Morning'
        elif 12 <= hour < 17: return 'Afternoon'
        elif 17 <= hour < 21: return 'Evening'
        else: return 'Night'

    df_processed['day_period'] = df_processed['date'].dt.hour.apply(get_day_period)

    # Schritt D: Die ursprüngliche 'date' Spalte löschen
    df_processed = df_processed.drop(columns=['date'])

    # Schritt E: Spalten sortieren (Zeit-Features nach vorne für bessere Übersicht)
    time_cols = ['year', 'month_name', 'weekday', 'season', 'day_period']
    other_cols = [col for col in df_processed.columns if col not in time_cols]
    df_processed = df_processed[time_cols + other_cols]

    # Schritt F: Ergebnis anzeigen (Erste 15 Zeilen)
    st.write("### 🚀 Lets work on our date-data")
    st.dataframe(df_processed.head(15), use_container_width=True)
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    
    # Bestätigung der Dimensionen
        html_status = """
    <style>
        .status-table { 
            width: 100%; 
            border-collapse: collapse; 
            font-family: sans-serif; 
            color: #31333F; 
        }
        /* SCHRIFTGRÖSSE FÜR ÜBERSCHRIFTEN (th) */
        .status-table th { 
            background-color: #f0f2f6; 
            padding: 15px; 
            text-align: left; 
            font-size: 18px;  /* Hier anpassen */
            font-weight: bold;
        }
        /* SCHRIFTGRÖSSE FÜR ZEILEN (td) */
        .status-table td { 
            border: 1px solid #e6e9ef; 
            padding: 15px; 
            text-align: left; 
            font-size: 16px;  /* Hier anpassen */
        }
        /* Zentrierung für Unique Values & Status */
        .status-table td:nth-child(2), .status-table td:nth-child(3) { 
            text-align: center; 
        }
    </style>
    <table class="status-table">
        <thead>
            <tr>
                <th>Column Name</th>
                <th>Unique Values</th>
                <th>Status</th>
            </tr>
        </thead>
        <tbody>
    """

    # ... (deine restliche Schleife bleibt gleich)

    st.markdown(html_status, unsafe_allow_html=True)

else:
    st.error("⚠️ No data found! Please load the dataset on the first page.")

st.markdown("<br><br>", unsafe_allow_html=True)

# 1. Liste der erledigten Spalten
cleaned_cols = ['year', 'month_name', 'weekday', 'season', 'day_period', 'review_text', 'review_text_clean', 'review_text_clean_advanced']  # Füge hier weitere Spalten hinzu, die du bereinigt hast

# 2. HTML-Tabelle zusammenbauen
html_status = """
<style>
    .status-table { width: 100%; border-collapse: collapse; font-family: sans-serif; color: #31333F; }
    .status-table th, .status-table td { border: 1px solid #e6e9ef; padding: 10px; text-align: left; }
    .status-table th { background-color: #f0f2f6; }
</style>
<table class="status-table">
    <thead>
        <tr>
            <th>Column Name</th>
            <th>Unique Values</th>
            <th>Status</th>
        </tr>
    </thead>
    <tbody>
"""

# 3. Zeilen dynamisch generieren
for col in df_processed.columns:
    unique_count = df_processed[col].nunique()
    status_icon = "✅" if col in cleaned_cols else "❌"
    
    html_status += f"<tr><td>{col}</td><td>{unique_count}</td><td>{status_icon}</td></tr>"

html_status += "</tbody></table>"

# 4. DER WICHTIGE TEIL: Nutze st.markdown mit unsafe_allow_html=True
st.write("### 📋 Preprocessing Status Overview")
st.markdown(html_status, unsafe_allow_html=True)
