import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import re
import streamlit as st
import os
import plotly.express as px

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




# 3. Retrieve data from memory (Session State) (from "01_Data_Exploration.py")
if 'raw_data' in st.session_state:
    # Wir nehmen die Daten und führen die gleiche Bereinigung wie auf Seite 1 aus
    df = st.session_state['raw_data'].copy()
    
    # Extraktion der Zahl aus dem SVG-String (für die 'rating' Spalte)
    if 'rating_svg' in df.columns:
        df['rating'] = df['rating_svg'].astype(str).str.extract('(\d+)').fillna(0).astype(int)
        
    # Entferne die störenden technischen Spalten vor der Anzeige
    columns_to_drop = ['rating_numeric', 'rating_svg']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    st.success(f"✅ Successfully linked to the dataset! ({len(df)} rows loaded)")

    # Rohdaten-Vorschau (Exakt wie auf der Vorseite)
    with st.expander("🔍 View Raw Data Columns"):
        st.write("Current columns in our dataset:")
        st.code(list(df.columns))
        
        st.subheader("Data Preview (First 10 rows)")
        # Zeige die ersten 10 Zeilen an, so wie im Screenshot der ersten Seite
        st.dataframe(df.head(10), use_container_width=True)

else:
    # Falls die Daten nicht im Speicher sind
    st.error("⚠️ No data found in memory!")
    st.info("Please go back to the **Data Exploration** page to initialize the dataset.")
    
    if st.button("Go to Data Exploration"):
        st.switch_page("pages/01_Data_Exploration.py")







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

# Daten für die Tabelle vorbereiten
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

#Zwei leere Zeilen und darunter eine Linie für die optische Trennung einfügen
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")


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
    info_html = f"""
    <div style="
        background-color: #e8f4f8; 
        border-radius: 5px; 
        padding: 15px; 
        border: 1px solid #c5e1eb; 
        color: #1e3a45; 
        font-size: 20px; 
        font-family: sans-serif;
        margin-bottom: 20px;">
        ℹ️ The new dataframe has <b>{df_processed.shape[0]}</b> rows and <b>{df_processed.shape[1]}</b> columns.
    </div>
    """

    st.markdown(info_html, unsafe_allow_html=True)
else:
    st.error("⚠️ No data found! Please load the dataset on the first page.")

st.markdown("<br><br>", unsafe_allow_html=True)


#den Zusammenhang zwischen dem rating (Sterne) und den neuen Zeit-Features (Jahr, Monat, Wochentag, Saison, Tageszeit) analysieren und visualisieren

# --- ANALYSE-SEKTION: TIME & RATING ---
st.divider()
st.header("🕵️ Correlation Analysis")

# --- ERSTE REIHE: DAY PERIOD & COMBINED YEAR CHART ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Rating by Day Period")
    # Daten aggregieren
    rating_period = df_processed.groupby('day_period')['rating'].mean().reset_index()
    period_order = ['Morning', 'Afternoon', 'Evening', 'Night']
    rating_period['day_period'] = pd.Categorical(rating_period['day_period'], categories=period_order, ordered=True)
    rating_period = rating_period.sort_values('day_period')

    fig_day = px.bar(
        rating_period, x='day_period', y='rating',
        text=rating_period['rating'].round(2),
        color='rating', color_continuous_scale='RdYlGn', range_color=[1, 5],
        title="Average Rating per Time of Day"
    )
    fig_day.update_traces(textposition='outside')
    
    # HÖHE ANGLEICHEN & RÄNDER OPTIMIEREN
    fig_day.update_layout(
        yaxis_range=[1, 5.5], 
        height=500,           
        margin=dict(l=50, r=50, t=80, b=50)
    )
    st.plotly_chart(fig_day, use_container_width=True)

with col2:
    st.subheader("2. Rating Trend & Review Volume by Year")
    
    import plotly.graph_objects as go
    year_stats = df_processed.groupby('year')['rating'].agg(['mean', 'count']).reset_index()
    year_stats.columns = ['year', 'avg_rating', 'review_count']
    
    fig_combined = go.Figure()

    # Balken (Rechte Achse)
    fig_combined.add_trace(go.Bar(
        x=year_stats['year'],
        y=year_stats['review_count'],
        name="Number of Reviews",
        marker_color='rgba(100, 149, 237, 0.3)',
        yaxis='y2'
    ))

    # Linie (Linke Achse)
    fig_combined.add_trace(go.Scatter(
        x=year_stats['year'],
        y=year_stats['avg_rating'],
        name="Avg Rating",
        mode='lines+markers+text',
        line=dict(color='firebrick', width=3),
        text=year_stats['avg_rating'].round(2),
        textposition='top center',
        yaxis='y'
    ))

    fig_combined.update_layout(
        title="Avg Rating vs. Volume per Year",
        xaxis=dict(type='category', title="Year"),
        yaxis=dict(title="Average Rating", range=[1, 5], side="left"),
        yaxis2=dict(
            title="Number of Reviews", 
            overlaying="y", 
            side="right", 
            range=[0, 2500], 
            dtick=250, 
            showgrid=False
        ),
        legend=dict(x=0.3, y=0.3, bgcolor="rgba(255,255,255,0.6)"),
        
        # EXAKT DIE GLEICHE HÖHE UND RÄNDER WIE LINKS
        height=500, 
        margin=dict(l=50, r=50, t=80, b=50)
    )
    st.plotly_chart(fig_combined, use_container_width=True)


# --- ZWEITE REIHE: HEATMAPS (ABSOLUT VS RELATIV) ---
st.markdown("### 🌡️ Seasonal Rating Distribution")
col3, col4 = st.columns(2)

# Heatmap Daten vorbereiten
# Hinweis: Stelle sicher, dass 'Autumn' in deiner get_season Funktion richtig geschrieben ist!
season_order = ['Spring', 'Summer', 'Autumn', 'Winter']
heatmap_abs = pd.crosstab(df_processed['rating'], df_processed['season'])

# Sicherstellen, dass alle Saisons da sind, auch wenn Daten fehlen könnten
for s in season_order:
    if s not in heatmap_abs.columns: heatmap_abs[s] = 0

heatmap_abs = heatmap_abs.reindex(columns=season_order).sort_index(ascending=False)

# Relative Heatmap berechnen
heatmap_rel = heatmap_abs.div(heatmap_abs.sum(axis=0), axis=1) * 100

with col3:
    st.subheader("3. Absolute Volume (Counts)")
    fig_abs = px.imshow(
        heatmap_abs, text_auto=True, aspect="auto",
        color_continuous_scale='YlGnBu',
        title="Total Reviews (Rating vs. Season)",
        labels=dict(x="Season", y="Rating", color="Count")
    )
    st.plotly_chart(fig_abs, use_container_width=True)

with col4:
    st.subheader("4. Relative Distribution (%)")
    fig_rel = px.imshow(
        heatmap_rel, 
        text_auto=".1f", 
        aspect="auto",
        color_continuous_scale='Viridis',
        title="Percentage of Ratings per Season",
        labels=dict(x="Season", y="Rating", color="Percentage (%)")
    )
    st.plotly_chart(fig_rel, use_container_width=True)












# Status-Tabelle mit den neuen Spalten (Jahr, Monat, Wochentag, Saison, Tageszeit) und den alten Spalten (review_text, etc.) erstellen
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
st.markdown("<br><br>", unsafe_allow_html=True)


st.markdown("---")
# ab hier wird die Spalte "location" bearbeitet, um neue Features zu erstellen (Stadt, Bundesland, Land)
st.subheader("📍 Location Analysis")

# Zähle die Häufigkeit der Standorte
location_counts = df_processed['location'].value_counts(dropna=False).reset_index()
location_counts.columns = ['Location', 'Count']

# Zeige die Top 25 an
st.write("Top 25 Locations (including missing values):")
st.dataframe(location_counts.head(25), use_container_width=True)






# 1. Daten vorbereiten (WICHTIG: dropna=False, damit die 894 'None' gezählt werden)
top_locations = df_processed['location'].value_counts(dropna=False).head(15).reset_index()
top_locations.columns = ['Location', 'Count']

# 2. 'None' (NaN) Werte für die Anzeige in 'Unknown' umbenennen
top_locations['Location'] = top_locations['Location'].fillna('Unknown')

# 3. Das Diagramm erstellen (Einfarbig Blau)
fig = px.bar(
    top_locations, 
    x='Location', 
    y='Count', 
    title='📍 Top 15 Review Locations (including Unknown)',
    text='Count',
    # Ein festes, sauberes Blau ohne Farbverlauf
    color_discrete_sequence=['#636EFA'] 
)

# 4. Design-Anpassungen (Schrift & Winkel)
fig.update_layout(
    xaxis_tickangle=-45,
    font=dict(size=14),
    height=500,
    xaxis_title="City / Location",
    yaxis_title="Number of Reviews",
    template="plotly_white",
    showlegend=False
)

# 5. In Streamlit anzeigen
st.plotly_chart(fig, use_container_width=True)




# 1. Daten vorbereiten (Nur Top-Länder)
top_locs = df_processed['location'].value_counts().head(10).index
df_sub = df_processed[df_processed['location'].isin(top_locs)]

# 2. Eine Pivot-Tabelle erstellen: Länder vs. JEDE Rating-Stufe (1-5)
# Das zählt, wie oft jedes Rating in jedem Land vorkommt
pivot_matrix = df_sub.groupby(['location', 'rating']).size().unstack(fill_value=0)

# 3. Normalisieren (Prozentual pro Zeile), damit DE nicht alles dominiert
pivot_norm = pivot_matrix.div(pivot_matrix.sum(axis=1), axis=0) * 100

# 4. Die Heatmap zeichnen (Jetzt siehst du alle 5 Stufen!)
fig = px.imshow(
    pivot_norm,
    labels=dict(x="Rating (Stars)", y="Location", color="Percentage %"),
    x=['1 Star', '2 Stars', '3 Stars', '4 Stars', '5 Stars'],
    y=pivot_norm.index,
    color_continuous_scale='RdYlGn', # Rot zu Grün
    text_auto='.1f', # Zeigt die Prozente im Kästchen
    aspect="auto"
)

fig.update_layout(title="🎯 Detailed Rating Distribution per Location", font=dict(size=14))
st.plotly_chart(fig, use_container_width=True)




st.markdown("<br><br>", unsafe_allow_html=True)
# 1. Die Spalte 'location' aus dem bearbeiteten DF entfernen
df_processed = df_processed.drop(columns=['location'])

# 2. Update deiner Status-Tabelle (wir entfernen sie aus der Liste der Spalten)
# Sie wird nun gar nicht mehr in der Tabelle auftauchen.
st.success("✅ Column 'location' was successfully dropped.")

# 3. Den aktuellen Stand des Dataframes kurz anzeigen (Kontrolle)
st.dataframe(df_processed.head(5), use_container_width=True)


st.markdown("<br><br>", unsafe_allow_html=True)
# 1. Listen definieren
cleaned_cols = ['year', 'month_name', 'weekday', 'season', 'day_period', 'review_text', 'review_text_clean', 'review_text_clean_advanced']  # Spalten, die wir bereinigt haben
dropped_cols = ['location']

# 2. Den HTML-Block EXAKT zusammenbauen
html_status = """
<style>
    .status-table { width: 100%; border-collapse: collapse; font-family: sans-serif; color: #31333F; }
    .status-table th, .status-table td { border-bottom: 1px solid #f0f2f6; padding: 12px; text-align: left; font-size: 14px; }
    .status-table th { background-color: #f0f2f6; font-weight: bold; }
    .strikethrough { text-decoration: line-through; color: #9e9e9e; opacity: 0.6; }
</style>
<table class="status-table">
    <thead>
        <tr>
            <th>Column Name</th>
            <th>Status</th>
        </tr>
    </thead>
    <tbody>
"""

# 3. Schleife über alle Spalten für die Zeilen
for col in list(df_processed.columns) + dropped_cols:
    if col in dropped_cols:
        html_status += f"<tr><td class='strikethrough'>{col}</td><td>🗑️ (Dropped)</td></tr>"
    elif col in cleaned_cols:
        html_status += f"<tr><td>{col}</td><td>✅</td></tr>"
    else:
        html_status += f"<tr><td>{col}</td><td>❌</td></tr>"

html_status += "</tbody></table>"

# 4. DER ENTSCHEIDENDE BEFEHL (KEIN st.write nutzen!)
st.markdown(html_status, unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")






#ab hier wird die Spalte "verified" bearbeitet, um neue Features zu erstellen (True/False, Ja/Nein, 1/0) 
#und die Spalte "review_text" wird bereinigt (cleaned) und in "review_text_clean" gespeichert. 
#Es wird auch eine erweiterte Bereinigung durchgeführt, um Emojis und Sonderzeichen zu entfernen, 
#und das Ergebnis wird in "review_text_clean_advanced" gespeichert.

# 1. Daten vorbereiten
# Wir gruppieren nach 'verified' und 'rating' und zählen die Vorkommen
verified_analysis = df_processed.groupby(['verified', 'rating']).size().reset_index(name='count')

# Um die Grafik schöner zu machen, benennen wir 0 und 1 um
verified_analysis['status'] = verified_analysis['verified'].map({0: 'Not Verified (0)', 1: 'Verified (1)'})

# 2. Ein gestapeltes Balkendiagramm erstellen (Prozentual für bessere Vergleichbarkeit)
fig_ver = px.bar(
    verified_analysis, 
    x='status', 
    y='count', 
    color='rating',
    title='🛡️ Verified Status vs. Rating Distribution',
    labels={'count': 'Number of Reviews', 'status': 'Verification Status', 'rating': 'Stars'},
    barmode='relative', # 'group' für nebeneinander, 'relative' für gestapelt
    color_continuous_scale='RdYlGn' # Rot für 1 Stern, Grün für 5 Sterne
)

# 3. Design-Anpassungen
fig_ver.update_layout(
    font=dict(size=14),
    xaxis_title="",
    yaxis_title="Count of Reviews",
    legend_title="Rating"
)

# 4. In Streamlit anzeigen
st.plotly_chart(fig_ver, use_container_width=True)

# 5. Statistische Kurzzusammenfassung
avg_verified = df_processed[df_processed['verified'] == 1]['rating'].mean()
avg_not_verified = df_processed[df_processed['verified'] == 0]['rating'].mean()

st.info(f"""
💡 **Quick Stats:**
* Average Rating (Verified): **{avg_verified:.2f} ⭐**
* Average Rating (Not Verified): **{avg_not_verified:.2f} ⭐**
""")




# 1. Daten vorbereiten
# Wir gruppieren nach 'verified' und 'rating' und berechnen die Prozente pro Gruppe
pivot_ver = df_processed.groupby(['verified', 'rating']).size().unstack(fill_value=0)
pivot_ver_norm = pivot_ver.div(pivot_ver.sum(axis=1), axis=0) * 100

# Namen für die Achse schöner machen
pivot_ver_norm.index = ['Not Verified (0)', 'Verified (1)']

# 2. Die Heatmap erstellen
fig_ver_heat = px.imshow(
    pivot_ver_norm,
    labels=dict(x="Rating (Stars)", y="Verification Status", color="Percentage %"),
    x=['1 Star', '2 Stars', '3 Stars', '4 Stars', '5 Stars'],
    y=pivot_ver_norm.index,
    color_continuous_scale='RdYlGn', # Rot (schlecht) zu Grün (gut)
    text_auto='.1f', # Zeigt Prozente direkt in den Kacheln
    aspect="auto"
)

# 3. Design-Feinschliff
fig_ver_heat.update_layout(
    title="🎯 Heatmap: Verified Status vs. Rating Distribution (%)",
    font=dict(size=14),
    height=400
)

# 4. In Streamlit anzeigen
st.plotly_chart(fig_ver_heat, use_container_width=True, key="heatmap_percent")

# 5. Die Spalte 'verified' in der Status-Tabelle abhaken
if 'verified' not in cleaned_cols:
    cleaned_cols.append('verified')


    # 1. Daten vorbereiten
# Wir gruppieren nach 'verified' und 'rating' und berechnen die Prozente pro Gruppe
pivot_ver = df_processed.groupby(['verified', 'rating']).size().unstack(fill_value=0)
pivot_ver_norm = pivot_ver.div(pivot_ver.sum(axis=1), axis=0) * 100

# Namen für die Achse schöner machen
pivot_ver_norm.index = ['Not Verified (0)', 'Verified (1)']

# 2. Die Heatmap erstellen
fig_ver_heat = px.imshow(
    pivot_ver_norm,
    labels=dict(x="Rating (Stars)", y="Verification Status", color="Percentage %"),
    x=['1 Star', '2 Stars', '3 Stars', '4 Stars', '5 Stars'],
    y=pivot_ver_norm.index,
    color_continuous_scale='RdYlGn', # Rot (schlecht) zu Grün (gut)
    text_auto='.1f', # Zeigt Prozente direkt in den Kacheln
    aspect="auto"
)

# 3. Design-Feinschliff
fig_ver_heat.update_layout(
    title="🎯 Heatmap: Verified Status vs. Rating Distribution (%)",
    font=dict(size=14),
    height=400
)

# 4. In Streamlit anzeigen
st.plotly_chart(fig_ver_heat, use_container_width=True)

# 5. Die Spalte 'verified' in der Status-Tabelle abhaken
if 'verified' not in cleaned_cols:
    cleaned_cols.append('verified')



pivot_ver2 = df_processed.groupby(['verified', 'rating']).size().unstack(fill_value=0)
pivot_ver2.index = ['Not Verified (0)', 'Verified (1)']

# 2. Deine Heatmap (die schon im Screenshot steht)
fig_ver_abs = px.imshow(
    pivot_ver2, # Hier nutzt du jetzt die neue Variable
    labels=dict(x="Rating (Stars)", y="Verification Status", color="Count of Reviews"),
    x=['1 Star', '2 Stars', '3 Stars', '4 Stars', '5 Stars'],
    y=pivot_ver2.index,
    color_continuous_scale='Blues',
    text_auto=True,
    aspect="auto"
)

fig_ver_abs.update_layout(
    title="📊 Absolute Counts: Verified Status vs. Rating",
    font=dict(size=14)
)

st.plotly_chart(fig_ver_abs, use_container_width=True, key="heatmap_absolute")


st.markdown("---")
st.markdown("<br><br>", unsafe_allow_html=True)
st.info("💡 **Feature Selection:** The 'company' column will been removed too, as it shows no significant correlation with the rating and could introduce model bias.")



#Spalte "company" löschen, da sie für die Analyse nicht relevant ist (sie könnte sogar zu Datenlecks führen, da es sich um eine ID handelt).
# 1. Listen definieren
cleaned_cols = ['year', 'month_name', 'weekday', 'season', 'day_period', 'review_text', 'verified', 'review_text_clean', 'review_text_clean_advanced']
dropped_cols = ['location', 'company']

# 2. Den HTML-String OHNE Einrückung am Zeilenanfang bauen
html_status = """<style>
.status-table { width: 100%; border-collapse: collapse; font-family: sans-serif; color: #31333F; }
.status-table th, .status-table td { border-bottom: 1px solid #f0f2f6; padding: 12px; text-align: left; font-size: 16px; }
.status-table th { background-color: #f0f2f6; font-weight: bold; }
.strikethrough { text-decoration: line-through; color: #9e9e9e; opacity: 0.7; font-style: italic; }
</style>
<table class="status-table">
<thead><tr><th>Column Name</th><th>Unique Values</th><th>Status</th></tr></thead>
<tbody>"""

# 3. Schleife über alle Spalten
display_cols = list(df_processed.columns) + [c for c in dropped_cols if c not in df_processed.columns]

for col in display_cols:
    is_dropped = col in dropped_cols
    row_class = 'class="strikethrough"' if is_dropped else ''
    u_count = df_processed[col].nunique() if col in df_processed.columns else "-"
    status_icon = "🗑️" if is_dropped else ("✅" if col in cleaned_cols else "❌")
    
    html_status += f'<tr {row_class}><td>{col}</td><td>{u_count}</td><td>{status_icon}</td></tr>'

html_status += "</tbody></table>"

st.markdown(html_status, unsafe_allow_html=True)

st.markdown("---")
st.markdown("<br><br>", unsafe_allow_html=True)




# supplier response analysieren, ob sie einen Einfluss auf die Bewertung hat (z.B. haben Kunden, die eine Antwort erhalten haben, tendenziell bessere Bewertungen?)



# 1. Daten transformieren: Hat geantwortet (1) oder nicht (0)
# Wir prüfen, ob der Wert in 'supplier_response' leer (NaN) oder None ist
df_processed['has_response'] = df_processed['supplier_response'].notna().astype(int)

# 2. Pivot-Tabelle erstellen: Antwort-Status vs. Rating
pivot_resp = df_processed.groupby(['has_response', 'rating']).size().unstack(fill_value=0)

# 3. Normalisierung (Prozentual pro Zeile), um Trends trotz unterschiedlicher Mengen zu sehen
pivot_resp_norm = pivot_resp.div(pivot_resp.sum(axis=1), axis=0) * 100
pivot_resp_norm.index = ['No Response (0)', 'Has Response (1)']

# 4. Die Heatmap erstellen
fig_resp = px.imshow(
    pivot_resp_norm,
    labels=dict(x="Rating (Stars)", y="Supplier Response Status", color="Percentage %"),
    x=['1 Star', '2 Stars', '3 Stars', '4 Stars', '5 Stars'],
    y=pivot_resp_norm.index,
    color_continuous_scale='RdYlGn', 
    text_auto='.1f',
    aspect="auto"
)

fig_resp.update_layout(
    title="🎯 Heatmap: Influence of Supplier Response on Ratings (%)",
    font=dict(size=14),
    height=400
)

st.plotly_chart(fig_resp, use_container_width=True)

# 5. Statistischer Check: Durchschnitts-Rating vergleichen
avg_resp = df_processed[df_processed['has_response'] == 1]['rating'].mean()
avg_no_resp = df_processed[df_processed['has_response'] == 0]['rating'].mean()

st.info(f"""
💡 **Quick Insight:**
* Average Rating with Response: **{avg_resp:.2f} ⭐**
* Average Rating without Response: **{avg_no_resp:.2f} ⭐**
""")



# 1. Daten vorbereiten (Absolutzahlen)
# Wir nutzen die Spalte 'has_response' (0 = Nein, 1 = Ja)
pivot_resp_abs = df_processed.groupby(['has_response', 'rating']).size().unstack(fill_value=0)
pivot_resp_abs.index = ['No Response (0)', 'Has Response (1)']

# 2. Die Heatmap mit Absolutzahlen erstellen
fig_resp_abs = px.imshow(
    pivot_resp_abs,
    labels=dict(x="Rating (Stars)", y="Supplier Response Status", color="Count of Reviews"),
    x=['1 Star', '2 Stars', '3 Stars', '4 Stars', '5 Stars'],
    y=pivot_resp_abs.index,
    color_continuous_scale='Blues', # Blau für Mengenverteilung
    text_auto=True,                 # Zeigt die echten Ganzzahlen in den Feldern
    aspect="auto"
)

fig_resp_abs.update_layout(
    title="📊 Absolute Counts: Supplier Response vs. Rating",
    font=dict(size=14),
    height=400
)

# 3. Anzeige in Streamlit
st.plotly_chart(fig_resp_abs, use_container_width=True, key="response_heatmap_absolute")






# 1. Sicherheitskopie für die Business-Analyse erstellen (bevor wir löschen)
df_analytics_copy = df_processed.copy()







# 2. Professionelle Info-Box (Business Insight)
st.info("""
    💡 **Business Intelligence Insight:** 
    Companies show a strong reactive pattern: they prioritize responding to **negative reviews** (1-star) 
    as part of crisis management, while positive feedback often remains unacknowledged. 
    To avoid 'Data Leakage' in our prediction model, this post-event feature will now be removed.
""")




# 1. Wir nutzen unsere 'df_analytics_copy', da dort die Antwort noch drin ist!
df_analysis = df_analytics_copy.copy()
df_analysis['has_response'] = df_analysis['supplier_response'].notna().astype(int)

# 2. Pivot-Tabelle: Verified vs. Response
# (Wir schauen: Wie viel % der Verifizierten bekommen eine Antwort?)
pivot_v_r = df_analysis.groupby(['verified', 'has_response']).size().unstack(fill_value=0)
pivot_v_r_norm = pivot_v_r.div(pivot_v_r.sum(axis=1), axis=0) * 100

# Namen für die Achsen
pivot_v_r_norm.index = ['Not Verified (0)', 'Verified (1)']
pivot_v_r_norm.columns = ['No Response', 'Has Response']

# 3. Die Heatmap erstellen
fig_vr = px.imshow(
    pivot_v_r_norm,
    labels=dict(x="Company Reaction", y="Customer Status", color="Percentage %"),
    color_continuous_scale='Purples', # Eine neue Farbe für neue Erkenntnisse
    text_auto='.1f',
    title="🛡️ Response Strategy: Do Companies care more about Verified Customers?"
)

st.plotly_chart(fig_vr, use_container_width=True)

# 4. Der "Maschinen-Check": Werden die 'Echten' bevorzugt?
rate_v = pivot_v_r_norm.loc['Verified (1)', 'Has Response']
rate_nv = pivot_v_r_norm.loc['Not Verified (0)', 'Has Response']

st.info(f"""
    📊 **Result:** 
    * **{rate_v:.1f}%** of verified customers got a reply.
    * **{rate_nv:.1f}%** of unverified customers got a reply.
""")












df_processed = df_processed.drop(columns=['supplier_response', 'has_response'])
cleaned_cols = ['year', 'month_name', 'weekday', 'season', 'day_period', 'review_text', 'verified', 'review_text_clean', 'review_text_clean_advanced']
dropped_cols = ['location', 'company', 'supplier_response', 'has_response']  # 'has_response' wird hier hinzugefügt, da es ein abgeleitetes Feature ist, das auf 'supplier_response' basiert

# 2. Den HTML-String OHNE Einrückung am Zeilenanfang bauen
html_status = """<style>
.status-table { width: 100%; border-collapse: collapse; font-family: sans-serif; color: #31333F; }
.status-table th, .status-table td { border-bottom: 1px solid #f0f2f6; padding: 12px; text-align: left; font-size: 16px; }
.status-table th { background-color: #f0f2f6; font-weight: bold; }
.strikethrough { text-decoration: line-through; color: #9e9e9e; opacity: 0.7; font-style: italic; }
</style>
<table class="status-table">
<thead><tr><th>Column Name</th><th>Unique Values</th><th>Status</th></tr></thead>
<tbody>"""

# 3. Schleife über alle Spalten
display_cols = list(df_processed.columns) + [c for c in dropped_cols if c not in df_processed.columns]

for col in display_cols:
    is_dropped = col in dropped_cols
    row_class = 'class="strikethrough"' if is_dropped else ''
    u_count = df_processed[col].nunique() if col in df_processed.columns else "-"
    status_icon = "🗑️" if is_dropped else ("✅" if col in cleaned_cols else "❌")
    
    html_status += f'<tr {row_class}><td>{col}</td><td>{u_count}</td><td>{status_icon}</td></tr>'

html_status += "</tbody></table>"

st.markdown(html_status, unsafe_allow_html=True)


st.markdown("---")
st.markdown("<br><br>", unsafe_allow_html=True)





# Abalyse von "company size" (Anzahl der Bewertungen pro Unternehmen) und "review length" (Anzahl der Wörter pro Bewertung) in Bezug auf die Bewertung (rating)

# 1. Daten für die Verteilung der Firmen-Webseiten vorbereiten
site_counts = df_processed['company_site'].value_counts().reset_index()
site_counts.columns = ['Company Site', 'Review Count']

# 2. Vertikales Balkendiagramm erstellen (Einfarbig Blau für den sauberen Look)
fig_site = px.bar(
    site_counts, 
    x='Company Site', 
    y='Review Count', 
    title='🌐 Distribution of Reviews by Company Site',
    text='Review Count',
    color_discrete_sequence=['#636EFA']
)

# 3. Design-Anpassungen (Schrift & Lesbarkeit)
fig_site.update_layout(
    xaxis_tickangle=-45,   # Schräg stellen, falls die URLs lang sind
    font=dict(size=14),
    height=500,
    xaxis_title="Website URL",
    yaxis_title="Number of Reviews",
    template="plotly_white"
)

# 4. In Streamlit anzeigen
st.plotly_chart(fig_site, use_container_width=True)

# 5. Der Maschinen-Check: Lohnt sich die Spalte für das Modell?
st.info(f"💡 **Insight:** We have {len(site_counts)} unique sites. If one or two sites dominate the whole dataset, we might consider dropping this column as well to avoid bias.")



# 1. Drop the 'company_site' column
if 'company_site' in df_processed.columns:
    df_processed = df_processed.drop(columns=['company_site'])

# 2. Professional explanation in English
st.info("""
    💡 **Feature Selection Update:** 
    The 'company_site' column has been removed due to extreme data imbalance. 
    Similar to 'location', the dominance of a single category (.de) would likely 
    introduce bias and hinder the model's ability to generalize across different sources.
""")

cleaned_cols = ['year', 'month_name', 'weekday', 'season', 'day_period', 'review_text', 'verified', 'review_text_clean', 'review_text_clean_advanced']
dropped_cols = ['location', 'company', 'supplier_response', 'has_response', 'company_site']  # 'has_response' wird hier hinzugefügt, da es ein abgeleitetes Feature ist, das auf 'supplier_response' basiert

# 2. Den HTML-String OHNE Einrückung am Zeilenanfang bauen
html_status = """<style>
.status-table { width: 100%; border-collapse: collapse; font-family: sans-serif; color: #31333F; }
.status-table th, .status-table td { border-bottom: 1px solid #f0f2f6; padding: 12px; text-align: left; font-size: 16px; }
.status-table th { background-color: #f0f2f6; font-weight: bold; }
.strikethrough { text-decoration: line-through; color: #9e9e9e; opacity: 0.7; font-style: italic; }
</style>
<table class="status-table">
<thead><tr><th>Column Name</th><th>Unique Values</th><th>Status</th></tr></thead>
<tbody>"""

# 3. Schleife über alle Spalten
display_cols = list(df_processed.columns) + [c for c in dropped_cols if c not in df_processed.columns]

for col in display_cols:
    is_dropped = col in dropped_cols
    row_class = 'class="strikethrough"' if is_dropped else ''
    u_count = df_processed[col].nunique() if col in df_processed.columns else "-"
    status_icon = "🗑️" if is_dropped else ("✅" if col in cleaned_cols else "❌")
    
    html_status += f'<tr {row_class}><td>{col}</td><td>{u_count}</td><td>{status_icon}</td></tr>'

html_status += "</tbody></table>"

st.markdown(html_status, unsafe_allow_html=True)


st.markdown("---")
st.markdown("<br><br>", unsafe_allow_html=True)




# issue categories analysieren (z.B. wie viele Bewertungen pro Kategorie, welche Kategorien haben die besten/schlechtesten Bewertungen, etc.)

# 1. Alle 69 Kategorien zählen und sortieren
issue_full_list = df_processed['issue_categories'].value_counts(dropna=False).reset_index()
issue_full_list.columns = ['Issue Category Content', 'Occurrences (Count)']

# 2. Anzeige als interaktive Tabelle in Streamlit
st.write("### 📋 Complete List of all 69 Issue Categories")

# Wir nutzen st.dataframe, damit du scrollen und nach Häufigkeit sortieren kannst
st.dataframe(
    issue_full_list, 
    use_container_width=True, 
    hide_index=True
)

# 3. Kleiner Helfer-Text für die Analyse
st.info(f"💡 **Insight:** You are seeing all **{len(issue_full_list)}** unique entries. Many look like technical 'Counter' objects. You can click on the column headers to sort them!")



# 1. Spalte 'issue_categories' aus dem Arbeits-DF löschen
if 'issue_categories' in df_processed.columns:
    df_processed = df_processed.drop(columns=['issue_categories'])

# 2. Professionelle Info-Box (English)
st.info("""
    💡 **Feature Selection Update:** 
    The 'issue_categories' column will been removed. Approximately **2/3 of the data (4,169 rows)** 
    contained no classification (empty Counter objects). The remaining entries were 
    technically corrupted, making the feature unreliable for model training.
""")



cleaned_cols = ['year', 'month_name', 'weekday', 'season', 'day_period', 'review_text', 'verified', 'review_text_clean', 'review_text_clean_advanced']
#aim_cols = ['rating']  # Spalte, die wir vorerst behalten, da sie unser Zielwert ist (auch wenn sie noch nicht bereinigt ist)
dropped_cols = ['location', 'company', 'supplier_response', 'has_response', 'company_site', 'issue_categories']  # 'has_response' wird hier hinzugefügt, da es ein abgeleitetes Feature ist, das auf 'supplier_response' basiert

# 2. Den HTML-String OHNE Einrückung am Zeilenanfang bauen
html_status = """<style>
.status-table { width: 100%; border-collapse: collapse; font-family: sans-serif; color: #31333F; }
.status-table th, .status-table td { border-bottom: 1px solid #f0f2f6; padding: 12px; text-align: left; font-size: 16px; }
.status-table th { background-color: #f0f2f6; font-weight: bold; }
.strikethrough { text-decoration: line-through; color: #9e9e9e; opacity: 0.7; font-style: italic; }
</style>
<table class="status-table">
<thead><tr><th>Column Name</th><th>Unique Values</th><th>Status</th></tr></thead>
<tbody>"""

# 3. Schleife über alle Spalten
display_cols = list(df_processed.columns) + [c for c in dropped_cols if c not in df_processed.columns]

for col in display_cols:
    is_dropped = col in dropped_cols
    row_class = 'class="strikethrough"' if is_dropped else ''
    u_count = df_processed[col].nunique() if col in df_processed.columns else "-"
    status_icon = "🗑️" if col in dropped_cols else ("🎯" if col == "rating" else ("✅" if col in cleaned_cols else "❌"))

    html_status += f'<tr {row_class}><td>{col}</td><td>{u_count}</td><td>{status_icon}</td></tr>'

html_status += "</tbody></table>"

st.markdown(html_status, unsafe_allow_html=True)


st.markdown("---")
st.markdown("<br><br>", unsafe_allow_html=True)





# Darstellen des bereinigten Dataframes mit den neuen Features (Jahr, Monat, Wochentag, Saison, Tageszeit) und den bereinigten Textspalten (review_text_clean, review_text_clean_advanced) in Streamlit.
# 1. Spaltenreihenfolge für die finale Ansicht optimieren
# Wir schieben die Zeit-Features und Verified nach vorne, Text und Rating ans Ende
final_order = [
    'year', 'month_name', 'weekday', 'season', 'day_period', 
    'verified', 'review_text', 'review_text_clean', 'review_text_clean_advanced', 'rating'
]

# Nur die Spalten nehmen, die auch wirklich im DF existieren
df_final_view = df_processed[[col for col in final_order if col in df_processed.columns]]

# 2. Die finale Vorschau (Erste 15 Zeilen)
st.write("### 🏆 Final Processed Dataset (Top 15 Rows)")

# Wir nutzen use_container_width=True für die volle Breite
st.dataframe(df_final_view.head(15), use_container_width=True)


st.markdown("<br><br><br>", unsafe_allow_html=True)


# Den Text für die Box vorbereiten
success_text = f"🏁 **Phase 'Preprocessing' Complete:** Our dataset is now high-octane fuel for Machine Learning! <br>Total: <b>{df_processed.shape[0]}</b> reviews and <b>{df_processed.shape[1]}</b> clean features."

# HTML-Box mit anpassbarer Schriftgröße (hier 20px)
success_html = f"""
<div style="
    background-color: #d4edda; 
    border-radius: 8px; 
    padding: 20px; 
    border: 1px solid #c3e6cb; 
    color: #155724; 
    font-size: 20px; 
    font-family: sans-serif;
    margin-top: 20px;
    margin-bottom: 20px;
    line-height: 1.5;">
    {success_text}
</div>
"""

# In Streamlit anzeigen
st.markdown(success_html, unsafe_allow_html=True)

