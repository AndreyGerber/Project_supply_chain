import streamlit as st
import pandas as pd
import plotly.express as px 
from pathlib import Path

# 1. Configuration
st.set_page_config(page_title="Auto parts store Review Dashboard", layout="wide")

# 2. Data Loading Function
@st.cache_data
def load_data():
    file_path = Path("src/data/clean/reviews_clean.csv")
    if not file_path.exists():
        st.error(f"Data file not found at: {file_path.absolute()}")
        return pd.DataFrame()

    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    
    if 'rating_svg' in df.columns:
        df['rating'] = df['rating_svg'].str.extract('(\d+)').astype(float).fillna(0).astype(int)
    
    columns_to_drop = ['rating_numeric', 'rating_svg']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    return df

# Initialize Data
df = load_data()





# Main Application Logic
if not df.empty:
    # 3. Sidebar Filtering
    st.sidebar.header("Filter Options")
    selected_rating = st.sidebar.multiselect(
        "Select Rating", 
        options=sorted(df['rating'].unique()), 
        default=sorted(df['rating'].unique())
    )
    df_filtered = df[df['rating'].isin(selected_rating)]

    # 4. Main Header
    st.title("📊 Auto parts store Customer Insights Dashboard")
    st.markdown("""
        <div style="
            text-align: left; 
            padding: 15px; 
            background-color: #e8f4f8; 
            border-radius: 10px; 
            color: #004085;
            font-size: 1.1em;
            border: 1px solid #b8daff;">
            🚀 The objective of this project is to extract meaningful information from customer comments. The main areas of work include:
        </div>

            1. Predicting customer satisfaction: A regression problem focused on predicting the number of stars.
            2. Identifying important entities in a message: Such as location, company name, etc.
            3. Extracting key topics from comments: For example delivery issues, defective items, etc., using an unsupervised approach.
            4.Analyzing supplier responses: Extracting relevant words and patterns from responses in order 
            to predict them based solely on the original comment.
        <div style=" 
            text-align: left; 
            padding: 15px; 
            background-color: #e8f4f8; 
            border-radius: 10px; 
            color: #004085;
            font-size: 1.1em;
            border: 1px solid #b8daff;">
            Our focus lies on German companies in the “Auto Parts Store” category on Trustpilot. 
            The dataset was scraped from Trustpilot.<br><br>
            The initial analytics are presented below—enjoy exploring!
        </div>""", unsafe_allow_html=True)
    st.markdown("---")

    # --- POSITION 1: RAW DATA PREVIEW ---
    st.subheader("📄 Raw Data Preview")
    st.info("Direct preview of the filtered dataset:")
    st.dataframe(df_filtered.head(15), use_container_width=True)
    
    # Fügt eine Leerzeile ein
    st.markdown("<br>", unsafe_allow_html=True)
    # Für einen wirklich großen Abstand zwischen der Tabelle und den nächsten Abschnitten
    st.markdown("<div style='margin-bottom: 50px;'></div>", unsafe_allow_html=True)


    # COMPANY VALUE COUNTS (table and chart) ---
    with st.container(border=True):
        st.markdown("#### 🏢 Company Distribution")
        if 'company' in df_filtered.columns:
            company_counts = df_filtered['company'].value_counts().reset_index()
            company_counts.columns = ['Company Name', 'Review Count']
            
            # 500px bieten genug Platz für 12 Zeilen + Header + Padding
            ui_height = 550 

            # Darstellung als Tabelle oder kleiner Bar Chart für bessere Übersicht
            c1, c2 = st.columns([1, 2]) # Tabelle links, Mini-Chart rechts
            with c1:
                # Wir zeigen alle Zeilen an
                st.dataframe(
                    company_counts, 
                    use_container_width=True, 
                    hide_index=True,
                    height=ui_height ) # <--- Das hat im Screenshot gefehlt     
            with c2:
                # 'height=380' entspricht in etwa der Höhe von 10 Tabellenzeilen + Header
                fig_comp = px.bar(company_counts, # alle Firmen anzeigen
                x='Review Count', 
                y='Company Name', 
                orientation='h', 
                height=ui_height, # <--- Dieser Wert ist entscheidend für die Angleichung
                title="Reviews per Company"
                )
            
                # Design-Anpassung für saubere Kanten
                fig_comp.update_layout(
                margin=dict(l=0, r=0, t=40, b=0), # Ränder minimieren
                yaxis={'categoryorder':'total ascending'} # Größte Balken oben
                )
            
                st.plotly_chart(fig_comp, use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)



       # --- 📅 Analysis Period & Timeline ---
    st.markdown("#### 📅 Analysis Period")
    if not df_filtered.empty and 'date' in df_filtered.columns:
        first_date = df_filtered['date'].min()
        last_date = df_filtered['date'].max()
        
        st.markdown(
            f"""
            <div style="
                background-color: #d4edda; 
                color: #155724; 
                padding: 15px; 
                border-radius: 5px; 
                font-size: 22px; 
                border: 1px solid #c3e6cb;">
                ✅ This dataset covers reviews from <b>{first_date.strftime('%d.%m.%Y')}</b> 
                to <b>{last_date.strftime('%d.%m.%Y')}</b>.
            </div>
            """, 
            unsafe_allow_html=True
          )

        # 1. Die Zeitachse (als Linie in Form eines kleinen Diagramms)
        timeline_df = pd.DataFrame({'date': [first_date, last_date], 'label': ['first comment', 'last comment'], 'y': [0, 0]})
        fig_timeline = px.line(timeline_df, x='date', y='y', markers=True, text='label')
        fig_timeline.update_traces(line_color='#2E7D32', line_width=4, marker=dict(size=12, 
        symbol='diamond'), textposition='top center', textfont=dict(size=16, weight='bold'))
        fig_timeline.update_layout(height=120, margin=dict(l=20, r=20, t=30, b=20), xaxis=dict(showgrid=False, title=""),
                                   yaxis=dict(showgrid=False, showticklabels=False, title=""), plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_timeline, use_container_width=True, config={'displayModeBar': False})


        # --- ABSTAND EINFÜGEN ---
        st.write("##") # Erzeugt einen vertikalen Abstand (ca. 30-40px)

        # -KPIs (Total Reviews, Average Rating, Supplier Response Rate) ---
        with st.container(border=True):
            st.markdown("""
            <style>
            [data-testid="stMetric"] {display: flex; flex-direction: column; align-items: center; text-align: center; }
            [data-testid="stMetricLabel"] >div {font-size: 22px !important; font-weight: bold !important; justify-content: center !important; text-align: center !important; }
            [data-testid="stMetricValue"] >div {font-size: 25px !important; font-weight: bold !important; justify-content: center !important; text-align: center !important; }
            </style>    """, unsafe_allow_html=True)

            avg_rating = df_filtered['rating'].mean()
            response_rate = df_filtered['supplier_response'].notna().mean() * 100

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Reviews", len(df_filtered))
            col2.metric("Average Rating", f"{avg_rating:.2f} / 5.0")
            col3.metric("Supplier Response Rate", f"{response_rate:.1f}%")
            #st.markdown("---")
        st.markdown("<br>", unsafe_allow_html=True)

        # NOCHMAL ABSTAND VOR DER NÄCHSTEN GRAFIK ---
        st.markdown("<br><br>", unsafe_allow_html=True) # Erzeugt zwei Zeilenumbrüche

        # --- HIER KOMMT DAS NEUE BALKENDIAGRAMM REIN, Kommentare über das Jahr ---
    with st.container(border=True): 
        st.markdown("#### 📊 Review Volume by Year")
        
        # Daten vorbereiten (Jahre extrahieren und zählen)
        df_filtered['Year'] = df_filtered['date'].dt.year.astype(str)
        yearly_counts = df_filtered['Year'].value_counts().sort_index().reset_index()
        yearly_counts.columns = ['Year', 'Number of Reviews']

        # Plotly Bar Chart
        fig_years = px.bar(
            yearly_counts, 
            x='Year', 
            y='Number of Reviews',
            text='Number of Reviews',
            color='Year', # Erzeugt die Legende
            color_discrete_sequence=px.colors.qualitative.Plotly,
            height=600  # <--- HIER: Gesamthöhe des Diagramms einstellen
        )

        # Layout-Feineinstellungen
        fig_years.update_layout(
            xaxis_type='category', 
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            
            # --- LEGENDE RECHTS ---
            legend=dict(title="(click to select):",orientation="v",    # Vertikal
                yanchor="top", y=1, xanchor="left", 
                x=1.02 # Platziert die Legende rechts außerhalb des Diagramms
            ),

            # --- SCHRIFTGRÖSSEN ---
            font=dict(size=14),     # Allgemeine Schriftgröße (optional)
            xaxis=dict(
                title_font=dict(size=20), # Größe der "Year" Beschriftung
                tickfont=dict(size=14)    # Größe der Jahreszahlen (2012, 2014...)
            ),
            yaxis=dict(
                title_font=dict(size=20), # Größe der "Number of Reviews" Beschriftung
                tickfont=dict(size=16),   # Größe der Zahlen an der Y-Achse
                showgrid=True, 
                gridcolor='LightGray'
            ),
            margin=dict(r=150) # Platz rechts lassen, damit die Legende nicht abgeschnitten wird
        )

        fig_years.update_traces(textposition='outside')

        st.plotly_chart(fig_years, use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)
        # --- ENDE DES ABSCHNITTS Review Volume by Year---

    st.markdown("---")
    





    # 6. Analysis Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Performance Trends", "💬 Feedback Analysis", "📍 Operations & Support"])

    with tab1:
        st.subheader("Customer Satisfaction Distribution")
        color_map = {1: "#2E7D32", 2: "#311B92", 3: "#FBC02D", 4: "#81D4FA", 5: "#C62828"}
        fig = px.histogram(
            df_filtered,
            x="rating",
            color="rating",
            title="Frequency of Ratings",
            labels={'rating': 'Star Rating', 'count': 'Number of Reviews'},
            nbins=5,
            color_discrete_map=color_map,
            height=600  # <--- HIER: Gesamthöhe des Diagramms einstellen
        )
        fig.update_layout(
         # --- SCHRIFTGRÖSSEN ---
            font=dict(size=14),     # Allgemeine Schriftgröße (optional)
            xaxis=dict(
                title_font=dict(size=20), # Größe der "Year" Beschriftung
                tickfont=dict(size=14)    # Größe der Jahreszahlen (2012, 2014...)
            ),
            yaxis=dict(
                title_font=dict(size=20), # Größe der "Number of Reviews" Beschriftung
                tickfont=dict(size=14),   # Größe der Zahlen an der Y-Achse
                showgrid=True, 
                gridcolor='LightGray'
            )
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("📈 Durchschnittsbewertung pro Unternehmen über die Jahre")

        # 1. Vorbereitung: Jahr extrahieren
        df_time = df_filtered.copy()
        df_time['year'] = df_time['date'].dt.year

        # 2. Filter-Parameter in der UI (Slider)
        min_reviews = st.slider(
            "Mindestanzahl an Kommentaren pro Jahr & Unternehmen:", 
            min_value=1, max_value=20, value=5
        )

        # 3. Gruppierung: Durchschnitts-Rating und Anzahl pro Jahr/Firma
        df_grouped = df_time.groupby(['year', 'company']).agg(
            avg_rating=('rating', 'mean'),
            review_count=('rating', 'count')
        ).reset_index()

        # 4. Filter anwenden
        df_trend = df_grouped[df_grouped['review_count'] >= min_reviews]

        if not df_trend.empty:
            # 5. Liniendiagramm erstellen
            fig_trend = px.line(
                df_trend,
                x="year",
                y="avg_rating",
                color="company",
                markers=True,
                title=f"Trends (Unternehmen mit mind. {min_reviews} Reviews/Jahr)",
                labels={'year': 'Jahr', 'avg_rating': 'Ø Bewertung', 'company': 'Unternehmen'},
                hover_data={'review_count': True} # Zeigt Anzahl der Reviews im Tooltip
            )

            # Design-Anpassungen
            fig_trend.update_layout(
                yaxis=dict(range=[1, 5.1], dtick=1), # Y-Achse von 1 bis 5 fixieren
                xaxis=dict(dtick=1),                 # Nur ganze Jahre anzeigen
                height=600,
                hovermode="x unified"
            )

            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.warning("Keine Daten gefunden, die den Filterkriterien entsprechen. Versuche, den Slider zu verringern.")

    with tab3:
        st.header("📍 Geographic & Support Performance")
        col_a, col_b = st.columns(2)
        with col_a:
            # 1. Alle Standorte zählen
            loc_counts = df_filtered['location'].value_counts()
            
            # 2. Die Top 9 extrahieren
            top_9 = loc_counts.head(9)
            
            # 3. Den Rest berechnen und als "Others" zusammenfassen
            others_count = loc_counts.iloc[9:].sum()
            
            # 4. "Others" nur hinzufügen, wenn es wirklich restliche Daten gibt
            if others_count > 0:
                others_series = pd.Series({'Others': others_count})
                final_loc_data = pd.concat([top_9, others_series])
            else:
                final_loc_data = top_9

            # 5. Diagramm erstellen
            fig_loc = px.pie(
                values=final_loc_data.values, 
                names=final_loc_data.index, 
                title="Top 9 Regions & Others", 
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel # Schöne Farben für viele Segmente
            )
            
            # 6. Beschriftung optimieren (Prozent und Name anzeigen)
            fig_loc.update_traces(textinfo='percent+label')
            
            st.plotly_chart(fig_loc, use_container_width=True)
        with col_b:
            df_filtered['has_response'] = df_filtered['supplier_response'].notna()
            resp_counts = df_filtered['has_response'].value_counts().rename({True: 'Responded', False: 'Pending'})
            fig_resp = px.bar(x=resp_counts.index, y=resp_counts.values, title="Response Status", color=resp_counts.index)
            st.plotly_chart(fig_resp, use_container_width=True)

    # 7. Personalized Footer
    st.markdown("---")
    
    # 1. Großer Dankeschön-Text (Zentriert & Doppelte Größe)
    st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <span style="font-weight: bold; color: #ff4b4b; font-size: 2.2em;">
                Thank you for exploring the Autodoc Review Dashboard!
            </span>
        </div>
    """, unsafe_allow_html=True)
    
    # 2. Zentrierter Ausblick-Satz
    st.markdown("""
    <div style="text-align: left; padding: 20px; background-color: #e8f4f8; border-radius: 10px; color: #004085; font-size: 1.1em; border: 1px solid #b8daff; line-height: 1.8;">
        
    <span style="font-size: 1.4em; font-weight: bold; display: block; margin-bottom: 12px;">
        Next Steps will be:
    </span>

    <div style="margin-bottom: 8px;">
        <strong style="color: black;">Machine Learning (Regression)</strong> → Predicting the number of stars
    </div>
        
    <div style="margin-bottom: 8px;">
         <strong style="color: black;">Named Entity Recognition (NER) / Information Extraction</strong> → Identifying important entities
    </div>
        
    <div style="margin-bottom: 8px;">
        <strong style="color: black;">Unsupervised Topic Modeling</strong> → Extracting common issues from comments
    </div>
        
     <div>
        <strong style="color: black;">Information Extraction / Text Matching</strong> → Generating automated supplier responses
    </div>
    </div>""", unsafe_allow_html=True)
# Diese Zeilen stehen GANZ LINKS (ohne Einrückung) am Ende der Datei
else:
    st.warning("Data could not be loaded. Please check the source file.")