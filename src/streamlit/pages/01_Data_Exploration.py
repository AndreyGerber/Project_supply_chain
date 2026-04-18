import streamlit as st
import pandas as pd
import plotly.express as px 
from pathlib import Path

# 1. Configuration
st.set_page_config(page_title="Auto parts store Review Dashboard", layout="wide")

@st.cache_data
def load_data():
    file_path = Path("src/data/clean/reviews_clean.csv")
    if not file_path.exists():
        st.error(f"Data file not found at: {file_path.absolute()}")
        return pd.DataFrame()

    df = pd.read_csv(file_path)
    # Nur Datum konvertieren, da das technisch notwendig für Plotly ist
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    return df

# --- INITIALISIERUNG ---
df_raw = load_data()

# WICHTIG: Speichere die ECHTEN ROHDATEN für die anderen Seiten
if not df_raw.empty:
    st.session_state['raw_data'] = df_raw.copy()

# Erstelle eine Arbeitskopie für die aktuelle Seite
df = df_raw.copy()

# BEREINIGUNG NUR AUSFÜHREN, WENN DIE SPALTE NOCH EIN TEXT IST
if 'rating_svg' in df.columns:
    # Wir stellen sicher, dass es Text ist (.astype(str)), bevor wir .str nutzen
    df['rating'] = df['rating_svg'].astype(str).str.extract('(\d+)').fillna(0).astype(int)

# Spalten erst löschen, nachdem die Extraktion fertig ist
columns_to_drop = ['rating_numeric', 'rating_svg']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Ab hier nutzt dein restlicher Code wieder "df" für die Grafiken





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
    st.title("📊 Phase 1: Data Exploration")

    # Fügt eine Leerzeile ein
    st.markdown("<br>", unsafe_allow_html=True)
    # Für einen wirklich großen Abstand zwischen der Tabelle und den nächsten Abschnitten
    st.markdown("<div style='margin-bottom: 50px;'></div>", unsafe_allow_html=True)

    st.markdown("""
        <div style="
            text-align: left; 
            padding: 15px; 
            background-color: #e8f4f8; 
            border-radius: 10px; 
            color: #004085;
            font-size: 1.1em;
            border: 1px solid #b8daff;">
            🚀 The objective of this part is to extract meaningful information from customer comments. The main areas of work include:
        </div>

            We analyze the raw dataset of the **Auto Parts Stores**. 
            We will try to understand the distribution of star ratings and get a first look at the customer comments.
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
            To collect customer feedback data, an automated web scraping pipeline was developed.
            Due to dynamically loaded content, a browser-based approach using Selenium was implemented to ensure reliable extraction of all relevant elements, including hidden or asynchronously loaded data.<br>
            <br>
        <style>
            /* Schwarze Punkte */
            li::marker {
                color: black;
                font-size: 1.2em;
            }
            
            /* NUR die Liste einrücken */
            .eingerueckte-liste {
                margin-left: 20px; /* Hier schiebst du nur die Punkte nach rechts */
                margin-top: 10px;
                line-height: 1.4; /* Optional: Erhöht den Zeilenabstand für bessere Lesbarkeit */
            }
        </style>

        <div style="font-size: 18px; line-height: 1.6;">
            The scraper iterates across multiple companies and pages, extracting the following attributes for each review:
            <ul class="eingerueckte-liste">
                <li><b style="color: #1E88E5;">review_text:</b> customer comment</li>
                <li><b style="color: #1E88E5;">rating_svg:</b> star rating</li>
                <li><b style="color: #1E88E5;">date:</b> timestamp of the review</li>
                <li><b style="color: #1E88E5;">location:</b> customer country</li>
                <li><b style="color: #1E88E5;">supplier_response:</b> company reply</li>
                <li><b style="color: #1E88E5;">verified:</b> review verification status</li>
                <li><b style="color: #1E88E5;">company:</b> retailer identifier</li>
            </ul>
        </div>
            <br>
            The initial analytics are presented below — enjoy exploring!
        </div>""", unsafe_allow_html=True)
    st.markdown("---")


    # --- POSITION 1: RAW DATA PREVIEW ---
    st.subheader("📄 Raw Data Preview")
    st.info("Direct preview of the filtered dataset:")

    # Spalten definieren, die NICHT angezeigt werden sollen
    cols_to_exclude = ["review_text_clean_advanced", "review_text_clean", "issue_categories"]
    
    # Wir zeigen nur die Spalten an, die nicht in der Ausschlussliste sind
    # .drop(columns=...) erzeugt eine Kopie ohne die genannten Spalten
    st.dataframe(
        df_filtered.drop(columns=cols_to_exclude, errors='ignore').head(76), 
        use_container_width=True,
        height=450 
    )
    
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
            ui_height = 650 

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
                title="Comments per Company"
                )
            
                # Design-Anpassung für saubere Kanten
                fig_comp.update_layout(
                margin=dict(l=0, r=0, t=40, b=0), # Ränder minimieren
                yaxis={'categoryorder':'total ascending'} # Größte Balken oben
                )
            
                st.plotly_chart(fig_comp, use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)



    # Rating vs Verified (table and chart)
    with st.container(border=True):
        st.markdown("#### 📊 Rating vs Verified")

        # Ein Violin-Plot mit 'points="all"' zeigt die Dichte der Bewertungen
        fig_ver = px.violin(
            df,
            x="verified",
            y="rating",
            color="verified",      # Unterschiedliche Farben für 0 und 1
            box=True,              # Zeichnet eine kleine Box in die Mitte
            points="all",          # Zeigt alle Datenpunkte mit Jitter (Streuung)
            color_discrete_map={0: "#EF553B", 1: "#00CC96"} # Rot für Unverified, Grün für Verified
        )

        # 1. Daten gruppieren und zählen (Wichtig: Spaltennamen 'rating' prüfen)
        counts = df.groupby(['verified', 'rating']).size().reset_index(name='count')

        # 2. Zahlen als Text-Labels hinzufügen
        for i, row in counts.iterrows():
            fig_ver.add_annotation(
                x=row['verified'],
                y=row['rating'],
                text=f"n={row['count']}",
                showarrow=False,
                yshift=10, 
                font=dict(size=12, color="black", family="Arial")
            )

        # 3. Layout verschönern (Beschriftung der Achsen)
        fig_ver.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=[0, 1],
                ticktext=['Non-Verified (0)', 'Verified (1)']
            )
        )


        st.plotly_chart(fig_ver, use_container_width=True)

        # Textbeschreibung
        st.markdown("""
        The inclusion of a **'verified'** indicator allows us to distinguish  
        between authenticated and non-authenticated customer feedback,  
        reducing potential bias and increasing the reliability of the analysis.
        """)


    # Abstand unten
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
        st.markdown("#### 📊 Number of comments by Year")
        
        # Daten vorbereiten (Jahre extrahieren und zählen)
        df_filtered['Year'] = df_filtered['date'].dt.year.astype(str)
        yearly_counts = df_filtered['Year'].value_counts().sort_index().reset_index()
        yearly_counts.columns = ['Year', 'Number of comments']

        # 1. Plotly Bar Chart - Jedes Jahr als eigene Gruppe, aber gleiche Farbe
        fig_years = px.bar(
            yearly_counts, 
            x='Year', 
            y='Number of comments',
            text='Number of comments',
            color='Year', # Wichtig: Damit bekommt jedes Jahr einen eigenen Legenden-Eintrag
            # Hier erzwingen wir, dass JEDES Jahr die gleiche Farbe bekommt:
            color_discrete_sequence=['#1E88E5'] * len(yearly_counts), 
            height=600
        )

        # 2. Layout-Feineinstellungen
        fig_years.update_layout(
            xaxis_type='category', 
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=True, 
            
            legend=dict(
                title="Select Year:", # Geänderter Titel für das Publikum
                orientation="v",
                yanchor="top", y=1, 
                xanchor="left", x=1.02
            ),

            # --- SCHRIFTGRÖSSEN ---
            font=dict(size=14),
            xaxis=dict(title_font=dict(size=20), tickfont=dict(size=14)),
            yaxis=dict(title_font=dict(size=20), tickfont=dict(size=16), showgrid=True, gridcolor='LightGray'),
            margin=dict(r=150)
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
            labels={'rating': 'Star Rating', 'count': 'Number of Comments'},
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
                title_font=dict(size=24), # Größe der "Number of Reviews" Beschriftung
                tickfont=dict(size=14),   # Größe der Zahlen an der Y-Achse
                showgrid=True, 
                gridcolor='LightGray',
                title = "Count of Star Ratings"
            )
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("📈 Average rating per company over the years")

        # --- 1. ZENTRALE EINSTELLUNGEN: SCHRIFTGRÖSSEN & LAYOUT ---
        font_size_axis_title = 20  # Größe der Achsen-Beschriftungen (X & Y)
        font_size_ticks = 14       # Größe der Zahlen an den Achsen
        font_size_legend = 14      # Größe der Firmennamen in der Legende
        font_size_title = 22       # Größe des Diagramm-Titels
        chart_height = 600

        # --- 2. DATEN VORBEREITEN ---
        df_time = df_filtered.copy()
        df_time['year'] = df_time['date'].dt.year

        # Zeitspanne: Erster Kommentar im Datensatz bis HEUTE
        min_year = int(df_time['year'].min())
        max_year = pd.Timestamp.now().year
        all_years = list(range(min_year, max_year + 1))

        # Filter-Slider
        min_reviews = st.slider(
            "min number of comments per year:", 
            min_value=5, max_value=15, value=7
        )

        # Gruppierung
        df_grouped = df_time.groupby(['year', 'company']).agg(
            avg_rating=('rating', 'mean'),
            review_count=('rating', 'count')
        ).reset_index()

        # Filter anwenden
        df_trend = df_grouped[df_grouped['review_count'] >= min_reviews].copy()

        # --- 3. LÜCKEN FÜLLEN (Stellt sicher, dass jedes Jahr auf der X-Achse existiert) ---
        companies = df_trend['company'].unique()
        if len(companies) > 0:
            mux = pd.MultiIndex.from_product([all_years, companies], names=['year', 'company'])
            df_trend = df_trend.set_index(['year', 'company']).reindex(mux).reset_index()

            # --- 4. DIAGRAMM ERSTELLEN ---
            fig_trend = px.line(
                df_trend,
                x="year",
                y="avg_rating",
                color="company",
                markers=True,
                title=f"Trends in Customer Satisfaction ({min_year} - {max_year})",
                labels={'year': 'Year', 'avg_rating': 'Ø Stars', 'company': 'Company'},
                hover_data={'review_count': True},
                height=chart_height,
                color_discrete_sequence=px.colors.qualitative.Safe # Gut unterscheidbare Farben
            )

            # --- 5. FINETUNING DER OPTIK & SCHRIFTGRÖSSEN ---
            fig_trend.update_layout(
                title_font=dict(size=font_size_title),
                xaxis=dict(
                    type='linear',
                    tickmode='linear',
                    dtick=1,
                    range=[min_year - 0.1, max_year + 0.1], # Achse fest bis heute
                    title_font=dict(size=font_size_axis_title),
                    tickfont=dict(size=font_size_ticks),
                    showgrid=True,
                    gridcolor='rgba(200, 200, 200, 0.3)'
                ),
                yaxis=dict(
                    range=[0.8, 5.2], # Skala von 1 bis 5 (mit etwas Puffer)
                    dtick=1,
                    title_font=dict(size=font_size_axis_title),
                    tickfont=dict(size=font_size_ticks),
                    title="Rating (Ø Stars)",
                    showgrid=True
                ),
                legend=dict(
                    font=dict(size=font_size_legend),
                    orientation="v",         # Vertikal
                    yanchor="top", y=1,      # Oben ausrichten
                    xanchor="left", x=1.02,  # Rechts neben dem Chart positionieren
                    title_font=dict(size=font_size_legend + 2)
                ),
                margin=dict(l=60, r=150, t=80, b=60), # Platz rechts für Legende reserviert
                hovermode="x unified",
                plot_bgcolor='white'
            )

            # WICHTIG: Linien nicht verbinden, wenn ein Jahr fehlt (kein "Drop to Zero")
            fig_trend.update_traces(connectgaps=False, line=dict(width=3))

            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Erhöhe den Filter oder wähle mehr Ratings aus, um Daten zu sehen.")

    with tab3:
        st.header("📍 Geographic & Support Performance")

        # --- Parameter für die Diagrammhöhe ---
        chart_height = 500

        # --- Aufteilung 70% zu 30% ---
        col_a, col_b = st.columns([7, 3]) 

        with col_a:
            # 1. Daten für Regionen (Top 9 + Others) vorbereiten
            loc_counts = df_filtered['location'].value_counts()
            top_9 = loc_counts.head(9)
            
            if len(loc_counts) > 9:
                others_count = loc_counts.iloc[9:].sum()
                others_series = pd.Series([others_count], index=['Others'])
                final_loc_data = pd.concat([top_9, others_series])
            else:
                final_loc_data = top_9

            # Daten für Plotly in DataFrame umwandeln
            plot_df = final_loc_data.reset_index()
            plot_df.columns = ['Region', 'Count']

            # 2. Vertikales Balkendiagramm
            fig_loc = px.bar(
                plot_df, 
                x='Region',  # Länder auf die X-Achse
                y='Count',   # Anzahl auf die Y-Achse
                title="Top 9 Regions & Others", 
                text='Count', # Zahlen über den Balken
                color='Region', 
                color_discrete_sequence=px.colors.qualitative.Pastel,
                height=chart_height
            )
        
            # Layout-Anpassungen für vertikale Optik
            fig_loc.update_layout(
                showlegend=False, # Legende aus, da X-Achse beschriftet ist
                xaxis_title="Region",
                yaxis_title="Number of Reviews",
                xaxis={'categoryorder':'total descending'}, # Höchster Balken links
                margin=dict(t=50, b=50, l=20, r=20)
            )
            
            # Zahlen über den Balken positionieren
            fig_loc.update_traces(textposition='outside')
            
            st.plotly_chart(fig_loc, use_container_width=True)

        with col_b:
            # 3. Response Status vorbereiten
            df_filtered['has_response'] = df_filtered['supplier_response'].notna()
            resp_counts = df_filtered['has_response'].value_counts().rename({True: 'Responded', False: 'Pending'})
            
            # 4. Balkendiagramm
            fig_resp = px.bar(
                x=resp_counts.index, 
                y=resp_counts.values, 
                title="Response Status", 
                color=resp_counts.index,
                # Farben exakt wie im Bild
                color_discrete_map={'Responded': '#2E6AD1', 'Pending': '#89C6FF'},
                height=chart_height # <--- Dynamische Höhe
            )
        
            fig_resp.update_layout(
                showlegend=False,
                xaxis_title=None,
                yaxis_title="Anzahl",
                margin=dict(t=50, b=50, l=20, r=20)
            )
        
            # Zahlen über den Balken anzeigen
            fig_resp.update_traces(texttemplate='%{y}', textposition='outside')
            
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
    
     # 2. Zentrierter Ausblick-Satz mit Verlinkung
    st.markdown("""
        <div style="text-align: left; padding: 20px; background-color: #e8f4f8; border-radius: 10px; color: #004085; font-size: 1.1em; border: 1px solid #b8daff; line-height: 1.8;">
            
        <span style="font-size: 1.4em; font-weight: bold; display: block; margin-bottom: 12px;">
            Next Steps will be:
        </span>

        <div style="margin-bottom: 8px;">
            <a href="/Preprocessing" target="_self" style="color: black; font-weight: bold; text-decoration: underline;">Preprocessing</a> 
            → preparing the data for machine learning models
        </div>
            
        <div style="margin-bottom: 8px;">
            <a href="/Modeling" target="_self" style="color: black; font-weight: bold; text-decoration: underline;">Modeling</a> 
            → analyzing the data with various machine learning techniques and choosing the best model for our use case.
        </div>
            
        <div style="margin-bottom: 8px;">
            <a href="/Live_Demo" target="_self" style="color: black; font-weight: bold; text-decoration: underline;">Play with our model</a> 
            → testing the model with new comments and evaluating the results.
        </div>
            
        </div>""", unsafe_allow_html=True)


# Diese Zeilen stehen GANZ LINKS (ohne Einrückung) am Ende der Datei
else:
    st.warning("Data could not be loaded. Please check the source file.")