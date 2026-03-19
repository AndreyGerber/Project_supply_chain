import streamlit as st
import pandas as pd
import plotly.express as px 
from pathlib import Path

# 1. Configuration
st.set_page_config(page_title="Copy of Auto parts store Review Dashboard", layout="wide")

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
    st.title("📊 Copy of auto parts store Customer Insights Dashboard")
    st.markdown("This dashboard provides a comprehensive analysis of customer feedback and supplier performance.")
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
            
            # Darstellung als Tabelle oder kleiner Bar Chart für bessere Übersicht
            c1, c2 = st.columns([1, 2]) # Tabelle links, Mini-Chart rechts
            with c1:
                # Wir zeigen 10 Zeilen an
                st.dataframe(company_counts.head(10), use_container_width=True, hide_index=True)
            with c2:
                # 'height=380' entspricht in etwa der Höhe von 10 Tabellenzeilen + Header
                fig_comp = px.bar(
                company_counts.head(10), # Nur die Top 10 zeigen, damit es zur Tabelle passt
                x='Review Count', 
                y='Company Name', 
                orientation='h', 
                height=380, # <--- Dieser Wert ist entscheidend für die Angleichung
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
        # --- LINIENDIAGRAMM: DURCHSCHNITTS-RATING PRO JAHR ---
        st.markdown("#### 📈 Average Rating Trends by Company")

        # 1. Daten vorbereiten (Datum konvertieren & Jahr extrahieren)
        df_filtered['date'] = pd.to_datetime(df_filtered['date'])
        df_filtered['Year'] = df_filtered['date'].dt.year

        # 2. Gruppieren: Durchschnittliches Rating pro Jahr und Firma
        df_trend = df_filtered.groupby(['Year', 'company'])['rating'].mean().reset_index()

        # 3. Plotly Liniendiagramm erstellen
        fig_trend = px.line(
            df_trend, 
            x='Year', 
            y='rating', 
            color='company',       # Erzeugt eine Linie pro Firma (und die Legende rechts)
            markers=True,          # Zeigt Punkte auf der Linie an
            labels={'rating': 'Average Rating', 'Year': 'Year'},
            height=500             # Gewünschte Diagrammhöhe
        )

        # 4. Design-Anpassungen (Achsen, Legende, Schrift)
        fig_trend.update_layout(
            xaxis_type='category', 
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(size=14),
            xaxis=dict(
                title_font=dict(size=16),
                tickfont=dict(size=13),
                showgrid=True, 
                gridcolor='LightGray'
            ),
            yaxis=dict(
                title_font=dict(size=16),
                tickfont=dict(size=13),
                range=[1, 5.1],     # Fixiert die Skala auf 1 bis 5 Sterne
                showgrid=True, 
                gridcolor='LightGray'
            ),
            legend=dict(
                title="Companies (click to toggle):",
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            ),
            margin=dict(r=150)     # Platz für die Legende rechts
        )

        st.plotly_chart(fig_trend, use_container_width=True)

    with tab3:
        st.header("📍 Geographic & Support Performance")
        col_a, col_b = st.columns(2)
        with col_a:
            top_loc = df_filtered['location'].value_counts().head(8)
            fig_loc = px.pie(values=top_loc.values, names=top_loc.index, title="Top Regions", hole=0.4)
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
        <div style="
            text-align: center; 
            padding: 15px; 
            background-color: #e8f4f8; 
            border-radius: 10px; 
            color: #004085;
            font-size: 1.1em;
            border: 1px solid #b8daff;">
            🚀 More detailed analysis will appear soon—with better visuals, emojis, and machine learning algorithms.
        </div>
    """, unsafe_allow_html=True)

# Diese Zeilen stehen GANZ LINKS (ohne Einrückung) am Ende der Datei
else:
    st.warning("Data could not be loaded. Please check the source file.")