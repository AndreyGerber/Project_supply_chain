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
    
        # --- NEU: COMPANY VALUE COUNTS (Direkt unter der Tabelle) ---
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
     
       # --- 📅 Analysis Period & Timeline ---
    st.markdown("#### 📅 Analysis Period")
    if not df_filtered.empty and 'date' in df_filtered.columns:
        first_date = df_filtered['date'].min()
        last_date = df_filtered['date'].max()
        
        st.success(f"This dataset covers reviews from **{first_date.strftime('%d.%m.%Y')}** to **{last_date.strftime('%d.%m.%Y')}**.")

        # 1. Die Zeitachse (Linie)
        timeline_df = pd.DataFrame({'date': [first_date, last_date], 'label': ['first comment', 'last comment'], 'y': [0, 0]})
        fig_timeline = px.line(timeline_df, x='date', y='y', markers=True, text='label')
        fig_timeline.update_traces(line_color='#2E7D32', marker=dict(size=12, symbol='diamond'), textposition='top center')
        fig_timeline.update_layout(height=120, margin=dict(l=20, r=20, t=30, b=20), xaxis=dict(showgrid=False, title=""),
                                   yaxis=dict(showgrid=False, showticklabels=False, title=""), plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_timeline, use_container_width=True, config={'displayModeBar': False})

        # --- HIER KOMMT DAS NEUE BALKENDIAGRAMM REIN ---
        st.markdown("#### 📊 Review Volume by Year")
        
        # Daten vorbereiten (Jahre extrahieren und zählen)
        df_filtered['Year'] = df_filtered['date'].dt.year
        yearly_counts = df_filtered['Year'].value_counts().sort_index().reset_index()
        yearly_counts.columns = ['Year', 'Number of Reviews']

        # Plotly Bar Chart erstellen
        fig_years = px.bar(
            yearly_counts, 
            x='Year', 
            y='Number of Reviews',
            text='Number of Reviews',
            color='Number of Reviews',
            color_continuous_scale='Greens'
        )

        fig_years.update_layout(
            xaxis_type='category', 
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            coloraxis_showscale=False
        )

        st.plotly_chart(fig_years, use_container_width=True)
        # --- ENDE DES NEUEN ABSCHNITTS ---

    st.markdown("---")
    






    # --- POSITION 2: KPIs (Jetzt UNTER der Preview) ---
    avg_rating = df_filtered['rating'].mean()
    response_rate = df_filtered['supplier_response'].notna().mean() * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Reviews", len(df_filtered))
    col2.metric("Average Rating", f"{avg_rating:.2f} / 5.0")
    col3.metric("Supplier Response Rate", f"{response_rate:.1f}%")
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
            color_discrete_map=color_map
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
        st.subheader("💬 Sentiment & Keyword Discovery")
        st.info("Search through individual comments or use the quick-filters below to find specific topics.")
        
        # 1. Such-Bereich
        st.write("**Quick-Filters (Common Topics):**")
        predefined_keywords = ["All", "shipping", "quality", "price", "support", "delivery", "service"]
        
        selected_quick_filter = st.selectbox("Choose a topic:", predefined_keywords, key="sb_quick")
        manual_search = st.text_input("...or type your own keyword:", "", key="ti_manual")

        # Logik für die Suchanfrage
        search_query = manual_search if manual_search else (None if selected_quick_filter == "All" else selected_quick_filter)

        st.markdown("---")

        # 2. Ergebnisse der Suche anzeigen
        if search_query:
            results = df_filtered[df_filtered['review_text'].str.contains(search_query, case=False, na=False)]
            
            if not results.empty:
                st.success(f"Found {len(results)} reviews containing '{search_query}':")
                st.dataframe(results[['rating', 'review_text', 'date']], use_container_width=True)
            else:
                st.warning(f"No reviews found containing '{search_query}'.")
        else:
            st.write("Showing latest reviews (All):")
            st.dataframe(df_filtered[['rating', 'review_text', 'date']].head(15), use_container_width=True)

        st.markdown("---")
        
        # 3. Keyword-Analyse (Optimiert)
    st.subheader("🔝 Top 10 Keywords (Overall Sentiment)")

    # Textvorbereitung: Alles klein, Sonderzeichen raus
    all_text = " ".join(df_filtered['review_text'].fillna("").astype(str)).lower()
    all_text = ''.join(e for e in all_text if e.isalnum() or e.isspace())
    words = pd.Series(all_text.split())

    # ERWEITERTE STOPWORD-LISTE (Hier liegt die Lösung)
    stop_words = [
        'the', 'and', 'to', 'for', 'is', 'it', 'with', 'a', 'in', 'of', 'i', 'was', 'on', 'at', 'as', 'be',
        'from', 'my', 'they', 'you', 'not', 'that', 'have', 'this', 'are', 'it', 'me', 'so', 'but', # Hinzugefügt
        'die', 'der', 'und', 'ist', 'das', 'für', 'ein', 'eine', 'mit', 'auf', 'zu', 'den', 'im', 'dem', 'es',
        'ich', 'sie', 'nicht', 'war', 'haben', 'habe', 'einer', 'einen', 'einem' # Deutsche Erweiterung
    ]

    # Filtern der Stopwords
    top_words = words[~words.isin(stop_words)].value_counts().head(20) # Top 20 Keywords

    if not top_words.empty:
        # Plotly Bar Chart (Horizontal)
        fig_words = px.bar(
            top_words, 
            x=top_words.values, 
            y=top_words.index, 
            orientation='h', 
            labels={'x': 'Frequency', 'index': 'Keyword'},
            color=top_words.values,
            color_continuous_scale='Blues' # Autodoc-Style
        )
        
        # Layout-Feinschliff
        fig_words.update_layout(
            yaxis={'categoryorder':'total ascending'}, # Höchster Wert oben
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        
        st.plotly_chart(fig_words, use_container_width=True)
    else:
        st.info("No relevant keywords found with the current filters.")

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