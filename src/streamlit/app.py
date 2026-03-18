import streamlit as st
import pandas as pd
import plotly.express as px 
from pathlib import Path

# 1. Configuration
st.set_page_config(page_title="Autodoc Review Dashboard", layout="wide")

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
    st.title("📊 Autodoc Customer Insights Dashboard")
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
            st.dataframe(company_counts, use_container_width=True, hide_index=True)
        with c2:
            fig_comp = px.bar(company_counts, x='Review Count', y='Company Name', 
                              orientation='h', height=250, title="Reviews per Company")
            st.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.warning("Column 'company' not found in data.")
        
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
        search = st.text_input("Filter comments by keyword", "")
        if search:
            results = df_filtered[df_filtered['review_text'].str.contains(search, case=False, na=False)]
            st.success(f"Matches found: {len(results)}")
            st.dataframe(results[['rating', 'review_text', 'date']], use_container_width=True)

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
    highlight = '<span style="font-weight: bold; color: #ff4b4b; font-size: 1.1em;">'
    st.markdown(f'''
    Dear {highlight}Olga</span> and cool {highlight}Robert</span>, 
    this is our English Streamlit interface for the Supply Chain Analysis project!
    ''', unsafe_allow_html=True)

else:
    st.warning("Data could not be loaded. Please check the source file.")