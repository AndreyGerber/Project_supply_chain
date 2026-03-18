import streamlit as st
import pandas as pd
import plotly.express as px 
import os
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
    
    # Date conversion
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    
    # 1. Extract the rating from the SVG (or existing column)
    # We'll just call it 'rating' and keep it clean
    df['rating'] = df['rating_svg'].str.extract('(\d+)').astype(int)
    
    # 2. DROP duplicate or unnecessary columns
    # We remove 'rating_numeric' and 'rating_svg' to keep the table slim
    columns_to_drop = ['rating_numeric', 'rating_svg']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    return df

# Initialize Data
df = load_data()

# Main Application Logic
if not df.empty:
    # 3. Sidebar Filtering
    st.sidebar.header("Filter Options")
    st.sidebar.write("Refine the data shown in the charts below:")
    
    selected_rating = st.sidebar.multiselect(
        "Select Rating", 
        options=sorted(df['rating'].unique()), 
        default=sorted(df['rating'].unique())
    )
    
    # Filter DataFrame based on selection
    df_filtered = df[df['rating'].isin(selected_rating)]

    # 4. Main Header
    st.title("📊 Autodoc Customer Insights Dashboard")
    st.markdown("This dashboard provides a comprehensive analysis of customer feedback and supplier performance.")
    st.markdown("---")
    
    # 5. Key Performance Indicators (KPIs)
    # Calculation of metrics
    avg_rating = df_filtered['rating'].mean()
    response_rate = df_filtered['supplier_response'].notna().mean() * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Reviews", len(df_filtered), help="Total number of reviews in the selected range.")
    col2.metric("Average Rating", f"{avg_rating:.2f} / 5.0", help="The arithmetic mean of all star ratings.")
    col3.metric("Supplier Response Rate", f"{response_rate:.1f}%", help="Percentage of reviews answered by Autodoc.")

    st.markdown("---")

    # 6. Analysis Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Performance Trends", "💬 Feedback Analysis", "📍 Operations & Support"])

    with tab1:
        st.subheader("Customer Satisfaction Distribution")
        st.info("This chart shows how many customers gave which star rating. It helps identify the overall sentiment shift.")
        
        fig = px.histogram(
            df_filtered,
            x="rating",
            color="rating",
            title="Frequency of Ratings",
            labels={'rating_numeric': 'Star Rating', 'count': 'Number of Reviews'},
            nbins=5,
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Raw Data Preview")
        st.dataframe(df_filtered.head(15), use_container_width=True)

    with tab2:
        st.subheader("💬 Sentiment & Keyword Discovery")
        st.info("Search through individual comments or identify the most frequent terms used by customers.")
        
        # Search Functionality
        search = st.text_input("Filter comments by keyword (e.g. 'shipping', 'quality')", "")
        
        if search:
            results = df_filtered[df_filtered['review_text'].str.contains(search, case=False, na=False)]
            st.success(f"Matches found: {len(results)}")
            st.dataframe(results[['rating_numeric', 'review_text', 'date']], use_container_width=True)
        
        st.markdown("---")
        st.subheader("🔝 Top 10 Keywords")
        
        # Word Frequency Calculation
        all_text = " ".join(df_filtered['review_text'].fillna("").astype(str)).lower()
        words = pd.Series(all_text.split())
        
        # Combined English/German Stopwords to clean the results
        stop_words = ['the', 'and', 'to', 'for', 'is', 'it', 'with', 'a', 'in', 'of', 'die', 'der', 'und', 'ist', 'das', 'für']
        top_words = words[~words.isin(stop_words)].value_counts().head(10)
        
        fig_words = px.bar(
            top_words, 
            x=top_words.values, 
            y=top_words.index, 
            orientation='h', 
            title="Most Frequent Words in Reviews",
            labels={'x': 'Frequency', 'y': 'Keyword'}
        )
        st.plotly_chart(fig_words, use_container_width=True)

    with tab3:
        st.header("📍 Geographic & Support Performance")
        st.info("Analysis of where reviews come from and how efficiently the support team responds to complaints.")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            top_loc = df_filtered['location'].value_counts().head(8)
            fig_loc = px.pie(
                values=top_loc.values, 
                names=top_loc.index, 
                title="Top Reviewer Regions", 
                hole=0.4
            )
            st.plotly_chart(fig_loc, use_container_width=True)
            
        with col_b:
            df_filtered['has_response'] = df_filtered['supplier_response'].notna()
            resp_counts = df_filtered['has_response'].value_counts().rename({True: 'Responded', False: 'Pending'})
            fig_resp = px.bar(
                x=resp_counts.index, 
                y=resp_counts.values, 
                title="Review Response Status",
                labels={'x': 'Support Status', 'y': 'Number of Reviews'},
                color=resp_counts.index,
                color_discrete_map={'Responded': '#00CC96', 'Pending': '#EF553B'}
            )
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