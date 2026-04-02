import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# --- 1. PAGE CONFIGURATION & RESOURCES ---
st.set_page_config(page_title="Phase 2: Preprocessing", layout="wide")

@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

download_nltk_data()

# --- 2. HEADER & INTRODUCTION ---
st.title("🧹 Phase 2: Natural Language Preprocessing (NLP)")
st.markdown("""
In this step, we prepare our raw review texts for Machine Learning. 
First, we verify our dataset and identify noise such as automated system replies and duplicates.
""")

# --- 3. DATA RETRIEVAL FROM SESSION STATE ---
if 'raw_data' in st.session_state:
    df = st.session_state['raw_data']
    st.success(f"✅ Dataset linked successfully! ({len(df)} rows loaded)")

    # --- 4. DATASET OVERVIEW (CUSTOM HTML TABLE) ---
    st.write("### 📋 Dataset Column Overview")
    
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

    # --- 5. DUPLICATE & SYSTEM REPLY ANALYSIS ---
    st.subheader("🔍 Deep Dive: Analyzing Duplicates")
    
    # Logic: Identify "Reply from" rows vs. genuine user text duplicates
    system_mask = df['review_text'].str.contains(r"^Reply from", na=False, case=False, regex=True)
    df_system = df[system_mask]
    df_no_system = df[~system_mask]
    
    # Calculate extra rows (duplicates) within the user comments only
    extra_rows_count = len(df_no_system) - df_no_system['review_text'].nunique()
    total_noise = len(df_system) + extra_rows_count

    # A. System Replies Summary
    st.write(f"**A. System Replies:** Found {len(df_system)} rows that are automated company responses.")
    
    if not df_system.empty:
        company_summary = df_system.groupby('company')['review_text'].agg(['count', 'first']).reset_index()
        company_summary.columns = ['Company', 'Count', 'Example Text Content']
        st.dataframe(company_summary.sort_values('Count', ascending=False), use_container_width=True, hide_index=True)

    # B. Genuine Comment Duplicates
    st.write(f"**B. Genuine Comment Duplicates:** Identified {extra_rows_count} redundant copies of customer phrases.")
    
    top_duplicates = df_no_system['review_text'].value_counts().head(5).reset_index()
    top_duplicates.columns = ['Review Content', 'Occurrence Count']
    
    st.dataframe(
        top_duplicates,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Review Content": st.column_config.TextColumn("Review Content", width="large"),
            "Occurrence Count": st.column_config.NumberColumn("Count")
        }
    )

    # Final Conclusion Box
    st.info(f"""
        💡 **Conclusion:** We have identified **{total_noise}** entries to be removed:
        * **{len(df_system)}** are automated system replies.
        * **{extra_rows_count}** are extra copies of common customer phrases.
        
        Target dataset size after cleaning: **{df['review_text'].nunique()}** unique customer reviews.
    """)

else:
    st.warning("⚠️ No data found in Session State. Please go back to Phase 1 and load the dataset.")

# --- 6. NEXT STEPS ---
st.markdown("---")
st.write("Ready to proceed with **Text Cleaning** (removing stopwords, special characters, and lowering case)?")
