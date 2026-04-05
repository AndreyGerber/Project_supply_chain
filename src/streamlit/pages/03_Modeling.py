import streamlit as st

# 1. Load data from session state
if 'ml_data' in st.session_state:
    df = st.session_state['ml_data']
    st.success(f"✅ Dataset with {df.shape[0]} rows loaded successfully!")

    # 2. The Expander (as requested, in English)
    with st.expander("🔍 View Raw Data Columns"):
        st.write("Current columns in our dataset:")
        # We use 'df' here instead of 'df_processed'
        remaining_cols = list(df.columns)
        st.code(f"{remaining_cols}")

    # 3. Preview of the first 10 rows
    st.write("### 📋 Data Preview (Top 10 Rows)")
    st.dataframe(df.head(10), use_container_width=True)

else:
    st.error("⚠️ No data found. Please run the Preprocessing first!")

st.markdown("---")