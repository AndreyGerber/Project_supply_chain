import streamlit as st

# 1. Load data from session state
if 'ml_data' in st.session_state:
    df = st.session_state['ml_data']
    st.success(f"✅ Dataset with {df.shape[0]} rows and {df.shape[1]} columns loaded!")

    # 2. The Expander for column names
    with st.expander("🔍 View Raw Data Columns"):
        st.write("Remaining columns in our dataset:")
        st.code(list(df.columns))

    st.markdown("---")

    # 3. DYNAMIC ROW SELECTOR (The Slider)
    # This lets the user choose how many rows to display
    num_rows = st.slider(
        label="Select number of rows to preview:", 
        min_value=1, 
        max_value=min(100, len(df)), # Max 100 or total length of DF
        value=10  # Default value
    )

        # 4. Preview with the dynamic variable 'num_rows'
        st.write(f"### 📋 Previewing the first {num_rows} Rows")
        st.dataframe(df.head(num_rows), use_container_width=True)

else:
    st.error("⚠️ Please run the Preprocessing step first to load the data.")

st.markdown("---")