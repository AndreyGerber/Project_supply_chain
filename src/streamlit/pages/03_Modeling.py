import streamlit as st

# 1. Expandable Section for Raw Data (as seen in your image)
with st.expander("🔍 View Raw Data Columns"):
    st.write("Current columns in our dataset:")
    
    # Highlighting the column names in a code block for better readability
    remaining_cols = list(df_processed.columns)
    st.code(f"{remaining_cols}")

# 2. Data Preview Header
st.write("### Data Preview (First 10 rows)")

# 3. Interactive Dataframe
# This will show only the remaining columns of your processed DF
st.dataframe(
    df_processed.head(10), 
    use_container_width=True,
    hide_index=False
)

st.markdown("---")