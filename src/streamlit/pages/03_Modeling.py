import streamlit as st

if 'ml_data' in st.session_state:
    df = st.session_state['ml_data'] # Hier definierst du 'df'
    st.success(f"Datensatz mit {df.shape[0]} Zeilen geladen!")
    
    st.write("### 🔍 Quick Preview of the first 10 rows")
    # Nutze hier 'df' statt 'df_final_view'
    st.dataframe(df.head(10), use_container_width=True) 

else:
    st.error("Bitte zuerst das Preprocessing durchlaufen!")

st.markdown("---")