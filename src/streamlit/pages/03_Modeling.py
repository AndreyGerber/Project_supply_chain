import streamlit as st

if 'ml_data' in st.session_state:
    df = st.session_state['ml_data']
    st.success(f"Datensatz mit {df.shape[0]} Zeilen geladen!")
else:
    st.error("Bitte zuerst das Preprocessing durchlaufen!")


st.write("### 🔍 Quick Preview of the first 10 rows")
st.dataframe(df_final_view.head(10), use_container_width=True)

st.markdown("---")