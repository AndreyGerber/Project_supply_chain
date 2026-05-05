import streamlit as st
from src.service.run_model import predict_rating

st.title("📊 Demo")

text = st.text_area("Enter review")

if st.button("Predict"):

    result = predict_rating(text)

    st.success(f"⭐ Rating: {result['prediction']}")

    st.bar_chart(result["probabilities"])