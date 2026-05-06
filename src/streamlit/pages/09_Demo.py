
import streamlit as st
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.service.run_model import predict_rating

st.title("📊 Demo")

text = st.text_area("Enter review")

if st.button("Predict"):

    result = predict_rating(text)

    st.success(f"⭐ Rating: {result}")