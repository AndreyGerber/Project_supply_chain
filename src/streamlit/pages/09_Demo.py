import streamlit as st
from src.models.predict_model import predict, load_model, preprocess_input

from scipy.sparse import hstack, csr_matrix
from src.features.build_features import generate_embeddings, get_structured_features


st.title("📊 Demo: Predict Review Rating")

text = st.text_area("Enter a review:", height=150)

if st.button("Predict"):

    if text.strip() == "":
        st.warning("Please enter text.")
    else:
        pred =predict([text])
        
        st.success(f"⭐ Predicted Rating: {pred}")
