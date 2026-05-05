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
        model, tfidf_pipeline, le, emb_version = load_model()

        df = preprocess_input([text])

        # TF-IDF
        X_tfidf = tfidf_pipeline.transform(df["review_text_clean_en"])

        # Embeddings
        X_emb = generate_embeddings(df, version=emb_version)

        # Structured
        X_struct = get_structured_features(df)

        X = hstack([
            X_tfidf,
            csr_matrix(X_emb),
            csr_matrix(X_struct)
        ])

        pred_encoded = model.predict(X)[0]
        pred = le.inverse_transform([pred_encoded])[0]

        proba = model.predict_proba(X)[0]

        st.success(f"⭐ Predicted Rating: {pred}")

        st.subheader("Confidence")

        for i, p in enumerate(proba):
            st.write(f"{i+1} stars: {p:.2f}")