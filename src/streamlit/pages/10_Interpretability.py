import streamlit as st
import pandas as pd
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.models.predict_model import load_model, preprocess_input
from src.features.build_features import (
    generate_embeddings,
    get_structured_features
)

from scipy.sparse import hstack, csr_matrix

from src.analysis.tfidf_explain import get_top_tfidf_words
from src.analysis.explain_local import explain_local_prediction


st.title("🔍 Model Interpretability Dashboard")

text = st.text_area("Enter a review for explanation:")

if st.button("Explain"):

    model, tfidf_pipeline, le, emb_version = load_model()

    df = preprocess_input([text])

    # ==============================
    # FEATURES
    # ==============================
    X_tfidf = tfidf_pipeline.transform(df["review_text_clean_en"])
    X_emb = generate_embeddings(df, version=emb_version)
    X_struct = get_structured_features(df)

    X = hstack([
        X_tfidf,
        csr_matrix(X_emb),
        csr_matrix(X_struct)
    ])

    # ==============================
    # PREDICTION
    # ==============================
    proba = model.predict_proba(X)[0]
    pred = np.argmax(proba) + 1

    st.subheader(f"⭐ Predicted Rating: {pred}")

    # ==============================
    # CONFIDENCE
    # ==============================
    st.subheader("📊 Confidence")

    fig, ax = plt.subplots()
    ax.bar([1,2,3,4,5], proba)
    st.pyplot(fig)

    # ==============================
    # LOCAL EXPLANATION
    # ==============================
    st.subheader("🔍 Local Feature Impact")

    importance = explain_local_prediction(model, X)

    fig2, ax2 = plt.subplots()
    ax2.bar(range(20), importance[:20])
    st.pyplot(fig2)

    # ==============================
    # TF-IDF TOP WORDS
    # ==============================
    st.subheader("📝 Key Words (TF-IDF)")

    top_words = get_top_tfidf_words(X_tfidf, tfidf_pipeline)

    st.dataframe(top_words)

    st.title("🔍 Feature Importance (Global)")



# =========================================
# LOAD RESULTS
# =========================================
RESULT_PATH = Path("analysis_results/feature_importance.csv")

if RESULT_PATH.exists():

    df = pd.read_csv(RESULT_PATH, index_col=0)

    st.subheader("📊 Feature Importance (Global)")

    st.dataframe(df)

    st.bar_chart(df["mean"])

else:
    st.warning("No analysis results found. Run feature analysis first.")

# =========================================
# EXPLANATION TEXT
# =========================================
st.markdown("""
### Explanation
To explains how the model makes decisions we use **Permutation Feature Importance** 
to measure how much each feature group contributes.
            
### Interpretation
- **TF-IDF**: captures specific words → strong for clear signals  
- **Embeddings**: capture semantic meaning → smoother predictions  
- **Structural Features**: stabilize predictions (e.g. negation, sentiment)

👉 This confirms that combining lexical and semantic features improves performance.
""")