
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))


from src.service.run_model import (
    explain_local,
    get_tfidf_words,
    predict_explain
)

from pathlib import Path

st.title("🔍 Model Interpretability Dashboard")

text = st.text_area("Enter a review for explanation:")

if st.button("Explain"):

    # =========================================
    # MODEL OUTPUT
    # =========================================
    result = predict_explain(text)

    proba = result["probabilities"]
    pred = result["prediction"]

    st.subheader(f"⭐ Predicted Rating: {pred}")

    # =========================================
    # CONFIDENCE
    # =========================================
    st.subheader("📊 Confidence")

    fig, ax = plt.subplots()
    ax.bar([1, 2, 3, 4, 5], proba)
    st.pyplot(fig)

    # =========================================
    # LOCAL EXPLANATION
    # =========================================
    st.subheader("🔍 Local Feature Impact")

    importance = explain_local(result)

    fig2, ax2 = plt.subplots()
    ax2.bar(range(20), importance[:20])
    st.pyplot(fig2)

    # =========================================
    # TF-IDF WORDS
    # =========================================
    st.subheader("📝 Key Words (TF-IDF)")

    top_words = get_tfidf_words(result)

    st.dataframe(top_words)


# =========================================
# GLOBAL FEATURE IMPORTANCE
# =========================================
st.title("🔍 Feature Importance (Global)")

RESULT_PATH = Path("reports/feature_importance.csv")

if RESULT_PATH.exists():

    df = pd.read_csv(RESULT_PATH, index_col=0)

    st.dataframe(df)
    st.bar_chart(df["mean"])

else:
    st.warning("No analysis results found. Run feature analysis first.")


# =========================================
# EXPLANATION TEXT
# =========================================
st.markdown("""
### 🧠 Interpretation

We use:

- TF-IDF → lexical signal (keywords)
- Embeddings → semantic meaning
- Structure → stability signals

👉 The combination improves robustness and reduces extreme misclassification.
""")