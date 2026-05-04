import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Modeling Foundation", layout="wide")

# -----------------------------
# Session State
# -----------------------------
if "study" not in st.session_state:
    st.session_state.study = "Study 1: Binary Classification"

# -----------------------------
# Header
# -----------------------------
st.title("📊 Modeling Foundation")
st.markdown("**We start simple to understand the models — then move to reality.**")
st.markdown("---")

# Controls
col1, col2 = st.columns([3, 1])
with col1:
    study = st.selectbox(
        "Select Study",
        ("Study 1: Binary Classification", "Study 2: Multi-Class Classification"),
        index=0 if st.session_state.study == "Study 1: Binary Classification" else 1
    )
    st.session_state.study = study

with col2:
    if st.button("➡️ However: Move to Real-World Problem"):
        st.session_state.study = "Study 2: Multi-Class Classification"
        st.rerun()

# -----------------------------
# STUDY 1
# -----------------------------
if st.session_state.study == "Study 1: Binary Classification":
    st.header("🔹 Study 1: Simplified Binary Task")

    st.markdown("""
    **Goal:** Understand model behavior in a simplified setting.
    
    Binary classification: **Rating 5 vs. Not 5**.
    """)

    data = pd.DataFrame({
        "Model": ["XGBoost", "Random Forest", "XGBoost", "Random Forest"],
        "Feature": ["Embeddings", "Embeddings", "TF-IDF", "TF-IDF"],
        "Accuracy": [0.972393, 0.959739, 0.947469, 0.854678],
        "F1": [0.972319, 0.959351, 0.946993, 0.831086]
    })

    with st.expander("📋 Show Raw Results"):
        st.dataframe(data, use_container_width=True)

    st.subheader("📊 Model Comparison")

    # Bar plot grouped
    fig, ax = plt.subplots()

    x = np.arange(len(data["Feature"].unique()))
    width = 0.35

    emb = data[data["Feature"] == "Embeddings"]
    tfidf = data[data["Feature"] == "TF-IDF"]

    ax.bar(x - width/2, emb["F1"], width, label="XGBoost")
    ax.bar(x + width/2, tfidf["F1"], width, label="Random Forest")

    ax.set_xticks(x)
    ax.set_xticklabels(["Embeddings", "TF-IDF"])
    ax.set_ylabel("F1 Score")
    ax.legend()

    st.pyplot(fig)

    st.info("All models perform strongly. Differences are small in this simplified setup.")

# -----------------------------
# STUDY 2
# -----------------------------
else:
    st.header("🔹 Study 2: Full Multi-Class Task")

    st.markdown("""
    **Goal:** Evaluate performance under real-world conditions.
    
    Ratings from **1 to 5**.
    """)

    results = pd.DataFrame({
        "Feature": ["Embeddings", "TF-IDF"],
        "Accuracy": [0.862933, 0.730725],
        "Macro F1": [0.684148, 0.468757],
        "RMSE": [0.480141, 0.708767]
    })

    with st.expander("📋 Show Raw Results"):
        st.dataframe(results, use_container_width=True)

    st.subheader("📊 Performance Comparison")

    fig2, ax2 = plt.subplots()

    x = np.arange(len(results))
    width = 0.25

    ax2.bar(x - width, results["Accuracy"], width, label="Accuracy")
    ax2.bar(x, results["Macro F1"], width, label="Macro F1")
    ax2.bar(x + width, results["RMSE"], width, label="RMSE")

    ax2.set_xticks(x)
    ax2.set_xticklabels(results["Feature"])
    ax2.legend()

    st.pyplot(fig2)

    # Confusion Matrices
    cm_emb = np.array([
        [337, 85, 29, 4, 0],
        [6, 53, 11, 5, 0],
        [3, 16, 55, 13, 2],
        [2, 4, 20, 142, 47],
        [1, 1, 28, 101, 1642]
    ])

    cm_tfidf = np.array([
        [222, 161, 51, 20, 1],
        [6, 36, 23, 8, 2],
        [2, 20, 37, 27, 3],
        [0, 10, 17, 87, 101],
        [0, 6, 48, 208, 1511]
    ])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Embeddings")
        fig3, ax3 = plt.subplots()
        cax = ax3.imshow(cm_emb)
        fig3.colorbar(cax)
        st.pyplot(fig3)

    with col2:
        st.subheader("TF-IDF")
        fig4, ax4 = plt.subplots()
        cax = ax4.imshow(cm_tfidf)
        fig4.colorbar(cax)
        st.pyplot(fig4)

    st.warning("Performance drops significantly. The challenge lies in fine-grained distinctions.")

st.markdown("---")
st.caption("Use dropdown or 'However' button to guide the narrative.")
