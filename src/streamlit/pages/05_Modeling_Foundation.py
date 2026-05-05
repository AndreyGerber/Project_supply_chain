import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Modeling Foundation", layout="wide")

# -----------------------------
# Session State for Navigation
# -----------------------------
if "study" not in st.session_state:
    st.session_state.study = "Study 1: Binary Classification"

# -----------------------------
# Header / Framing
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
        st.experimental_rerun()

# -----------------------------
# STUDY 1
# -----------------------------
if st.session_state.study == "Study 1: Binary Classification":
    st.header("🔹 Study 1: Simplified Binary Task")

    st.markdown(
        """
        **Goal:** Understand model behavior in a simplified setting.
        
        Binary classification: **Rating 5 vs. Not 5**.
        
        This isolates the effect of **feature representation** and **model choice** without fine-grained ambiguity.
        """
    )

    data = pd.DataFrame({
        "Model": ["XGBoost", "Random Forest", "XGBoost", "Random Forest"],
        "Feature Type": ["Embeddings", "Embeddings", "TF-IDF", "TF-IDF"],
        "Accuracy": [0.972393, 0.959739, 0.947469, 0.854678],
        "F1 Score": [0.972319, 0.959351, 0.946993, 0.831086]
    })

    st.subheader("📈 Results")
    st.dataframe(data, use_container_width=True)

    st.subheader("📊 Comparison (Accuracy & F1)")
    pivot = data.pivot(index="Feature Type", columns="Model", values="F1 Score")
    st.bar_chart(pivot)

    st.subheader("🧠 Key Insight")
    st.info(
        "All models perform strongly in the simplified setting. "
        "Embeddings consistently outperform TF-IDF, but overall the task appears easy."
    )

    st.markdown("💬 *Interpretation:* The task does not yet expose the real difficulty of the problem.")

# -----------------------------
# STUDY 2
# -----------------------------
else:
    st.header("🔹 Study 2: Full Multi-Class Task")

    st.markdown(
        """
        **Goal:** Evaluate model performance under real-world conditions.
        
        5-class classification: **Ratings from 1 to 5**.
        
        This introduces **semantic complexity** and **class ambiguity**.
        """
    )

    results = pd.DataFrame({
        "Feature Type": ["Embeddings", "TF-IDF"],
        "Accuracy": [0.862933, 0.730725],
        "Macro F1": [0.684148, 0.468757],
        "RMSE": [0.480141, 0.708767]
    })

    st.subheader("📉 Results")
    st.dataframe(results, use_container_width=True)

    st.subheader("📊 Performance Drop Visualization")
    st.bar_chart(results.set_index("Feature Type")[["Accuracy", "Macro F1"]])

    # -----------------------------
    # Confusion Matrix Comparison
    # -----------------------------
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

    st.header("🔍 Confusion Matrix Comparison (Baseline)")

    def plot_cm(cm, title):
        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap="Blues")

        # Annotate cells (like seaborn style)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=9)

        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(cm.shape[1]))
        ax.set_yticks(range(cm.shape[0]))

        ax.set_xticklabels([1,2,3,4,5])
        ax.set_yticklabels([1,2,3,4,5])

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        return fig

    col1, col2 = st.columns(2)

    with col1:
        st.pyplot(plot_cm(cm_emb, "Embeddings"))

    with col2:
        st.pyplot(plot_cm(cm_tfidf, "TF-IDF"))

    st.subheader("🧠 Key Insight")
    st.warning(
        "Performance drops significantly in the multi-class setting. "
        "The main challenge lies in distinguishing between similar rating levels, "
        "not extreme cases."
    )

    st.markdown("💬 *Interpretation:* The models do not fail at understanding sentiment — they fail at distinguishing its intensity.")

st.markdown("---")
st.caption("Use the dropdown or the 'However' button to guide the narrative during your presentation.")