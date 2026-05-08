import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score


st.set_page_config(page_title="Modeling Foundation", layout="wide")

# -----------------------------
# Session State for Navigation
# -----------------------------
if "study" not in st.session_state:
    st.session_state.study = "Binary Classification"

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
        ("Binary Classification", "Multi-Class Classification"),
        index=0 if st.session_state.study == "Binary Classification" else 1
    )
    st.session_state.study = study

with col2:
    if st.button("➡️ Move to Real-World Problem"):
        st.session_state.study = "Multi-Class Classification"
        st.experimental_rerun()

# -----------------------------
# STUDY 1
# -----------------------------
if st.session_state.study == "Binary Classification":
    st.header("Simplified Binary Task")

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

    st.subheader("Insight")
    st.info(
        "All models perform strongly in the simplified setting. "
        "Embeddings consistently outperform TF-IDF, but overall the task appears easy."
    )

    st.markdown("💬 *Interpretation:* The task does not yet expose the real difficulty of the problem.")

# -----------------------------
# STUDY 2
# -----------------------------
else:
    st.header("Full Multi-Class Task")

    st.markdown(
        """
        **Goal:** Evaluate XG Boost model performance under real-world conditions.
        
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


# =========================
# Daten
# =========================

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

# =========================
# Helper Function
# =========================

def metrics_from_cm(cm):

    y_true = []
    y_pred = []

    for true_class in range(cm.shape[0]):
        for pred_class in range(cm.shape[1]):
            count = cm[true_class, pred_class]

            y_true.extend([true_class] * count)
            y_pred.extend([pred_class] * count)

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None
    )

    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "f1_per_class": f1
    }

# =========================
# Compute Metrics
# =========================

metrics_emb = metrics_from_cm(cm_emb)
metrics_tfidf = metrics_from_cm(cm_tfidf)

# =========================
# Plot Function
# =========================

def create_plot(metrics, title):

    classes = [f"Klasse {i}" for i in range(len(metrics["f1_per_class"]))]

    fig, ax = plt.subplots(figsize=(8,5))

    bars = ax.bar(classes, metrics["f1_per_class"])

    ax.set_ylim(0, 1)
    ax.set_ylabel("F1-Score")
    ax.set_title(title)

    # Global metrics as horizontal lines
    ax.axhline(
        metrics["accuracy"],
        linestyle="--",
        label=f'Accuracy: {metrics["accuracy"]:.3f}'
    )

    ax.axhline(
        metrics["macro_f1"],
        linestyle="-.",
        label=f'Macro-F1: {metrics["macro_f1"]:.3f}'
    )

    ax.axhline(
        metrics["weighted_f1"],
        linestyle=":",
        label=f'Weighted-F1: {metrics["weighted_f1"]:.3f}'
    )

    ax.legend()

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.01,
            f"{height:.2f}",
            ha='center'
        )

    return fig

# =========================
# Streamlit Layout
# =========================

st.title("Vergleich: Embeddings vs TF-IDF")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Embeddings")
    fig1 = create_plot(metrics_emb, "F1-Scores pro Klasse")
    st.pyplot(fig1)

with col2:
    st.subheader("TF-IDF")
    fig2 = create_plot(metrics_tfidf, "F1-Scores pro Klasse")
    st.pyplot(fig2)

    # -----------------------------
    # Confusion Matrix Comparison
    # -----------------------------

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

    st.subheader("Insight")
    st.warning(
        "Performance drops significantly in the multi-class setting. "
        "The main challenge lies in distinguishing between similar rating levels, "
        "not extreme cases."
    )

    st.markdown("💬 *Interpretation:* The models do not fail at understanding sentiment — they fail at distinguishing its intensity.")