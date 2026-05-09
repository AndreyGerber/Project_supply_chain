import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    f1_score,
    mean_squared_error
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Modeling Foundation",
    layout="wide"
)

# =========================================================
# SESSION STATE
# =========================================================

if "study" not in st.session_state:
    st.session_state.study = "Binary Classification"

# =========================================================
# HEADER
# =========================================================

st.title("📊 Modeling Foundation")

st.markdown(
    """
    **We start with a controlled setup to isolate representation quality — then move to the real-world problem.**
    """
)

st.markdown("---")

# =========================================================
# NAVIGATION
# =========================================================

col1, col2 = st.columns([4, 1])

with col1:
    study = st.selectbox(
        "Select Study",
        (
            "Binary Classification",
            "Multi-Class Classification"
        ),
        index=0 if st.session_state.study == "Binary Classification" else 1
    )

    st.session_state.study = study

with col2:
    st.write("")
    st.write("")

    if st.button("➡️ Move to Real-World Problem"):
        st.session_state.study = "Multi-Class Classification"
        st.experimental_rerun()

# =========================================================
# STUDY 1 — BINARY
# =========================================================

if st.session_state.study == "Binary Classification":

    st.header("1️⃣ Simplified Binary Task")

    st.markdown(
        """
        ### Goal
        
        The binary setup serves as a **controlled experiment** to isolate the effect of:
        
        - feature representation
        - model architecture
        
        before introducing **ordinal ambiguity** and semantic overlap between neighboring classes.
        
        Binary classification:
        
        **Rating 5 vs. Not 5**
        """
    )

    # -----------------------------------------------------
    # DATA
    # -----------------------------------------------------

    binary_data = pd.DataFrame({
        "Model": [
            "XGBoost",
            "Random Forest",
            "XGBoost",
            "Random Forest"
        ],
        "Feature Type": [
            "Embeddings",
            "Embeddings",
            "TF-IDF",
            "TF-IDF"
        ],
        "Accuracy": [
            0.972393,
            0.959739,
            0.947469,
            0.854678
        ],
        "F1 Score": [
            0.972319,
            0.959351,
            0.946993,
            0.831086
        ]
    })

    # -----------------------------------------------------
    # RESULTS TABLE
    # -----------------------------------------------------

    st.subheader("📋 Results Overview")

    st.dataframe(
        binary_data,
        use_container_width=True
    )

    # -----------------------------------------------------
    # PLOT
    # -----------------------------------------------------

    st.subheader("📈 Binary Classification Performance")

    fig_binary = px.bar(
        binary_data,
        x="Feature Type",
        y="F1 Score",
        color="Model",
        barmode="group",
        text="F1 Score",
        height=500
    )

    fig_binary.update_traces(
        texttemplate='%{text:.3f}',
        textposition='outside'
    )

    fig_binary.update_layout(
        yaxis_range=[0, 1],
        yaxis_title="F1 Score",
        xaxis_title="Feature Representation",
        legend_title="Model"
    )

    st.plotly_chart(fig_binary, use_container_width=True)

    # -----------------------------------------------------
    # INSIGHT
    # -----------------------------------------------------

    st.info(
        """
        **Insight**
        
        All models perform strongly in the simplified setting.
        
        Embeddings consistently outperform TF-IDF, but the task itself remains relatively easy due to the absence of fine-grained semantic distinctions.
        
        The experiment therefore acts as a baseline to understand representation quality before introducing ordinal complexity.
        """
    )

# =========================================================
# STUDY 2 — MULTI CLASS
# =========================================================

else:

    st.header("2️⃣ Full Multi-Class Task")

    st.markdown(
        """
        ### Goal
        
        Evaluate model performance under realistic conditions.
        
        5-class classification:
        
        **Ratings from 1 to 5**
        
        This introduces:
        
        - semantic overlap between neighboring ratings
        - ordinal ambiguity
        - class imbalance
        """
    )

    # =====================================================
    # CONFUSION MATRICES
    # =====================================================

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

    # =====================================================
    # METRICS
    # =====================================================

    def metrics_from_cm(cm):

        y_true = []
        y_pred = []

        for true_class in range(cm.shape[0]):
            for pred_class in range(cm.shape[1]):

                count = cm[true_class, pred_class]

                y_true.extend([true_class] * count)
                y_pred.extend([pred_class] * count)

        accuracy = accuracy_score(y_true, y_pred)

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            average=None
        )

        macro_f1 = f1_score(
            y_true,
            y_pred,
            average="macro"
        )

        weighted_f1 = f1_score(
            y_true,
            y_pred,
            average="weighted"
        )

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "rmse": rmse,
            "f1_per_class": f1
        }

    metrics_emb = metrics_from_cm(cm_emb)
    metrics_tfidf = metrics_from_cm(cm_tfidf)

    # =====================================================
    # KPI CARDS
    # =====================================================

    st.subheader("📋 Overall Performance")

    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)

    with kpi1:
        st.metric(
            "Emb Accuracy",
            f"{metrics_emb['accuracy']:.3f}"
        )

    with kpi2:
        st.metric(
            "Emb Macro-F1",
            f"{metrics_emb['macro_f1']:.3f}"
        )

    with kpi3:
        st.metric(
            "Emb RMSE",
            f"{metrics_emb['rmse']:.3f}"
        )

    with kpi4:
        st.metric(
            "TF-IDF Accuracy",
            f"{metrics_tfidf['accuracy']:.3f}"
        )

    with kpi5:
        st.metric(
            "TF-IDF Macro-F1",
            f"{metrics_tfidf['macro_f1']:.3f}"
        )

    with kpi6:
        st.metric(
            "TF-IDF RMSE",
            f"{metrics_tfidf['rmse']:.3f}"
        )

    st.markdown("---")

    # =====================================================
    # F1 COMPARISON
    # =====================================================

    st.subheader("📊 F1-Score per Class")

    f1_df = pd.DataFrame({
        "Class": [
            "1",
            "2",
            "3",
            "4",
            "5"
        ],
        "Embeddings": metrics_emb["f1_per_class"],
        "TF-IDF": metrics_tfidf["f1_per_class"]
    })

    f1_melted = f1_df.melt(
        id_vars="Class",
        var_name="Representation",
        value_name="F1 Score"
    )

    fig_f1 = px.bar(
        f1_melted,
        x="Class",
        y="F1 Score",
        color="Representation",
        barmode="group",
        text="F1 Score",
        height=500
    )

    fig_f1.update_traces(
        texttemplate='%{text:.2f}',
        textposition='outside'
    )

    fig_f1.update_layout(
        yaxis_range=[0, 1],
        xaxis_title="Rating Class",
        yaxis_title="F1 Score",
        legend_title="Representation"
    )

    st.plotly_chart(fig_f1, use_container_width=True)

    # =====================================================
    # NORMALIZED CONFUSION MATRICES
    # =====================================================

    st.subheader("🔍 Normalized Confusion Matrices")

    def normalize_cm(cm):
        return cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    norm_emb = normalize_cm(cm_emb)
    norm_tfidf = normalize_cm(cm_tfidf)

    col1, col2 = st.columns(2)

    with col1:

        fig_emb = px.imshow(
            norm_emb,
            text_auto=".2f",
            color_continuous_scale="Blues",
            labels=dict(
                x="Predicted",
                y="True",
                color="Ratio"
            ),
            x=["1", "2", "3", "4", "5"],
            y=["1", "2", "3", "4", "5"],
            title="Embeddings"
        )

        st.plotly_chart(
            fig_emb,
            use_container_width=True
        )

    with col2:

        fig_tfidf = px.imshow(
            norm_tfidf,
            text_auto=".2f",
            color_continuous_scale="Blues",
            labels=dict(
                x="Predicted",
                y="True",
                color="Ratio"
            ),
            x=["1", "2", "3", "4", "5"],
            y=["1", "2", "3", "4", "5"],
            title="TF-IDF"
        )

        st.plotly_chart(
            fig_tfidf,
            use_container_width=True
        )

    # =====================================================
    # INTERPRETATION
    # =====================================================

    st.info(
        """
        ### Insight
        
        Performance drops substantially in the multi-class setting.
        
        The main challenge is not detecting sentiment polarity itself, but distinguishing between neighboring sentiment intensities.
        
        The confusion matrices show that:
        
        - extreme ratings are classified reliably
        - middle classes exhibit strong semantic overlap
        - TF-IDF struggles particularly with neighboring classes
        
        Embeddings achieve consistently higher F1-scores across nearly all classes, indicating a better semantic representation.
        
        ---
        
        ### Why Macro-F1?
        
        The dataset is strongly imbalanced, with class 5 dominating the distribution.
        
        Accuracy alone would therefore overestimate performance.
        
        Macro-F1 evaluates all classes equally and provides a more reliable assessment of model robustness across minority classes.
        
        ---
        
        ### Why RMSE?
        
        Ratings are ordinal by nature.
        
        Misclassifying a 5-star review as 4 is less severe than predicting 1.
        
        RMSE captures this ordinal distance between predicted and true ratings, which standard classification metrics cannot express directly.
        """
    )