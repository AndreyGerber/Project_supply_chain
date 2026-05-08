import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="Final Model Evolution",
    layout="wide"
)

# =========================================================
# LOAD REPORTS
# =========================================================

comparison_df = pd.read_csv(
    "reports/final_model_comparison.csv"
)

per_class_df = pd.read_csv(
    "reports/final_per_class_f1.csv"
)

# =========================================================
# TITLE
# =========================================================

st.title("🚀 Final Model Evolution")

st.markdown(
    """
    After identifying feature representation as the dominant factor,
    the strongest components were combined into a final optimized pipeline.
    """
)

st.markdown("---")

# =========================================================
# SECTION 1 — FINAL PIPELINE
# =========================================================

st.header("1️⃣ Final Modeling Strategy")

col1, col2 = st.columns([1.3, 1])

with col1:

    st.info(
        """
        ### Motivation
        
        Previous experiments showed that:
        
        - sampling strategies produced limited improvements  
        - representation quality strongly influenced performance  
        
        We therefore focused on combining the strongest
        representation components into a unified model.
        """
    )

with col2:

    st.success(
        """
        ### Final Pipeline
        
        - XGBoost  
        - Structural Features  
        - Embeddings  
        - TF-IDF  
        - Hyperparameter Optimization
        """
    )

# =========================================================
# SECTION 2 — MODEL EVOLUTION
# =========================================================

st.header("2️⃣ Model Evolution")

fig, ax = plt.subplots(figsize=(8,5))

ax.plot(
    comparison_df["stage"],
    comparison_df["macro_f1_mean"],
    marker="o",
    linewidth=2
)

ax.set_ylim(0,1)

ax.set_ylabel("Macro-F1")
ax.set_title("Performance Evolution")

for i, value in enumerate(comparison_df["macro_f1_mean"]):
    ax.text(
        i,
        value + 0.01,
        f"{value:.3f}",
        ha="center"
    )

plt.xticks(rotation=10)

st.pyplot(fig)

st.info(
    """
    ### Observation
    
    Each modeling decision was directly motivated
    by the findings of the previous investigations.
    
    The largest improvements were achieved through:
    
    - stronger feature representations  
    - hybrid semantic + lexical signals  
    - targeted hyperparameter optimization
    """
)

# =========================================================
# SECTION 3 — FINAL PER-CLASS COMPARISON
# =========================================================

st.header("3️⃣ Baseline vs Final Hybrid Model")

col1, col2 = st.columns(2)

# ---------- BASELINE ----------

with col1:

    fig, ax = plt.subplots(figsize=(5,4))

    ax.bar(
        per_class_df["class"],
        per_class_df["baseline_f1"]
    )

    ax.set_ylim(0,1)

    ax.set_title("Baseline XGB + Embeddings")
    ax.set_xlabel("Class")
    ax.set_ylabel("F1 Score")

    st.pyplot(fig)

# ---------- HYBRID ----------

with col2:

    fig, ax = plt.subplots(figsize=(5,4))

    ax.bar(
        per_class_df["class"],
        per_class_df["hybrid_f1"]
    )

    ax.set_ylim(0,1)

    ax.set_title("Final Hybrid Model")
    ax.set_xlabel("Class")
    ax.set_ylabel("F1 Score")

    st.pyplot(fig)

# =========================================================
# SECTION 4 — FINAL METRICS
# =========================================================

st.header("4️⃣ Final Metrics Summary")

summary = comparison_df[
    ["stage", "macro_f1_mean", "rmse"]
]

summary.columns = [
    "Model",
    "Macro-F1",
    "RMSE"
]

st.dataframe(
    summary,
    use_container_width=True
)

# =========================================================
# SECTION 5 — FINAL CONCLUSION
# =========================================================

st.header("5️⃣ Scientific Conclusion")

st.success(
    """
    ### Key Finding
    
    The strongest improvements were achieved through
    representation enhancement rather than sampling-based interventions.
    
    ---
    
    Combining:
    
    - semantic embeddings  
    - lexical TF-IDF signals  
    - structural features  
    
    produced the most robust and stable model behavior.
    
    ---
    
    The final hybrid model achieved the best balance between:
    
    - semantic understanding  
    - lexical precision  
    - ordinal consistency
    """
)