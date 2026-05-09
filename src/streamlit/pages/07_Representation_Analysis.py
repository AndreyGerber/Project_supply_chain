import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import ast

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Feature Representation vs Sampling",
    layout="wide"
)

# =========================================================
# HEADER
# =========================================================

st.title("🧠 Feature Representation vs Sampling")

st.markdown(
    """
    After observing that balancing alone could not fully resolve classification errors,
    we investigated whether model performance is primarily influenced by:

    - sampling strategy  
    - or feature representation
    """
)

st.markdown("---")

# =========================================================
# LOAD DATA
# =========================================================

df_emb_basic = pd.read_csv("reports/sampling_comparison.csv")
df_emb_ext = pd.read_csv("reports/sampling_comparison_ETL.csv")
df_tfidf = pd.read_csv("reports/sampling_comparison_TF-idf.csv")

# =========================================================
# PREPARE STUDIES
# =========================================================

studies = {
    "Embeddings + Basic Features": df_emb_basic,
    "Embeddings + Extended Features": df_emb_ext,
    "TF-IDF + Extended Features": df_tfidf
}

combined = []

for name, df in studies.items():
    temp = df.copy()
    temp["study"] = name
    combined.append(temp)

combined_df = pd.concat(combined)

# =========================================================
# SECTION 1 — RESEARCH QUESTION
# =========================================================

st.header("1️⃣ Research Question")

col1, col2 = st.columns([1.3, 1])

with col1:

    st.info(
        """
        ### Motivation
        
        Previous balancing experiments showed that:
        
        - class imbalance affects performance  
        - but balancing alone does not fully resolve classification errors  
        
        Persistent weaknesses in neighboring rating classes suggested
        that semantic ambiguity may be a stronger limitation.
        """
    )

with col2:

    st.success(
        """
        ### Research Question
        
        Which factor has the larger impact on performance?
        
        - Sampling strategy  
        or  
        - Feature representation
        
        ---
        
        We therefore compared multiple sampling strategies
        across TF-IDF and embedding-based representations.
        """
    )

# =========================================================
# SECTION 2 — GLOBAL PERFORMANCE COMPARISON
# =========================================================

st.header("2️⃣ Global Performance Comparison")

st.markdown(
    """
    To evaluate the relative influence of sampling and representation,
    Macro-F1 and RMSE were compared across all experiments.
    """
)

# ---------------------------------------------------------
# PLOTS
# ---------------------------------------------------------

col1, col2 = st.columns(2)

# =========================================================
# MACRO F1
# =========================================================

with col1:

    fig_f1 = px.line(
        combined_df,
        x="experiment",
        y="macro_f1",
        color="study",
        markers=True,
        title="Macro-F1 Across Experiments"
    )

    fig_f1.update_layout(
        yaxis_title="Macro-F1",
        xaxis_title="Experiment",
        height=450
    )

    st.plotly_chart(
        fig_f1,
        use_container_width=True
    )

# =========================================================
# RMSE
# =========================================================

with col2:

    fig_rmse = px.line(
        combined_df,
        x="experiment",
        y="rmse",
        color="study",
        markers=True,
        title="RMSE Across Experiments"
    )

    fig_rmse.update_layout(
        yaxis_title="RMSE",
        xaxis_title="Experiment",
        height=450
    )

    st.plotly_chart(
        fig_rmse,
        use_container_width=True
    )

# =========================================================
# INTERPRETATION
# =========================================================

st.info(
    """
    ### Initial Observation
    
    - Sampling strategies produced only limited improvements  
    - Undersampling consistently reduced performance  
    - Class weighting was generally more effective than aggressive resampling  
    
    ---
    
    Most importantly:
    
    - Embedding-based representations consistently outperformed TF-IDF  
    - TF-IDF produced higher RMSE values across nearly all configurations  
    
    This suggests that feature representation has a larger influence on model behavior than sampling strategy alone.
    """
)

# =========================================================
# SECTION 3 — ERROR STRUCTURE ANALYSIS
# =========================================================

st.header("3️⃣ Error Structure Analysis")

st.markdown(
    """
    To better understand model behavior,
    we analyzed the structure and severity of prediction errors.
    """
)

# =========================================================
# LOAD BASELINE MATRICES
# =========================================================

cm_emb = np.array(ast.literal_eval(
    df_emb_ext[
        df_emb_ext["experiment"] == "baseline"
    ]["confusion_matrix"].values[0]
))

cm_tfidf = np.array(ast.literal_eval(
    df_tfidf[
        df_tfidf["experiment"] == "baseline"
    ]["confusion_matrix"].values[0]
))

# =========================================================
# CONFUSION MATRIX FUNCTION
# =========================================================

def plot_cm(cm, title):

    fig, ax = plt.subplots(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[1,2,3,4,5],
        yticklabels=[1,2,3,4,5],
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    return fig

# =========================================================
# MATRICES
# =========================================================

col1, col2 = st.columns(2)

with col1:
    st.pyplot(plot_cm(cm_emb, "Embeddings"))

with col2:
    st.pyplot(plot_cm(cm_tfidf, "TF-IDF"))

# =========================================================
# ERROR METRICS
# =========================================================

def compute_error_distance(cm):

    total_error = 0
    total_samples = cm.sum()

    for i in range(len(cm)):
        for j in range(len(cm)):
            total_error += abs(i - j) * cm[i][j]

    return total_error / total_samples

# ---------------------------------------------------------
# COMPUTE METRICS
# ---------------------------------------------------------

err_emb = compute_error_distance(cm_emb)
err_tfidf = compute_error_distance(cm_tfidf)

critical_emb = cm_emb[0:2, 3:5].sum()
critical_tfidf = cm_tfidf[0:2, 3:5].sum()

# ---------------------------------------------------------
# TABLE
# ---------------------------------------------------------

st.subheader("📋 Error Summary")

error_df = pd.DataFrame({
    "Model": ["Embeddings", "TF-IDF"],
    "Average Error Distance": [
        round(err_emb, 3),
        round(err_tfidf, 3)
    ],
    "Critical Misclassifications": [
        int(critical_emb),
        int(critical_tfidf)
    ]
})

st.dataframe(
    error_df,
    use_container_width=True
)

# =========================================================
# INTERPRETATION
# =========================================================

st.info(
    """
    ### Key Insight
    
    TF-IDF produced:
    
    - larger prediction distances  
    - more severe misclassifications  
    - weaker separation of neighboring classes  
    
    In contrast, embeddings tended to produce nearby prediction errors,
    suggesting a better preservation of semantic relationships.
    
    ---
    
    This indicates that embeddings capture semantic structure more effectively,
    particularly in ambiguous rating regions.
    """
)

# =========================================================
# SECTION 4 — SCIENTIFIC CONCLUSION
# =========================================================

st.header("4️⃣ Scientific Interpretation")

st.success(
    """
    ### Core Finding
    
    Sampling strategies produced only limited improvements,
    while feature representation consistently changed both
    overall performance and error behavior.
    
    ---
    
    Embeddings achieved:
    
    - higher Macro-F1  
    - lower RMSE  
    - smaller prediction distances  
    - fewer critical errors  
    
    This suggests that representation quality —
    not sampling strategy —
    is the dominant factor in handling semantic ambiguity.
    
    ---
    
    ### Transition to Next Phase
    
    The results raised an important question:
    
    Could TF-IDF and embeddings capture complementary signals
    rather than competing representations?
    
    → This motivated the development of hybrid models in the next phase.
    """
)