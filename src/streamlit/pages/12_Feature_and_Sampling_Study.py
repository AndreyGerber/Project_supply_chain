import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast

st.set_page_config(layout="wide")

st.title("🧠 Feature Representation vs Sampling")

# =====================================
# Load Data
# =====================================

df_emb_basic = pd.read_csv("reports/sampling_comparison.csv")
df_emb_ext = pd.read_csv("reports/sampling_comparison_ETL.csv")
df_tfidf = pd.read_csv("reports/sampling_comparison_TF-idf.csv")

# Filter relevant experiments
selected = ["baseline", "undersample+weights", "weights_only"]

df_emb = df_emb_ext[df_emb_ext["experiment"].isin(selected)]
df_tfidf = df_tfidf[df_tfidf["experiment"].isin(selected)]

# =====================================
# 1. Performance Comparison
# =====================================

st.header("📊 Performance: TF-IDF vs Embeddings")

perf_df = pd.DataFrame({
    "Feature": ["Embeddings", "TF-IDF"],
    "Macro F1": [
        df_emb[df_emb["experiment"] == "baseline"]["macro_f1"].values[0],
        df_tfidf[df_tfidf["experiment"] == "baseline"]["macro_f1"].values[0]
    ],
    "RMSE": [
        df_emb[df_emb["experiment"] == "baseline"]["rmse"].values[0],
        df_tfidf[df_tfidf["experiment"] == "baseline"]["rmse"].values[0]
    ]
})

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    ax.bar(perf_df["Feature"], perf_df["Macro F1"])
    ax.set_title("Macro F1 Comparison")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    ax.bar(perf_df["Feature"], perf_df["RMSE"])
    ax.set_title("RMSE Comparison")
    st.pyplot(fig)

st.markdown("""
### 🧠 Key Insight

- Embeddings significantly outperform TF-IDF in Macro F1  
- TF-IDF shows much higher RMSE  

> This indicates that TF-IDF produces more severe misclassifications
""")

# =====================================
# 2. Sampling Impact
# =====================================

st.header("⚖️ Impact of Sampling Strategies")

fig, ax = plt.subplots()

for df, label in [(df_emb, "Embeddings"), (df_tfidf, "TF-IDF")]:
    ax.plot(df["experiment"], df["macro_f1"], marker='o', label=label)

ax.set_title("Sampling Impact on Macro F1")
ax.legend()

st.pyplot(fig)

st.markdown("""
### 🧠 Insight

- Baseline consistently performs best  
- Undersampling significantly degrades performance  
- Weighting provides only marginal improvements  

> Sampling does not solve the core problem
""")

# =====================================
# 3. Confusion Matrix Comparison
# =====================================

st.header("🔍 Confusion Matrix Comparison (Baseline)")

def plot_cm(cm, title):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    return fig

cm_emb = np.array(ast.literal_eval(
    df_emb[df_emb["experiment"] == "baseline"]["confusion_matrix"].values[0]
))

cm_tfidf = np.array(ast.literal_eval(
    df_tfidf[df_tfidf["experiment"] == "baseline"]["confusion_matrix"].values[0]
))

col1, col2 = st.columns(2)

with col1:
    st.pyplot(plot_cm(cm_emb, "Embeddings"))

with col2:
    st.pyplot(plot_cm(cm_tfidf, "TF-IDF"))

# =====================================
# 4. Error Distance Analysis
# =====================================

st.header("📏 Error Distance Analysis")

def compute_error_distance(cm):
    total_error = 0
    total_samples = cm.sum()

    for i in range(len(cm)):
        for j in range(len(cm)):
            total_error += abs(i - j) * cm[i][j]

    return total_error / total_samples

err_emb = compute_error_distance(cm_emb)
err_tfidf = compute_error_distance(cm_tfidf)

st.metric("Avg Error Distance (Embeddings)", f"{err_emb:.3f}")
st.metric("Avg Error Distance (TF-IDF)", f"{err_tfidf:.3f}")

st.markdown("""
### 🧠 Key Insight

- TF-IDF produces significantly larger prediction errors  
- Embeddings tend to make *nearby mistakes*  

Examples:
- TF-IDF: 1⭐ → 5⭐ (extreme error)  
- Embeddings: 3⭐ → 4⭐ (boundary error)  

> Embeddings better capture semantic relationships between classes
""")

# =====================================
# 5. Final Insight
# =====================================

st.header("🏁 Key Conclusions")

st.markdown("""
### Core Findings

- Sampling strategies do not significantly improve performance  
- Feature representation is the dominant factor  
- TF-IDF and embeddings capture fundamentally different signals  

### Interpretation

- TF-IDF:
  - Strong class separation  
  - Weak semantic understanding  
  - Leads to extreme errors  

- Embeddings:
  - Better semantic representation  
  - More stable predictions  
  - Smaller error distances  

### Final Insight

> The main challenge is not imbalance, but semantic ambiguity between classes.

This insight directly motivated the development of hybrid models in the next phase.
""")