import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast

st.set_page_config(layout="wide")

st.title("🧠 Feature Representation vs Sampling")

# ---------- Helper Functions ----------

def parse_confusion_matrix(cm_str):
    return np.array(ast.literal_eval(cm_str))


def compute_per_class_metrics(cm):
    num_classes = cm.shape[0]
    precision, recall, f1 = [], [], []

    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0

        precision.append(p)
        recall.append(r)
        f1.append(f)

    return precision, recall, f1


def confusion_distance(cm):
    distance = 0
    total = 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            distance += abs(i - j) * cm[i, j]
            total += cm[i, j]
    return distance / total


def plot_bar(df, metric, title):
    fig, ax = plt.subplots()
    ax.bar(df['experiment'], df[metric])
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.set_xticks(range(len(df['experiment'])))
    ax.set_xticklabels(df['experiment'], rotation=45)
    st.pyplot(fig)


def plot_per_class(metric_values, title):
    fig, ax = plt.subplots()
    classes = [1, 2, 3, 4, 5]
    ax.bar(classes, metric_values)
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_xticks(classes)
    st.pyplot(fig)


def plot_confusion_matrix(cm):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=[1,2,3,4,5], yticklabels=[1,2,3,4,5])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

# =====================================
# Load Data
# =====================================

df_emb_basic = pd.read_csv("reports/sampling_comparison.csv")
df_emb_ext = pd.read_csv("reports/sampling_comparison_ETL.csv")
df_tfidf = pd.read_csv("reports/sampling_comparison_TF-idf.csv")

# ---------------------------overview---------------------------
studies = {
    "Study 1 (Embeddings + basic features)": df_emb_basic,
    "Study 2 (Embeddings + extended features)": df_emb_ext,
    "Study 3 (TF-IDF + extended features)": df_tfidf
}


# ---------- Overview ----------
st.header("Overview")
st.markdown("""
This study evaluates sampling strategies and feature configurations for review rating prediction.

Focus:
- Improve class balance
- Detect critical reviews (1–2 stars)
""")

combined = []
for name, df in studies.items():
    temp = df.copy()
    temp['study'] = name
    combined.append(temp)
combined_df = pd.concat(combined)

# ---------- Global Comparison ----------
st.header("Global Metrics Comparison Across Studies")

metrics = ['accuracy', 'macro_f1', 'rmse']
titles = ['Accuracy', 'Macro F1', 'RMSE']

cols = st.columns(4)  # 3 plots + 1 legend

handles = None
labels = None

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    fig, ax = plt.subplots()

    for study in combined_df['study'].unique():
        subset = combined_df[combined_df['study'] == study]

        line, = ax.plot(
            subset['experiment'],
            subset[metric],
            marker='o',
            label=study
        )

        # Save legend once
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()

    # Best line
    best_value = combined_df[metric].max() if metric != "rmse" else combined_df[metric].min()
    ax.axhline(best_value, linestyle='--', color='red', label='Best')

    ax.set_title(title)
    ax.tick_params(axis='x', rotation=45)

    with cols[idx]:
        st.pyplot(fig)

# Separate legend
with cols[3]:
    st.markdown("### Legend")
    fig_leg, ax_leg = plt.subplots()
    ax_leg.legend(handles, labels, loc='center')
    ax_leg.axis('off')
    st.pyplot(fig_leg)


# =====================================
# 1. Sampling Impact
# =====================================
st.markdown("""
### ⚖️ Impact of Sampling Strategies

- Baseline consistently performs best  
- Undersampling significantly degrades performance   
- Class weights are more effective than sampling
- Still Strong bias toward predicting class 5
- The mid-range classes remain difficult to learn
- Most errors are small (±1), but critical errors still exist            

> Sampling does not solve the core problem
""")

# =====================================
# 2. Performance Comparison TF-IDF vs Embeddings
# =====================================
df_emb = df_emb_ext[df_emb_ext["experiment"] == "baseline"]
df_tfidf = df_tfidf[df_tfidf["experiment"] == "baseline"]

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

def compute_error_distance(cm):
    total_error = 0
    total_samples = cm.sum()

    for i in range(len(cm)):
        for j in range(len(cm)):
            total_error += abs(i - j) * cm[i][j]

    return total_error / total_samples

st.header("📏 Error Analysis")

# --- Metrics berechnen ---
err_emb = compute_error_distance(cm_emb)
err_tfidf = compute_error_distance(cm_tfidf)

emb_fn_critical = cm_emb[0:2, 3:5].sum()
tfidf_fn_critical = cm_tfidf[0:2, 3:5].sum()

boundary_emb = cm_emb[3, 4] + cm_emb[4, 3]
boundary_tfidf = cm_tfidf[3, 4] + cm_tfidf[4, 3]

# --- Tabelle 1 ---
error_table = pd.DataFrame({
    "Error Distance": [err_emb, err_tfidf],
    "Critical False Negatives": [emb_fn_critical, tfidf_fn_critical],
    "Boundary Errors": [boundary_emb, boundary_tfidf]
}, index=["Embeddings", "TF-IDF"])

st.subheader("Error Summary")
st.dataframe(error_table)

# Top Errors
#Top Errors (Largest Deviations)
errors_emb = []
errors_tfidf = []
for i in range(5):
    for j in range(5):
        if i != j:
            errors_emb.append(((i+1, j+1), abs(i-j), cm_emb[i,j]))
            errors_tfidf.append(((i+1, j+1), abs(i-j), cm_tfidf[i,j]))

errors_sorted_emb = sorted(errors_emb, key=lambda x: (x[1], x[2]), reverse=True)
errors_sorted_tfidf = sorted(errors_tfidf, key=lambda x: (x[1], x[2]), reverse=True)


top_errors_emb = errors_sorted_emb[:5]
top_errors_tfidf = errors_sorted_tfidf[:5]

st.subheader("Top Errors")

def build_error_df(errors):
    return pd.DataFrame([
        {
            "True": t,
            "Pred": p,
            "Distance": d,
            "Count": c
        }
        for (t, p), d, c in errors
    ])

df_top_emb = build_error_df(top_errors_emb)
df_top_tfidf = build_error_df(top_errors_tfidf)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Embeddings")
    st.dataframe(df_top_emb, use_container_width=True)

with col2:
    st.markdown("### TF-IDF")
    st.dataframe(df_top_tfidf, use_container_width=True)


st.markdown("""
### 🧠 Key Insight

- TF-IDF produces significantly larger prediction errors  
- Embeddings tend to make *nearby mistakes*  

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
- TF-IDF and embeddings capture different signals  

### Interpretation

- TF-IDF: 
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