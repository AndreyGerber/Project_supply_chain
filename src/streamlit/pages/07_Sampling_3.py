import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast

st.set_page_config(layout="wide")

st.title("📊 Executive Dashboard: Sampling Strategy Evaluation")

# =====================
# Helper Functions
# =====================

def parse_confusion_matrix(cm_str):
    return np.array(ast.literal_eval(cm_str))


def compute_per_class_metrics(cm):
    precision, recall, f1 = [], [], []

    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        p = tp / (tp + fp) if (tp + fp) else 0
        r = tp / (tp + fn) if (tp + fn) else 0
        f = 2 * p * r / (p + r) if (p + r) else 0

        precision.append(p)
        recall.append(r)
        f1.append(f)

    return np.array(precision), np.array(recall), np.array(f1)


def confusion_distance(cm):
    dist, total = 0, 0
    for i in range(5):
        for j in range(5):
            dist += abs(i - j) * cm[i, j]
            total += cm[i, j]
    return dist / total


def plot_bar(df, metric, title):
    fig, ax = plt.subplots()
    ax.bar(df['experiment'], df[metric])
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df['experiment'], rotation=45)
    st.pyplot(fig)


def plot_heatmap(cm):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[1,2,3,4,5],
                yticklabels=[1,2,3,4,5], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)


def plot_per_class(values, title):
    fig, ax = plt.subplots()
    ax.bar([1,2,3,4,5], values)
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    st.pyplot(fig)


# =====================
# Load Data
# =====================

df1 = pd.read_csv("reports/sampling_comparison.csv")
df2 = pd.read_csv("reports/sampling_comparison_ETL.csv")
df3 = pd.read_csv("reports/sampling_comparison_TF-idf.csv")

studies = {
    "Study 1 (Embeddings + basic)": df1,
    "Study 2 (Embeddings + extended)": df2,
    "Study 3 (TF-IDF + extended)": df3
}

all_df = pd.concat([
    df1.assign(study="S1"),
    df2.assign(study="S2"),
    df3.assign(study="S3")
])

# =====================
# Executive Summary
# =====================

st.header("🏆 Executive Summary")

best_row = all_df.loc[all_df['macro_f1'].idxmax()]

col1, col2, col3, col4 = st.columns(4)

col1.metric("Best Macro F1", f"{best_row['macro_f1']:.3f}", best_row['experiment'])
col2.metric("Best Accuracy", f"{all_df['accuracy'].max():.3f}")
col3.metric("Best RMSE", f"{all_df['rmse'].min():.3f}")

cd_values = all_df.apply(lambda r: confusion_distance(parse_confusion_matrix(r['confusion_matrix'])), axis=1)
col4.metric("Best Confusion Distance", f"{cd_values.min():.3f}")

# =====================
# Key Insight Cards
# =====================

st.subheader("Key Insights")
st.markdown("""
- 📌 Embeddings significantly outperform TF-IDF
- 📌 Extended features improve class separation
- 📌 Class weighting > sampling strategies
- 📌 Major weakness: overprediction of class 5
- 📌 Critical reviews (1–2 stars) still under-detected
""")

# =====================
# Global Comparison
# =====================

st.header("📊 Cross-Study Comparison")

for metric in ['accuracy', 'macro_f1', 'rmse']:
    fig, ax = plt.subplots()

    for name, df in studies.items():
        ax.plot(df['experiment'], df[metric], marker='o', label=name)

    ax.set_title(metric.upper())
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

# =====================
# Study Deep Dive
# =====================

st.header("🔬 Deep Dive Analysis")

study_name = st.selectbox("Select Study", list(studies.keys()))
df = studies[study_name]

best_exp = df.loc[df['macro_f1'].idxmax()]['experiment']
st.success(f"Best config in this study: {best_exp}")

plot_bar(df, 'accuracy', "Accuracy")
plot_bar(df, 'macro_f1', "Macro F1")
plot_bar(df, 'rmse', "RMSE")

# =====================
# Confusion Matrix
# =====================

st.subheader("Confusion Matrix")
exp = st.selectbox("Experiment", df['experiment'])
cm = parse_confusion_matrix(df[df['experiment'] == exp]['confusion_matrix'].values[0])

plot_heatmap(cm)

# =====================
# Per-Class Metrics
# =====================

precision, recall, f1 = compute_per_class_metrics(cm)

col1, col2, col3 = st.columns(3)
with col1:
    plot_per_class(precision, "Precision per Class")
with col2:
    plot_per_class(recall, "Recall per Class")
with col3:
    plot_per_class(f1, "F1 per Class")

# =====================
# Critical KPI Section
# =====================

st.header("⚠️ Critical Performance KPIs")

cd = confusion_distance(cm)
fn_critical = cm[0:2, 3:5].sum()
boundary = cm[3,4] + cm[4,3]
recall_low = recall[:2].mean()

k1, k2, k3 = st.columns(3)

k1.metric("Confusion Distance", f"{cd:.3f}")
k2.metric("Critical FN (1–2 → 4–5)", int(fn_critical))
k3.metric("Boundary Errors (4↔5)", int(boundary))

st.metric("Low-Class Recall (1–2)", f"{recall_low:.3f}")

# =====================
# Top Errors
# =====================

st.subheader("Top Misclassifications")
errors = []
for i in range(5):
    for j in range(5):
        if i != j:
            errors.append((abs(i-j), cm[i,j], i+1, j+1))

errors = sorted(errors, key=lambda x: (x[0], x[1]), reverse=True)

for dist, count, true, pred in errors[:6]:
    st.write(f"True {true} → Pred {pred} | Distance {dist} | Count {count}")

# =====================
# Recommendation Engine
# =====================

st.header("📌 Recommendation")

st.markdown("""
**Production-Ready Setup:**
- Embeddings + extended features
- Class weights instead of sampling
- Optimize hyperparameters (XGBoost)
- Use ordinal-aware loss functions

**Primary Objective:**
Improve recall for classes 1–2 while reducing 4↔5 confusion.
""")
