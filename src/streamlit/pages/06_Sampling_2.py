import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast

st.set_page_config(layout="wide")

st.title("📊 Sampling Strategy Evaluation for Review Classification")

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

# ---------- Load Data ----------

file1 = "reports/sampling_comparison.csv"
file2 = "reports/sampling_comparison_ETL.csv"
file3 = "reports/sampling_comparison_TF-idf.csv"

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)

studies = {
    "Study 1 (Embeddings + basic features)": df1,
    "Study 2 (Embeddings + extended features)": df2,
    "Study 3 (TF-IDF + extended features)": df3
}

# ---------- Overview ----------
st.header("Overview")
st.markdown("""
This study evaluates sampling strategies and feature configurations for review rating prediction.

Focus:
- Improve class balance
- Detect critical reviews (1–2 stars)
""")

# ---------- Global Comparison ----------
st.header("Global Metrics Comparison Across Studies")

combined = []
for name, df in studies.items():
    temp = df.copy()
    temp['study'] = name
    combined.append(temp)
combined_df = pd.concat(combined)

for metric in ['accuracy', 'macro_f1', 'rmse']:
    fig, ax = plt.subplots()
    for study in combined_df['study'].unique():
        subset = combined_df[combined_df['study'] == study]
        ax.plot(subset['experiment'], subset[metric], marker='o', label=study)

    ax.set_title(metric.upper())
    ax.set_xticklabels(subset['experiment'], rotation=45)
    ax.legend()
    st.pyplot(fig)

# ---------- Study Detail ----------
st.header("Detailed Study Analysis")

selected_study = st.selectbox("Select Study", list(studies.keys()))
df = studies[selected_study]

plot_bar(df, 'accuracy', "Accuracy")
plot_bar(df, 'macro_f1', "Macro F1")
plot_bar(df, 'rmse', "RMSE")

# ---------- Confusion Matrix ----------
st.subheader("Confusion Matrix Analysis")
selected_exp = st.selectbox("Select Experiment", df['experiment'])
row = df[df['experiment'] == selected_exp].iloc[0]
cm = parse_confusion_matrix(row['confusion_matrix'])

plot_confusion_matrix(cm)

# ---------- Per-Class Metrics ----------
st.subheader("Per-Class Metrics")
precision, recall, f1 = compute_per_class_metrics(cm)

col1, col2, col3 = st.columns(3)
with col1:
    plot_per_class(precision, "Precision")
with col2:
    plot_per_class(recall, "Recall")
with col3:
    plot_per_class(f1, "F1 Score")

# ---------- Advanced Error Analysis ----------
st.header("Advanced Error Analysis")

cd = confusion_distance(cm)
st.metric("Confusion Distance", round(cd, 3))

# Critical FN (1–2 → 4–5)
fn_critical = cm[0:2, 3:5].sum()
st.metric("Critical False Negatives (1-2 → 4-5)", int(fn_critical))

# Boundary Errors (4 vs 5)
boundary = cm[3, 4] + cm[4, 3]
st.metric("Boundary Errors (4 ↔ 5)", int(boundary))

# Top Errors
st.subheader("Top Errors (Largest Deviations)")
errors = []
for i in range(5):
    for j in range(5):
        if i != j:
            errors.append(((i+1, j+1), abs(i-j), cm[i,j]))

errors_sorted = sorted(errors, key=lambda x: (x[1], x[2]), reverse=True)

top_errors = errors_sorted[:5]

for (true, pred), dist, count in top_errors:
    st.write(f"True {true} → Pred {pred} | Distance: {dist} | Count: {count}")

# ---------- Critical Class Recall ----------
st.header("Critical Class Performance")

recall_1_2 = np.mean(recall[0:2])
st.metric("Avg Recall (Class 1 & 2)", round(recall_1_2, 3))

# ---------- Insights ----------
st.header("Key Insights")
st.markdown("""
- Embeddings clearly outperform TF-IDF
- Extended features improve class separation
- Class weights are more effective than sampling
- Strong bias toward predicting class 5
- Most errors are small (±1), but critical errors still exist
""")

# ---------- Recommendations ----------
st.header("Recommendations")
st.markdown("""
- Use embeddings with extended features
- Focus on recall for low ratings (1–2)
- Apply hyperparameter tuning
- Consider ordinal loss functions
""")
