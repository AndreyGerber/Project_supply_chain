import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

st.set_page_config(layout="wide")

st.title("📊 Sampling Strategy Evaluation for Review Classification")

# ---------- Helper Functions ----------

def parse_confusion_matrix(cm_str):
    return np.array(ast.literal_eval(cm_str))


def compute_per_class_metrics(cm):
    num_classes = cm.shape[0]
    precision = []
    recall = []
    f1 = []

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
    ax.set_xticklabels(df['experiment'], rotation=45)
    st.pyplot(fig)


def plot_per_class(metric_values, title):
    fig, ax = plt.subplots()
    classes = list(range(1, len(metric_values) + 1))
    ax.bar(classes, metric_values)
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    st.pyplot(fig)

# ---------- Load Data ----------

file1 = "reports/sampling_comparison.csv"
file2 = "reports/sampling_comparison_ETL.csv"
file3 = "reports/sampling_comparison_TF-idf.csv"

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)

# ---------- Overview ----------
st.header("Overview")
st.write("This study evaluates sampling strategies and feature configurations for review rating prediction.")

# ---------- Global Metrics ----------
st.header("Global Metrics Comparison")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Study 1")
    plot_bar(df1, 'accuracy', "Accuracy")
    plot_bar(df1, 'macro_f1', "Macro F1")
    plot_bar(df1, 'rmse', "RMSE")

with col2:
    st.subheader("Study 2")
    plot_bar(df2, 'accuracy', "Accuracy")
    plot_bar(df2, 'macro_f1', "Macro F1")
    plot_bar(df2, 'rmse', "RMSE")

with col3:
    st.subheader("Study 3")
    plot_bar(df3, 'accuracy', "Accuracy")
    plot_bar(df3, 'macro_f1', "Macro F1")
    plot_bar(df3, 'rmse', "RMSE")

# ---------- Confusion Matrix Analysis ----------
st.header("Confusion Matrix Analysis")

selected_study = st.selectbox("Select Study", ["Study 1", "Study 2", "Study 3"])

if selected_study == "Study 1":
    df = df1
elif selected_study == "Study 2":
    df = df2
else:
    df = df3

selected_exp = st.selectbox("Select Experiment", df['experiment'])
row = df[df['experiment'] == selected_exp].iloc[0]
cm = parse_confusion_matrix(row['confusion_matrix'])

st.subheader("Confusion Matrix")
st.write(cm)

# ---------- Per-Class Metrics ----------
st.header("Per-Class Metrics")

precision, recall, f1 = compute_per_class_metrics(cm)

col1, col2, col3 = st.columns(3)

with col1:
    plot_per_class(precision, "Precision per Class")

with col2:
    plot_per_class(recall, "Recall per Class")

with col3:
    plot_per_class(f1, "F1 Score per Class")

# ---------- Error Analysis ----------
st.header("Error Analysis")

# Confusion Distance
cd = confusion_distance(cm)
st.metric("Confusion Distance", round(cd, 3))

# False Negatives (1-2 predicted as 4-5)
fn_critical = cm[0:2, 3:5].sum()
st.metric("Critical False Negatives (1-2 → 4-5)", int(fn_critical))

# Boundary Errors (4 vs 5)
boundary = cm[3, 4] + cm[4, 3]
st.metric("Boundary Errors (4 ↔ 5)", int(boundary))

# ---------- Key Insights ----------
st.header("Key Insights")

st.markdown("""
- Embeddings outperform TF-IDF significantly
- Extended features (sentiment, negation) improve performance
- Class weights outperform sampling strategies
- Major errors occur between 4 and 5 star ratings
- Detecting critical reviews remains challenging
""")

# ---------- Recommendations ----------
st.header("Recommendations")

st.markdown("""
- Use embeddings with extended features
- Prefer class weights over sampling
- Apply hyperparameter tuning
- Focus on recall for low ratings (1–2)
""")
