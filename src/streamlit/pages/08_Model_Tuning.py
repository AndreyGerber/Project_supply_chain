import streamlit as st
import pandas as pd

df = pd.read_csv("data/experiments/full_study.csv")

st.title("📊 Experiment Comparison")

st.subheader("Mean Metrics")
st.dataframe(df.groupby("experiment").mean())

st.subheader("Accuracy")
st.bar_chart(df.groupby("experiment")["accuracy"].mean())

st.subheader("Macro F1")
st.bar_chart(df.groupby("experiment")["f1_macro"].mean())

st.subheader("MAE (ordinal)")
st.bar_chart(df.groupby("experiment")["mae"].mean())

st.subheader("Kappa (ordinal agreement)")
st.bar_chart(df.groupby("experiment")["kappa"].mean())

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

st.title("🔍 Confusion Matrix Analysis")

df = pd.read_csv("data/experiments/predictions.csv")

experiments = df["experiment"].unique()

exp_a = st.selectbox("Select Experiment A", experiments, index=2)
exp_b = st.selectbox("Select Experiment B", experiments, index=3)

df_a = df[df["experiment"] == exp_a]
df_b = df[df["experiment"] == exp_b]

labels = sorted(df["y_true"].unique())

cm_a = confusion_matrix(df_a["y_true"], df_a["y_pred"], labels=labels)
cm_b = confusion_matrix(df_b["y_true"], df_b["y_pred"], labels=labels)

# =========================================
# Plot helper
# =========================================
def plot_cm(cm, title):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    st.pyplot(fig)

# =========================================
# Show matrices
# =========================================
st.subheader(f"{exp_a} Confusion Matrix")
plot_cm(cm_a, exp_a)

st.subheader(f"{exp_b} Confusion Matrix")
plot_cm(cm_b, exp_b)

# =========================================
# Difference
# =========================================
diff = cm_b - cm_a

st.subheader(f"Difference ({exp_b} - {exp_a})")
plot_cm(diff, "Difference Matrix")

# =========================================
# Normalized Error Distance
# =========================================
def avg_distance(cm):
    total = 0
    count = 0
    for i in range(len(cm)):
        for j in range(len(cm)):
            total += abs(i - j) * cm[i, j]
            count += cm[i, j]
    return total / count

dist_a = avg_distance(cm_a)
dist_b = avg_distance(cm_b)

st.subheader("📏 Average Error Distance")

st.write(f"{exp_a}: {dist_a:.3f}")
st.write(f"{exp_b}: {dist_b:.3f}")

#================================
# Comparison of Models
# improvement_dashboard
#========================================
import streamlit as st
import pandas as pd

from src.analysis.improvement_analysis import (
    compute_global_improvement,
    compute_classwise_improvement,
    compute_error_distance
)

st.title("🚀 Model Improvement Analysis")

df_results = pd.read_csv("data/experiments/full_study.csv")
df_pred = pd.read_csv("data/experiments/predictions.csv")

# =========================================
# GLOBAL IMPROVEMENT
# =========================================
st.header("📊 Global Improvement (Exp4 vs Exp3)")

improvement = compute_global_improvement(df_results)

for metric, value in improvement.items():
    st.metric(label=metric, value=f"{value:.4f}")

# =========================================
# CLASS-WISE
# =========================================
st.header("📈 Class-wise Improvement")

df_class = compute_classwise_improvement(df_pred)
st.dataframe(df_class)

# =========================================
# ERROR DISTANCE
# =========================================
st.header("📏 Error Distance (Ordinal Quality)")

dist = compute_error_distance(df_pred)
st.bar_chart(dist)