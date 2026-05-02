import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df_res = pd.read_csv("data/experiments/full_study.csv")
    df_pred = pd.read_csv("data/experiments/predictions.csv")
    return df_res, df_pred

df_res, df_pred = load_data()

# Aggregate
df_avg = df_res.groupby("experiment").mean().reset_index()

# =========================
# TITLE
# =========================
st.title("📊 Model Evolution: From Baseline to Hybrid Intelligence")

st.markdown("""
This dashboard shows how the model improves step-by-step through systematic experimentation.
Each experiment represents a targeted improvement in the ML pipeline.
""")

# =========================
# 1. PERFORMANCE EVOLUTION
# =========================
st.header("📈 Performance Evolution")

fig, ax = plt.subplots()

for metric in ["accuracy", "f1_macro", "mae"]:
    ax.plot(df_avg["experiment"], df_avg[metric], marker="o", label=metric)

ax.set_title("Model Improvement Across Experiments")
ax.legend()

st.pyplot(fig)

st.markdown("""
🔍 **Insight:**
- Strong improvement from Exp1 → Exp3
- Exp4 (Ordinal) fails dramatically → requires redesign
""")

# =========================
# 2. ERROR REDUCTION
# =========================
st.header("📉 Error Reduction (MAE)")

fig, ax = plt.subplots()
ax.plot(df_avg["experiment"], df_avg["mae"], marker="o")

ax.set_title("Mean Absolute Error Reduction")

st.pyplot(fig)

st.markdown("""
📌 Lower MAE = better ordinal predictions  
👉 Hybrid model significantly reduces error
""")

# =========================
# 3. PER-CLASS ANALYSIS
# =========================
st.header("🎯 Per-Class Performance")

exp = st.selectbox("Select Experiment", df_pred["experiment"].unique())

df_exp = df_pred[df_pred["experiment"] == exp]

from sklearn.metrics import classification_report

report = classification_report(
    df_exp["y_true"],
    df_exp["y_pred"],
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(report_df)

st.markdown("""
🔍 **Insight:**
- Lower ratings (1–3) are harder to predict
- Model biased toward high ratings
""")

# =========================
# 4. CONFUSION MATRIX
# =========================
st.header("📊 Confusion Matrix")

from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(df_exp["y_true"], df_exp["y_pred"])

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", ax=ax)

st.pyplot(fig)

# =========================
# 5. EXPERIMENT INSIGHTS
# =========================
st.header("🧠 Key Insights")

st.markdown("""
### 🚀 What worked
- Hybrid features (Exp3) deliver best performance
- GridSearch adds consistent improvement

### ⚠️ What failed
- Ordinal weighting collapsed performance
- Likely over-penalizing class differences

### 🎯 Business Impact
- Improved detection of low ratings still needed
- High recall for class 1–2 is critical
""")

# =========================
# 6. NEXT STEPS
# =========================
st.header("🔬 Next Steps")

st.markdown("""
- Calibrated ordinal loss (not naive weights)
- Class-balanced training
- Stacking architecture
- Threshold optimization for low ratings
""")

