import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(layout="wide")

st.title("⚖️ Balancing Strategies – Benefits and Trade-offs")

# =====================================
# 1. Original vs Balanced Distribution
# =====================================

st.header("📊 Data Distribution Before vs After Balancing")

labels = ["Low (1-2⭐)", "Mid (3-4⭐)", "High (5⭐)"]

before = [1645, 874, 4344]
after = [3500, 3500, 3500]

x = np.arange(len(labels))

fig, ax = plt.subplots()
ax.bar(x - 0.2, before, width=0.4, label="Original")
ax.bar(x + 0.2, after, width=0.4, label="Balanced")

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_title("Training Data Distribution")
ax.legend()

st.pyplot(fig)

st.markdown("""
### 🧠 Interpretation

- Strong initial imbalance (dominance of high ratings)  
- Balanced dataset created via:
  - Oversampling (Low, Mid)
  - Undersampling (High)  

> Balancing was applied **only on training data** to prevent data leakage
""")

# =====================================
# 2. Model Performance
# =====================================

st.header("📈 Model Performance After Balancing")

st.metric("Accuracy", "0.866")
st.metric("Macro F1", "0.797")

st.markdown("""
### 🧠 Interpretation

- Overall performance is solid  
- Balanced setup improves fairness across classes  
- However, performance does not dramatically improve  

> Balancing alone is not sufficient to solve the problem
""")

# =====================================
# 3. Per-Class Performance
# =====================================

st.header("📊 Per-Class Performance")

classes = ["High (5⭐)", "Low (1-2⭐)", "Mid (3-4⭐)"]
f1_scores = [0.918, 0.888, 0.583]

fig, ax = plt.subplots()
ax.bar(classes, f1_scores)
ax.set_title("F1 Score per Class")

st.pyplot(fig)

st.markdown("""
### 🧠 Insight

- High and Low classes perform well  
- Mid class remains significantly weaker  

> The model struggles with semantically overlapping classes
""")

# =====================================
# 4. Key Trade-off
# =====================================

st.header("⚠️ Trade-off Analysis")

st.markdown("""
### What improved:
- Minority class recall increased  
- More balanced predictions  

### What did NOT improve:
- Mid-class separability  
- Overall robustness  

### Key Insight

> Balancing improves fairness, but introduces trade-offs and does not resolve semantic ambiguity.
""")

# =====================================
# 5. Final Takeaway
# =====================================

st.header("🏁 Key Takeaway")

st.markdown("""
Balancing the dataset was a necessary step to address class imbalance.

However, the results show that:

- The main limitation is not only imbalance  
- The real challenge lies in semantic overlap between classes  

> This motivated a shift towards feature representation analysis in the next phase.
""")