import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(layout="wide")

st.title("🧠 From False Confidence to Real Problem Understanding")

# =====================================
# 1. Binary vs Multiclass Comparison
# =====================================

st.header("📊 Phase 1 vs Phase 2 – Performance Reality Check")

labels = ["Binary (early)", "TF-IDF Baseline", "Embedding Baseline"]
f1_scores = [0.972, 0.469, 0.684]

fig, ax = plt.subplots()
ax.bar(labels, f1_scores)
ax.set_ylabel("Macro F1")
ax.set_title("Performance Comparison")

st.pyplot(fig)

st.markdown("""
### 🧠 Interpretation

- The binary setup shows extremely high performance  
- Moving to multiclass drastically reduces performance  
- Even strong embeddings cannot fully recover performance  

> High performance in early experiments masked the true complexity of the problem
""")

# =====================================
# 2. Data Leakage Highlight
# =====================================

st.header("⚠️ Critical Finding – Data Leakage")

st.markdown("""
In the initial binary setup, embeddings were computed on the full dataset, introducing data leakage.

### Impact:
- Artificially inflated performance
- Reduced generalization reliability

> This explains part of the overly optimistic results in early experiments
""")

# =====================================
# 3. Class Distribution
# =====================================

st.header("⚖️ Real Class Distribution")

classes = ["1⭐", "2⭐", "3⭐", "4⭐", "5⭐"]
distribution = [17.45, 2.87, 3.43, 8.23, 68.01]

fig, ax = plt.subplots()
ax.bar(classes, distribution)
ax.set_ylabel("Percentage")
ax.set_title("Class Imbalance")

st.pyplot(fig)

st.markdown("""
### 🧠 Insight

- Strong dominance of 5-star reviews (~68%)  
- Very limited representation of mid-range classes  

> The problem is highly imbalanced and structurally difficult
""")

# =====================================
# 4. Confusion Matrix – TF-IDF
# =====================================

st.header("🔍 Error Analysis – TF-IDF Baseline")

cm_tfidf = np.array([
    [222, 161, 51, 20, 1],
    [6, 36, 23, 8, 2],
    [2, 20, 37, 27, 3],
    [0, 10, 17, 87, 101],
    [0, 6, 48, 208, 1511]
])

fig, ax = plt.subplots()
im = ax.imshow(cm_tfidf)

ax.set_xticks(range(5))
ax.set_yticks(range(5))
ax.set_xticklabels(["1", "2", "3", "4", "5"])
ax.set_yticklabels(["1", "2", "3", "4", "5"])

plt.colorbar(im)

st.pyplot(fig)

st.markdown("""
### 🧠 Insight

- Strong confusion between neighboring classes  
- Significant misclassification toward dominant class (5⭐)  

> The model struggles especially with mid-range ratings
""")

# =====================================
# 5. Confusion Matrix – Embeddings
# =====================================

st.header("🔍 Error Analysis – Embedding Baseline")

cm_emb = np.array([
    [337, 85, 29, 4, 0],
    [6, 53, 11, 5, 0],
    [3, 16, 55, 13, 2],
    [2, 4, 20, 142, 47],
    [1, 1, 28, 101, 1642]
])

fig, ax = plt.subplots()
im = ax.imshow(cm_emb)

ax.set_xticks(range(5))
ax.set_yticks(range(5))
ax.set_xticklabels(["1", "2", "3", "4", "5"])
ax.set_yticklabels(["1", "2", "3", "4", "5"])

plt.colorbar(im)

st.pyplot(fig)

st.markdown("""
### 🧠 Insight

- Fewer extreme misclassifications compared to TF-IDF  
- Still significant overlap in mid-range classes  

> Semantic understanding improves stability but does not solve class ambiguity
""")

# =====================================
# 6. Key Takeaways
# =====================================

st.header("🏁 Key Takeaways")

st.markdown("""
### What we learned

- Early results were overly optimistic due to:
  - Simplified problem formulation
  - Data leakage

- Real-world setup reveals:
  - Strong class imbalance  
  - High semantic overlap between classes  

### Core Insight

> The main challenge is not classification itself, but understanding nuanced differences between rating levels.
""")