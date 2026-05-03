import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(layout="wide")

st.title("🚀 Final Modeling – Hybrid Approach")

# =====================================
# Data
# =====================================

data = {
    "Experiment": ["Embeddings + Struct", "GridSearch", "Hybrid", "Ordinal"],
    "Accuracy": [0.939, 0.942, 0.944, 0.236],
    "F1": [0.850, 0.857, 0.860, 0.313],
    "MAE": [0.113, 0.102, 0.101, 1.015],
    "Kappa": [0.872, 0.880, 0.883, 0.155]
}

df = pd.DataFrame(data)

# =====================================
# 1. Performance Overview
# =====================================

st.header("📊 Model Performance Comparison")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    ax.bar(df["Experiment"], df["F1"])
    ax.set_title("Macro F1")
    ax.tick_params(axis='x', rotation=30)
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    ax.bar(df["Experiment"], df["MAE"])
    ax.set_title("Mean Absolute Error")
    ax.tick_params(axis='x', rotation=30)
    st.pyplot(fig)

st.markdown("""
### 🧠 Key Insight

- Hybrid model achieves the best overall performance  
- Improvements are consistent but incremental  
- Ordinal approach fails significantly  

> Model performance is bounded by data representation, not model complexity
""")

# =====================================
# 2. Incremental Gains
# =====================================

st.header("📈 Incremental Improvements")

improvements = pd.DataFrame({
    "Step": ["Baseline → Tuning", "Tuning → Hybrid"],
    "F1 Gain": [0.857 - 0.850, 0.860 - 0.857]
})

fig, ax = plt.subplots()
ax.bar(improvements["Step"], improvements["F1 Gain"])
ax.set_title("F1 Improvements per Step")

st.pyplot(fig)

st.markdown("""
### 🧠 Interpretation

- Hyperparameter tuning provides measurable improvements  
- Hybrid modeling adds further gains  

> Performance improvements become smaller as the model matures
""")

# =====================================
# 3. Why Hybrid Works
# =====================================

st.header("🧠 Why the Hybrid Model Works")

st.markdown("""
The hybrid model combines:

- **TF-IDF features**
  - Strong class separation  
- **Embeddings**
  - Semantic understanding  
- **Structural features**
  - Additional contextual signals  

### Result

- Better balance between precision and generalization  
- Reduced extreme errors  
- Improved robustness across classes  

> Hybrid models leverage complementary feature representations
""")

# =====================================
# 4. Ordinal Failure
# =====================================

st.header("❌ Why Ordinal Modeling Failed")

st.markdown("""
The ordinal approach assumes:

- Ordered relationship between classes  
- Smooth transitions between ratings  

### Observed behavior

- Severe performance degradation  
- High error magnitude (MAE > 1.0)  
- Loss of class discrimination  

### Explanation

- Feature space is not strictly ordinal  
- Semantic overlap between classes breaks ordinal assumptions  

> The rating scale is ordinal, but the feature representation is not
""")

# =====================================
# 5. Limitations
# =====================================

st.header("⚠️ Limitations")

st.markdown("""
### Why perfect classification is unlikely

- Semantic ambiguity between classes (e.g. 3⭐ vs 4⭐)  
- Subjective nature of ratings  
- Noise in user-generated text  

### Observed pattern

- Most errors occur between neighboring classes  
- Extreme errors are reduced but not eliminated  

> The problem has an inherent uncertainty that cannot be fully resolved
""")

# =====================================
# 6. Final Takeaway
# =====================================

st.header("🏁 Final Takeaways")

st.markdown("""
### Core Findings

- Feature representation is the key driver of performance  
- Sampling strategies alone are insufficient  
- Hybrid models provide the best balance  

### Final Conclusion

> The main limitation is not model choice, but the inherent ambiguity of the data.

### Future Work

- Better representation learning (e.g. fine-tuned embeddings)  
- Ordinal-aware loss functions (instead of full ordinal models)  
- Error-aware training strategies  
""")