import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Balancing as Hypothesis Test",
    layout="wide"
)

# =========================================================
# HEADER
# =========================================================

st.title("⚖️ Balancing as a Hypothesis-Driven Intervention")

st.markdown(
    """
    After identifying strong class imbalance and weak mid-class performance,
    balancing strategies were evaluated as an initial intervention to test whether
    distributional imbalance was a major source of classification error.
    """
)

st.markdown("---")

# =========================================================
# SECTION 1 — OBSERVATION
# =========================================================

st.header("1️⃣ Observed Problem")

col1, col2 = st.columns([1.2, 1])

# ---------------------------------------------------------
# Distribution Chart
# ---------------------------------------------------------

with col1:

    original_distribution = pd.DataFrame({
        "Class Group": [
            "Low (1-2⭐)",
            "Mid (3-4⭐)",
            "High (5⭐)"
        ],
        "Samples": [
            434,
            874,
            4344
        ]
    })

    fig_dist = px.bar(
        original_distribution,
        x="Class Group",
        y="Samples",
        text="Samples",
        height=400
    )

    fig_dist.update_traces(
        textposition="outside"
    )

    fig_dist.update_layout(
        title="Original Training Distribution",
        yaxis_title="Number of Samples",
        xaxis_title=""
    )

    st.plotly_chart(
        fig_dist,
        use_container_width=True
    )

# ---------------------------------------------------------
# Observation Box
# ---------------------------------------------------------

with col2:

    st.info(
        """
        ### Observed Issues
        
        - Strong dominance of high ratings  
        - Underrepresentation of minority classes  
        - Weak separability of mid-rating classes  
        - Potential bias toward majority predictions  
        
        ---
        
        The observed imbalance raised the question whether
        distributional skewness was a major contributor
        to classification failure.
        """
    )

# =========================================================
# SECTION 2 — HYPOTHESIS
# =========================================================

st.header("2️⃣ Initial Hypothesis")

st.success(
    """
    ### Hypothesis
    
    Class imbalance contributes significantly to weak minority-class performance
    and poor robustness across rating categories.
    """
)

# =========================================================
# SECTION 3 — BALANCING STRATEGY
# =========================================================

st.header("3️⃣ Balancing Strategy")

col1, col2 = st.columns([1.3, 1])

# ---------------------------------------------------------
# Before vs After
# ---------------------------------------------------------

with col1:

    balancing_df = pd.DataFrame({
        "Class Group": [
            "Low (1-2⭐)",
            "Mid (3-4⭐)",
            "High (5⭐)",
            "Low (1-2⭐)",
            "Mid (3-4⭐)",
            "High (5⭐)"
        ],
        "Samples": [
            434,
            874,
            4344,
            3500,
            3500,
            3500
        ],
        "Distribution": [
            "Original",
            "Original",
            "Original",
            "Balanced",
            "Balanced",
            "Balanced"
        ]
    })

    fig_balance = px.bar(
        balancing_df,
        x="Class Group",
        y="Samples",
        color="Distribution",
        barmode="group",
        text="Samples",
        height=450
    )

    fig_balance.update_traces(
        textposition="outside"
    )

    fig_balance.update_layout(
        title="Training Distribution Before vs After Balancing",
        yaxis_title="Number of Samples",
        xaxis_title=""
    )

    st.plotly_chart(
        fig_balance,
        use_container_width=True
    )

# ---------------------------------------------------------
# Strategy Explanation
# ---------------------------------------------------------

with col2:

    st.info(
        """
        ### Applied Strategy
        
        A combined balancing strategy was chosen:
        
        - **Oversampling** for minority classes  
        - **Undersampling** for the dominant majority class  
        
        The combination aimed to:
        
        - improve class parity  
        - reduce majority dominance  
        - avoid excessive duplication  
        - limit information loss
        
        ---
        
        Balancing was applied **only on training data**
        to prevent data leakage.
        """
    )

# =========================================================
# SECTION 4 — PERFORMANCE IMPACT
# =========================================================

st.header("4️⃣ Performance Impact")

# ---------------------------------------------------------
# KPI COMPARISON
# ---------------------------------------------------------

st.subheader("📋 Overall Performance")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(
        "Before Macro-F1",
        "0.472"
    )

with kpi2:
    st.metric(
        "Before RMSE",
        "0.711"
    )

with kpi3:
    st.metric(
        "After Macro-F1",
        "0.797",
        delta="+0.325"
    )

with kpi4:
    st.metric(
        "After RMSE",
        "0.514",
        delta="-0.197"
    )

st.markdown("---")

# ---------------------------------------------------------
# F1 COMPARISON
# ---------------------------------------------------------

col1, col2 = st.columns(2)

# =========================================================
# LEFT — PREVIOUS CLASS PERFORMANCE
# =========================================================

with col1:

    st.subheader("Before Balancing — F1 per Class")

    before_df = pd.DataFrame({
        "Class": ["1", "2", "3", "4", "5"],
        "F1 Score": [
            0.65,
            0.31,
            0.34,
            0.42,
            0.90
        ]
    })

    fig_before = px.bar(
        before_df,
        x="Class",
        y="F1 Score",
        text="F1 Score",
        height=450
    )

    fig_before.update_traces(
        texttemplate='%{text:.2f}',
        textposition='outside'
    )

    fig_before.update_layout(
        yaxis_range=[0, 1],
        xaxis_title="Rating Class",
        yaxis_title="F1 Score"
    )

    st.plotly_chart(
        fig_before,
        use_container_width=True
    )

# =========================================================
# RIGHT — GROUPED PERFORMANCE
# =========================================================

with col2:

    st.subheader("After Balancing — Grouped Class Performance")

    grouped_df = pd.DataFrame({
        "Group": [
            "Low (1-2⭐)",
            "Mid (3-4⭐)",
            "High (5⭐)"
        ],
        "F1 Score": [
            0.888,
            0.583,
            0.918
        ]
    })

    fig_grouped = px.bar(
        grouped_df,
        x="Group",
        y="F1 Score",
        text="F1 Score",
        height=450
    )

    fig_grouped.update_traces(
        texttemplate='%{text:.3f}',
        textposition='outside'
    )

    fig_grouped.update_layout(
        yaxis_range=[0, 1],
        xaxis_title="Grouped Classes",
        yaxis_title="F1 Score"
    )

    st.plotly_chart(
        fig_grouped,
        use_container_width=True
    )

# =========================================================
# SECTION 5 — SCIENTIFIC CONCLUSION
# =========================================================

st.header("5️⃣ Scientific Interpretation")

st.info(
    """
    ### Key Insight
    
    Balancing improved overall class consistency and reduced majority-class dominance.
    
    The substantial increase in Macro-F1 indicates that the model became more robust across underrepresented classes.
    
    However, persistent weaknesses in the mid-rating groups suggest that class imbalance alone could not fully explain the observed classification errors.
    
    ---
    
    ### Scientific Conclusion
    
    The results indicate that imbalance amplifies the problem,
    but semantic overlap between neighboring rating classes remains the dominant challenge.
    
    This motivated a deeper investigation into:
    
    - alternative balancing strategies  
    - feature representations  
    - semantic separability between neighboring classes
    """
)