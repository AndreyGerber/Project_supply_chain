import streamlit as st

# 1. Konfiguration der Seite
st.set_page_config(
    page_title="Introduction to Supply Chain Analytics",
    page_icon="📊",
    layout="wide"
)

# 2. Titel und Einleitung
st.title("🤖 Supply Chain Analytics")
st.subheader("From Data Scraping to Star Predictions Using Machine Learning")

st.markdown("""
---
### 📝 Project Overview
This project focuses on analyzing customer comments to automatically predict the corresponding **star rating**. 
This represents a classic **Natural Language Processing (NLP)** and **Classification** challenge.

<br>

### 🎯 Project Objectives
*   **Pattern Recognition:** Identify which words correlate most strongly with positive or negative reviews. Find the negative and positve review patterns. 
*   **Automation:** Train a model to accurately detect the sentiment (stars) of any given text.
*   **Interaction:** Real-time testing of the model using your-comments.

<br>

---
### 🚀 Workflow & Navigation
Use the **sidebar on the left** to navigate through the different phases of the project:
""", unsafe_allow_html=True)

# 3. Project Phases visualized as Columns
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.info("**Phase 1**")
    st.write("📊 **Data Exploration**")
    st.caption("Upload raw data, generate initial statistics, and visualize the dataset distribution.")

with col2:
    st.info("**Phase 2**")
    st.write("🧹 **Preprocessing**")
    st.caption("Text cleaning, stopword removal, and analyzing correlations between features.")

with col3:
    st.info("**Phase 3**")
    st.write("⚙️ **Modeling**")
    st.caption("Comparison of various ML algorithms regarding accuracy, performance, and training time.")

with col4:
    st.info("**Phase 4**")
    st.write("✨ **Live Demo**")
    st.caption("Interactive Prediction: Enter your own comment and let the AI predict the rating!")

st.markdown("---")

st.markdown("<div style='margin-bottom: 50px;'></div>", unsafe_allow_html=True)

font_size = "20px"
st.markdown(
    f"""
    <div style="
        background-color: #d4edda; 
        color: #155724; 
        padding: 15px; 
        border-radius: 5px; 
        border: 1px solid #c3e6cb;
        font-size: {font_size};
        display: flex;
        align-items: center;
    ">
        💡 <span style="margin-left: 10px;">
            <b>Ready to Start:</b> then let's move to Data Exploration.
        </span>
    </div>
    """, 
    unsafe_allow_html=True
)