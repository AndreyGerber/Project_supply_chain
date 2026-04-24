import streamlit as st
import pandas as pd
import scipy.sparse as sp
import plotly.express as px
import sys
import os

# --- 1. Path Correction for Imports ---
# We add the project root to the path so Python can find 'src.streamlit.pages._utils'
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if root_path not in sys.path:
    sys.path.append(root_path)

try:
    # Absolute import starting from project root
    from src.streamlit.pages._utils import preprocess_text_full
except ImportError:
    # Fallback: local import within the same directory
    from _utils import preprocess_text_full

# --- 2. UI SETUP ---
st.title("🚀 Live Model Demo")
st.info("💡 **Note:** This model was trained on **English** text. Please enter your reviews in English.")
st.write("Type a review below to see how the model classifies it!")

# Check if Model & Vectorizer are available in Session State
if 'final_model' in st.session_state and 'tfidf_data' in st.session_state:
    model = st.session_state['final_model']
    tfidf = st.session_state['tfidf_data']['vectorizer']
    
    # Input Form
    with st.form("prediction_form"):
        user_text = st.text_area("Enter Review Text:", 
                                 placeholder="e.g., This was a great experience, I love it!")
        is_verified = st.checkbox("Is the user verified?", value=True)
        submit = st.form_submit_button("Classify Review")

    if submit:
        if not user_text.strip():
            st.warning("Please enter some text first!")
        else:
            with st.spinner("Analyzing text..."):
                # 1. Preprocessing (must be identical to training)
                cleaned_text = preprocess_text_full(user_text) 
                
                # 2. Vectorization (Text -> TF-IDF)
                text_vector = tfidf.transform([cleaned_text])
                
                # 3. Add Verified Feature (Sparse Matrix Format)
                verified_val = 1 if is_verified else 0
                verified_vector = sp.csr_matrix([[verified_val]])
                
                # 4. Combine Features (Text + Verified)
                final_input = sp.hstack((text_vector, verified_vector))
                
                # 5. Prediction & Probabilities
                prediction = model.predict(final_input)[0]
                probabilities = model.predict_proba(final_input)[0]
                classes = model.classes_

            # --- RESULTS DISPLAY ---
            st.divider()
            
            # Define color based on result
            color = "#00CC96" if "High" in prediction else "#AB63FA" if "Mid" in prediction else "#EF553B"
            st.subheader(f"Result: :{color}[{prediction}]")

            # Probability Chart
            prob_df = pd.DataFrame({
                'Rating Group': classes,
                'Probability': probabilities
            })
            
            fig_prob = px.bar(
                prob_df, 
                x='Rating Group', 
                y='Probability',
                title="Model Confidence",
                color='Rating Group',
                color_discrete_map={"High (5 ⭐)": "#00CC96", "Mid (3-4 ⭐)": "#AB63FA", "Low (1-2 ⭐)": "#EF553B"},
                text_auto='.2%'
            )
            
            fig_prob.update_layout(yaxis_range=[0, 1]) # Fix scale 0-100%
            st.plotly_chart(fig_prob, use_container_width=True)

            st.info(f"**Processed text used for prediction:** *{cleaned_text if cleaned_text else '[Text was empty after cleaning]'}*")

else:
    st.warning("⚠️ **No model found!**")
    st.info("Please go to the **'03 Modeling'** page first to train your model.")
