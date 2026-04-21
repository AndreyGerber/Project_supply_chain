import streamlit as st
import pandas as pd
import scipy.sparse as sp
from utils import preprocess_text_full
st.title("🚀 Live Model Demo")
st.write("Type a review below to see how the model classifies it!")

# --- 1. Prüfen, ob Modell & Vectorizer vorhanden sind ---
if 'final_model' in st.session_state and 'tfidf_data' in st.session_state:
    model = st.session_state['final_model']
    tfidf = st.session_state['tfidf_data']['vectorizer']
    
    # --- 2. Eingabe-Bereich ---
    with st.form("prediction_form"):
        user_text = st.text_area("Enter Review Text:", placeholder="e.g., This product is amazing and works perfectly!")
        is_verified = st.checkbox("Is the user verified?", value=True)
        submit = st.form_submit_button("Classify Review")

    if submit and user_text:
        # --- 3. Preprocessing (Muss exakt wie im Training sein!) ---
        # Wir nutzen deine zuvor definierte Funktion
        cleaned_text = preprocess_text_full(user_text) 
        
        # --- 4. Vektorisierung ---
        # Text in TF-IDF umwandeln
        text_vector = tfidf.transform([cleaned_text])
        
        # 'Verified' Status hinzufügen (als Sparse Matrix)
        verified_val = 1 if is_verified else 0
        verified_vector = sp.csr_matrix([[verified_val]])
        
        # Features kombinieren (hstack)
        final_input = sp.hstack((text_vector, verified_vector))
        
        # --- 5. Vorhersage ---
        prediction = model.predict(final_input)[0]
        probabilities = model.predict_proba(final_input)[0]
        classes = model.classes_
        
        # --- 6. Ergebnis-Anzeige ---
        st.divider()
        st.subheader(f"Result: **{prediction}**")
        
        # Wahrscheinlichkeiten als Balkendiagramm zeigen
        prob_df = pd.DataFrame({
            'Rating': classes,
            'Probability': probabilities
        })
        import plotly.express as px
        fig_prob = px.bar(prob_df, x='Rating', y='Probability', 
                          title="Prediction Confidence",
                          color='Rating',
                          range_y=[0, 1])
        st.plotly_chart(fig_prob, use_container_width=True)

else:
    st.warning("⚠️ No trained model found. Please run the **Modeling** page first to train the model!")
