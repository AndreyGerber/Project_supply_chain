import streamlit as st
import pandas as pd
import scipy.sparse as sp
import plotly.express as px
import sys
import os

# --- 1. Pfad-Korrektur für den Import aus dem übergeordneten Ordner ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from utils import preprocess_text_full
except ImportError:
    st.error("❌ 'utils.py' not found. Please ensure it's in the parent directory.")

# --- 2. Seite Setup ---
st.title("🚀 Live Model Demo")
st.write("Type a review below to see how the model classifies it!")

# --- 3. Check für Modell & Vectorizer im Session State ---
if 'final_model' in st.session_state and 'tfidf_data' in st.session_state:
    model = st.session_state['final_model']
    tfidf = st.session_state['tfidf_data']['vectorizer']
    
    # --- 4. Eingabe-Formular ---
    with st.form("prediction_form"):
        user_text = st.text_area("Enter Review Text:", placeholder="e.g., I am very happy with this purchase, it works great!")
        is_verified = st.checkbox("Is the user verified?", value=True)
        submit = st.form_submit_button("Classify Review")

    if submit:
        if user_text.strip() == "":
            st.warning("Please enter some text first!")
        else:
            # --- 5. Preprocessing & Vektorisierung ---
            with st.spinner("Analyzing text..."):
                # Reinigung durch die importierte Funktion
                cleaned_text = preprocess_text_full(user_text) 
                
                # In TF-IDF Vektor umwandeln
                text_vector = tfidf.transform([cleaned_text])
                
                # 'Verified' Status als Sparse-Matrix vorbereiten
                verified_val = 1 if is_verified else 0
                verified_vector = sp.csr_matrix([[verified_val]])
                
                # Features zusammenführen (identisch zum Training!)
                final_input = sp.hstack((text_vector, verified_vector))
                
                # --- 6. Vorhersage & Wahrscheinlichkeiten ---
                prediction = model.predict(final_input)[0]
                probabilities = model.predict_proba(final_input)[0]
                classes = model.classes_

            # --- 7. Ergebnisanzeige ---
            st.divider()
            
            # Farbe je nach Ergebnis wählen
            color = "#00CC96" if "High" in prediction else "#AB63FA" if "Mid" in prediction else "#EF553B"
            st.subheader(f"Result: :{color}[{prediction}]")

            # Balkendiagramm für die Konfidenz
            prob_df = pd.DataFrame({
                'Rating Group': classes,
                'Probability': probabilities
            })
            
            fig_prob = px.bar(
                prob_df, 
                x='Rating Group', 
                y='Probability',
                title="Model Confidence per Class",
                color='Rating Group',
                color_discrete_map={"High (5 ⭐)": "#00CC96", "Mid (3-4 ⭐)": "#AB63FA", "Low (1-2 ⭐)": "#EF553B"},
                text_auto='.2%' # Zeigt Prozentwerte auf den Balken
            )
            
            fig_prob.update_layout(yaxis_range=[0, 1]) # Skala von 0 bis 100%
            st.plotly_chart(fig_prob, use_container_width=True)

            # Detail-Info für den Nutzer
            st.info(f"**Cleaned text used for prediction:** *{cleaned_text if cleaned_text else '[Text was empty after cleaning]'}*")

else:
    # Falls der User direkt auf die Demo-Seite springt
    st.warning("⚠️ **Model not found!**")
    st.info("Please go to the **'03 Modeling'** page first, train the model, and then come back here.")
    
    if st.button("Go to Modeling Page"):
        st.switch_page("pages/03_Modeling.py") # Falls dein Dateiname so lautet
