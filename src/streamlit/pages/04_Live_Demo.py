import streamlit as st
import pandas as pd
import scipy.sparse as sp
import plotly.express as px

# Import der Funktion aus der versteckten Datei im selben Ordner
try:
    from _utils import preprocess_text_full
except ImportError:
    st.error("❌ Die Datei '_utils.py' wurde im 'pages'-Ordner nicht gefunden.")

# --- UI SETUP ---
st.title("🚀 Live Model Demo")
st.write("Gib unten einen Kommentar ein, um zu sehen, wie das Modell ihn einstuft!")

# Prüfen, ob Modell & Vectorizer aus der Modeling-Seite im Session State sind
if 'final_model' in st.session_state and 'tfidf_data' in st.session_state:
    model = st.session_state['final_model']
    tfidf = st.session_state['tfidf_data']['vectorizer']
    
    # Eingabe-Formular
    with st.form("prediction_form"):
        user_text = st.text_area("Rezensionstext eingeben:", 
                                 placeholder="z.B. This was a great experience, I love it!")
        is_verified = st.checkbox("Ist der Nutzer verifiziert?", value=True)
        submit = st.form_submit_button("Kommentar klassifizieren")

    if submit:
        if not user_text.strip():
            st.warning("Bitte gib zuerst einen Text ein!")
        else:
            with st.spinner("Analyse läuft..."):
                # 1. Vorverarbeitung (identisch zum Training)
                cleaned_text = preprocess_text_full(user_text) 
                
                # 2. Vektorisierung (Text -> TF-IDF)
                text_vector = tfidf.transform([cleaned_text])
                
                # 3. Verified-Feature hinzufügen (Sparse Matrix Format)
                verified_val = 1 if is_verified else 0
                verified_vector = sp.csr_matrix([[verified_val]])
                
                # 4. Features kombinieren (Text + Verified)
                final_input = sp.hstack((text_vector, verified_vector))
                
                # 5. Vorhersage & Wahrscheinlichkeiten
                prediction = model.predict(final_input)[0]
                probabilities = model.predict_proba(final_input)[0]
                classes = model.classes_

            # --- ERGEBNIS-ANZEIGE ---
            st.divider()
            
            # Farbe definieren
            color = "#00CC96" if "High" in prediction else "#AB63FA" if "Mid" in prediction else "#EF553B"
            st.subheader(f"Ergebnis: :{color}[{prediction}]")

            # Wahrscheinlichkeits-Diagramm
            prob_df = pd.DataFrame({
                'Bewertungsgruppe': classes,
                'Wahrscheinlichkeit': probabilities
            })
            
            fig_prob = px.bar(
                prob_df, 
                x='Bewertungsgruppe', 
                y='Wahrscheinlichkeit',
                title="Konfidenz des Modells",
                color='Bewertungsgruppe',
                color_discrete_map={"High (5 ⭐)": "#00CC96", "Mid (3-4 ⭐)": "#AB63FA", "Low (1-2 ⭐)": "#EF553B"},
                text_auto='.2%'
            )
            
            fig_prob.update_layout(yaxis_range=[0, 1]) # Skala fix auf 0-100%
            st.plotly_chart(fig_prob, use_container_width=True)

            st.info(f"**Verarbeiteter Text für die Vorhersage:** *{cleaned_text if cleaned_text else '[Text war nach Reinigung leer]'}*")

else:
    st.warning("⚠️ **Kein Modell gefunden!**")
    st.info("Bitte trainiere zuerst das Modell auf der Seite **'03 Modeling'**.")
