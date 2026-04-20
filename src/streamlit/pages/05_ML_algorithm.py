import streamlit as st
import pandas as pd
import os

st.title("Private Datenanalyse - Reviews")

# Der Pfad zu deiner CSV im privaten Projekt
file_path = "src/data/clean/reviews_clean.csv"

if os.path.exists(file_path):
    # Daten laden
    df = pd.read_csv(file_path)
    
    st.success("Datei erfolgreich geladen!")
    
    st.write("### Vorschau: Die ersten 15 Zeilen")
    # .head(15) zeigt genau die gewünschte Anzahl an
    st.dataframe(df.head(15), use_container_width=True) 
    
    # Kurze Info zur Datensatzgröße
    st.info(f"Der Datensatz enthält insgesamt {df.shape[0]} Zeilen und {df.shape[1]} Spalten.")
else:
    st.error(f"Datei nicht gefunden! Bitte prüfe, ob sie hier liegt: {file_path}")
