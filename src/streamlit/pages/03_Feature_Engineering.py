#Einleitung:


#Zeige die Rohdaten(wirklich raw data!!) vor dem feature engineering

import streamlit as st
from pathlib import Path
import json
st.set_page_config(layout="wide")
st.title("📊 Feature Engineering")

file_path = Path("data/raw/trustpilot_reviews_production.json")

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

#st.write(data)

import pandas as pd

df = pd.DataFrame(data)
st.dataframe(df.head(20))
st.write("The scraped raw data consists of eight columns review_text,rating_svg,date,location,supplier_response,verified,company ")


#processed data



#option = st.selectbox(
#    "Choose your step:",
#    ["Encoding","Feature Selection", "Feature Creation"]
#)



#if option == "Encoding":
#    st.write("Ich nutze One-Hot-Encoding")
#    df = pd.get_dummies(df, columns=["gender"])


#if option == "Feature Creation":
#    st.write("Ich erstelle ein neues Feature 'family_size'")
#    df["family_size"] = df["siblings"] + df["parents"]


#if option == "Feature Selection":
#    st.write("Ich ersetze fehlende Werte mit dem Median")
#    df["age"].fillna(df["age"].median(), inplace=True)


#vorher nachher vergleich
#st.write("Nach Feature Engineering")
#st.dataframe(df.head())


#visualisieren warum die features wichtig sind
#import matplotlib.pyplot as plt

#fig, ax = plt.subplots()
#df["family_size"].hist(ax=ax)
#st.pyplot(fig)

#6. Optional: Modell-Impact zeigen

#Wenn du noch einen draufsetzen willst:

#Modell vor und nach Feature Engineering vergleichen
#Accuracy / RMSE anzeigen