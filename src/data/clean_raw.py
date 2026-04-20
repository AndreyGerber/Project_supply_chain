from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
import nltk
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
import re


###
# OB 17.04.26
# clean json Dataset
###

BASE_RAW = "../data/raw/"
os.makedirs(BASE_RAW, exist_ok=True)

BASE_CLEAN = "../data/clean/"
os.makedirs(BASE_CLEAN, exist_ok=True)


#------------------ Funktionen ----------------

def split_company(company):
    if pd.isna(company):
        return pd.Series([None, None])
    
    #falls die company Spalte schon den Domainnamen enthält, z.B. "Amazon.de"
    if "." in company:
        parts = company.split(".")
        if len(parts) == 2:
            return pd.Series([parts[0], parts[1]])
        if len(parts) == 3:
            return pd.Series([parts[1], parts[2]])
        
    #falls die company Spalte nur den Firmennamen mit Unterstrich enthält, z.B. "Amazon_de"
    if "_" in company:
        parts = company.split("_")
        if len(parts) == 2:
            return pd.Series([parts[0], parts[1]])
        if len(parts) == 3:
            return pd.Series([parts[1], parts[2]])
    #aktuell gibt es eine sonderausnahme
    if "-de" in company:
        parts = company.split("-")
        if len(parts) == 2:
            return pd.Series([parts[0], None])
        else:
            return pd.Series([company, None])
    else:
            return pd.Series([company, None])


#Ablauf für die Bereinigung der 3 Datensätze:
# 1 den neuen Datensatz laden 
df1 = pd.read_json(BASE_RAW +"trustpilot_autosave.json")
df1 = df1.copy()
# 2 alle nicht benötigten Spalten entfernen
df1.drop(columns=["review_id", "response_date"], inplace=True)
# 3 alle Spaltennamen in ein einheitliches Format bringen
df1.rename(columns={
    "rating": "rating",
    "review_text": "review_text",
    "review_date": "date",
    "location": "location",
    "supplier_response": "supplier_response",
    "verified": "verified",
    "company": "company"
}, inplace=True)

# 4 company Namen extrahieren und bereinigen

df1[["company", "domain"]] = df1["company"].apply(split_company)


# 6 die beiden alten Datensätze laden
df2 = pd.read_json(BASE_RAW +"trustpilot_reviews_production.json")
df3 = pd.read_json(BASE_RAW +"trustpilot_reviews_production2.json")

# 7 die beiden zusammenbringen 
df4 = pd.concat([df2, df3], ignore_index=True)

# 8 die Spaltennamen in ein einheitliches Format bringen
df4.rename(columns={
    "rating_svg": "rating",
    "review_text": "review_text",
    "date": "date",
    "location": "location",
    "supplier_response": "supplier_response",
    "verified": "verified",
    "company": "company"
}, inplace=True)
# ratings müssen von svg in float umgewandelt werden
df4["rating"] = df4["rating"].astype(float)

# 9 die company Namen extrahieren und bereinigen
df4[["company", "domain"]] = df4["company"].apply(split_company)

# 10 die beiden Datensätze alt und neu zusammenbringen
df = pd.concat([df1, df4], ignore_index=True)

# alles in einer einheitlichen Strucktur abspeichern

def strucktured_raw(df: pd.DataFrame) -> pd.DataFrame:
    
    return {
        
        "rating": df.rating,
        "review_text": df.review_text,
        "date": df.date,
        "location": df.location,
        "supplier_response": df.supplier_response,
        "verified": df.verified,
        "company": df.company,
        "domain": df.domain
    }


df.apply(strucktured_raw, axis=1 ).to_json(BASE_RAW + "trustpilot_reviews_concat.json", orient="records", indent=2)

