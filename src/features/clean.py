from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
import nltk
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
import re

nltk.download('stopwords')
nltk.download('wordnet')



###
# OB 17.03.26
# clean json Dataset
###

BASE_RAW = "../data/raw/"
os.makedirs(BASE_RAW, exist_ok=True)

BASE_CLEAN = "../data/clean/"
os.makedirs(BASE_CLEAN, exist_ok=True)

df1 = pd.read_json(BASE_RAW +"trustpilot_reviews_production.json")
df2 = pd.read_json(BASE_RAW +"trustpilot_reviews_production2.json")

df = pd.concat([df1, df2], ignore_index=True)


#zerlegt in numerisches rating
def extract_numeric_rating(svg):
    number = svg.split('-')[1].split('.')[0]
    return number
    

# bereinigt rewiew_text -> überarbeitet in text_preprocessing.py
def clean_text(text):
    # Entferne Zeilenumbrüche und überflüssige Leerzeichen
    cleaned_text = ' '.join(text.split())
    # Entferne HTML-Tags, falls vorhanden
    cleaned_text = BeautifulSoup(cleaned_text, "html.parser").get_text()
    # entferne Emojis und Sonderzeichen
    cleaned_text = ''.join(e for e in cleaned_text if e.isalnum() or e.isspace())

    return cleaned_text


german_stopwords = set(stopwords.words('german'))
english_stopwords = set(stopwords.words('english'))

stop_words = german_stopwords.union(english_stopwords)
custom_stopwords = {
    "sehr", "wirklich", "eigentlich",
    "schon", "noch", "immer",
    "bitte", "danke",
    "mal", "halt", "eben" "grüße", "liebe", "lieber"
}

stop_words = stop_words.union(custom_stopwords)
lemmatizer = WordNetLemmatizer()

def clean_text_advanced(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# Company identifiers were standardized by separating brand names and country-specific 
# domains, enabling more granular analysis across geographic markets.
# trennt die company in company und companie_site
def split_company(company):
    if pd.isna(company):
        return pd.Series([None, None])
    
    parts = company.split("_")
    
    if len(parts) == 2:
        return pd.Series([parts[0], parts[1]])
    else:
        return pd.Series([company, None])
    

# The keyword dictionary is in multiple languages and is including semantically 
# related expressions. This helps to classify and capture a broader range of customer issues across 
# international markets.
# vergleichbare Kategorien erstellen
issue_dict = {

    "Delivery Delay": [
        # EN
        "delay", "late", "not delivered", "delivery time", "shipping delay", "slow delivery",
        # DE
        "verspätet", "lieferung", "lieferzeit", "zu spät", "lange gewartet",
        # ES
        "retraso", "tarde", "entrega", "envío tarde", "demora",
        # FR
        "retard", "livraison tardive", "délai", "expédition lente",
        # IT
        "ritardo", "consegna in ritardo", "spedizione lenta", "tempo di consegna",
        # NL (Belgien)
        "vertraging", "late levering", "trage verzending"
    ],

    "Damaged Product": [
        # EN
        "broken", "damage", "damaged", "defective", "scratched", "faulty",
        # DE
        "kaputt", "beschädigt", "defekt", "zerkratzt",
        # ES
        "roto", "dañado", "defectuoso",
        # FR
        "cassé", "endommagé", "défectueux",
        # IT
        "rotto", "danneggiato", "difettoso",
        # NL
        "kapot", "beschadigd", "defect"
    ],

    "Wrong Item": [
        # EN
        "wrong item", "incorrect", "not what i ordered", "different product",
        # DE
        "falsch", "falsches teil", "nicht bestellt",
        # ES
        "incorrecto", "equivocado", "producto incorrecto",
        # FR
        "mauvais produit", "incorrect", "pas commandé",
        # IT
        "prodotto sbagliato", "errato", "non ordinato",
        # NL
        "verkeerd product", "fout", "niet besteld"
    ],

    "Refund Issue": [
        # EN
        "refund", "money back", "return problem", "no refund", "refund delay",
        # DE
        "rückerstattung", "geld zurück", "keine rückzahlung",
        # ES
        "reembolso", "devolución", "sin reembolso",
        # FR
        "remboursement", "pas de remboursement",
        # IT
        "rimborso", "nessun rimborso",
        # NL
        "terugbetaling", "geen terugbetaling"
    ],

    "Customer Service": [
        # EN
        "service", "support", "no response", "no reply", "unhelpful", "bad service",
        # DE
        "kundenservice", "keine antwort", "schlechter service",
        # ES
        "atención", "sin respuesta", "mal servicio",
        # FR
        "service client", "aucune réponse", "mauvais service",
        # IT
        "servizio clienti", "nessuna risposta", "servizio scarso",
        # NL
        "klantenservice", "geen antwoord", "slechte service"
    ],

    "Delivery Issue": [
        # EN
        "delivery problem", "shipping issue", "package lost", "missing package",
        # DE
        "lieferproblem", "paket verloren", "nicht angekommen",
        # ES
        "problema entrega", "paquete perdido",
        # FR
        "problème livraison", "colis perdu",
        # IT
        "problema consegna", "pacco perso",
        # NL
        "leveringsprobleem", "pakket verloren"
    ],

    "Product Availability": [
        # EN
        "out of stock", "not available", "unavailable",
        # DE
        "nicht verfügbar", "ausverkauft",
        # ES
        "no disponible", "agotado",
        # FR
        "indisponible", "en rupture",
        # IT
        "non disponibile", "esaurito",
        # NL
        "niet beschikbaar", "uitverkocht"
    ]
}

def categorize_issues(text):
    if pd.isna(text):
        return []
    
    text = text.lower()
    found_categories = []
    
    for category, keywords in issue_dict.items():
        for kw in keywords:
            if re.search(rf"\b{kw}\b", text):
                found_categories.append(category)
                break  # verhindert doppelte Kategorie
    
    return found_categories if found_categories else ["Other"]

# A weighted keyword approach is used to capture not only the presence but also 
# the intensity of specific supply chain issues within customer reviews.
# Limitation: Keyword frequency does not always reflect true importance, 
# as repetition may be stylistic rather than indicative of severity.
# gewichtetes Vorkommen von Kategorien in einem Text
def categorize_issues_weighted(text):
    text = text.lower()
    matches = []

    for category, keywords in issue_dict.items():
        for kw in keywords:
            if re.search(rf"\b{kw}\b", text):
                matches.append(category)

    return Counter(matches)

#neue Spalte rating
#df["rating"] = df["rating_svg"].apply(extract_numeric_rating)

#weg mit allen zeilen ohne Komentar 
df = df.dropna(subset=["review_text"])

#korrektur des company eintrags
df["company"] = df["company"].str.replace(r"-de$", "_de", regex=True)
#df["company"] = df["company"].str.replace(r"-(?=[^_]*$)", "_", regex=True)

#neue Spalte review_text_clean
df["review_text_clean"] = df["review_text"].apply(clean_text)

#neue Spalte supplier_response_clean
#df["supplier_response_clean"] = df["supplier_response"].apply(clean_text)

#weg mit Duplikaten
df = df.drop_duplicates()

# teilen der company Spalte in company und company_site
df[['company', 'company_site']] = df['company'].apply(split_company)

# neue spalte review_text_clean_advanced
df["review_text_clean_advanced"] = df["review_text_clean"].apply(clean_text_advanced)

#neue Spalte issue_category
df['issue_categories'] = df['review_text'].apply(categorize_issues_weighted)

#speichern unter -> wichtig zum später aufrufen
df.to_csv(BASE_CLEAN + "reviews_clean_big.csv", index=False)

print("Clean dataset:", len(df))

#df.head(20)
#df_csv = pd.read_csv(BASE_CLEAN + "reviews_clean.csv")

#df_csv.head(20)

