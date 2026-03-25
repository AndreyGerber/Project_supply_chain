from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
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

df = pd.read_json(BASE_RAW +"trustpilot_reviews_production.json")

#zerlegt in numerisches rating
def extract_numeric_rating(svg):
    number = svg.split('-')[1].split('.')[0]
    return number
    
# bereinigt supplier_response    
#ToDo

# bereinigt rewiew_text
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

# vergleichbare Kategorien erstellen
issue_dict = {
    "Delivery Delay": [
        # Englisch
        "delay", "late", "not delivered", "delivery time",
        # Deutsch
        "verspätet", "lieferung", "lieferzeit", "zu spät",
        # Spanisch
        "retraso", "tarde", "entrega", "envío tarde"
    ],
    
    "Damaged Product": [
        "broken", "damage", "damaged",
        "kaputt", "beschädigt", "defekt",
        "roto", "dañado"
    ],
    
    "Wrong Item": [
        "wrong item", "incorrect", "not what i ordered",
        "falsch", "falsches teil",
        "incorrecto", "equivocado"
    ],
    
    "Refund Issue": [
        "refund", "money back", "return problem",
        "rückerstattung", "geld zurück",
        "reembolso", "devolución"
    ],
    
    "Customer Service": [
        "service", "support", "no response",
        "kundenservice", "keine antwort",
        "atención", "sin respuesta"
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

#neue Spalte rating
#df["rating"] = df["rating_svg"].apply(extract_numeric_rating)

#weg mit allen zeilen ohne Komentar 
df = df.dropna(subset=["review_text"])

#neue Spalte review_text_clean
df["review_text_clean"] = df["review_text"].apply(clean_text)

#neue Spalte supplier_response_clean
#df["supplier_response_clean"] = df["supplier_response"].apply(clean_text)

#weg mit Duplikaten
df = df.drop_duplicates()

# neue spalte review_text_clean_advanced
df["review_text_clean_advanced"] = df["review_text_clean"].apply(clean_text_advanced)

#neue Spalte issue_category
df['issue_categories'] = df['review_text'].apply(categorize_issues)

#speichern unter -> wichtig zum später aufrufen
df.to_csv(BASE_CLEAN + "reviews_clean_test.csv", index=False)

print("Clean dataset:", len(df))

#df.head(20)
#df_csv = pd.read_csv(BASE_CLEAN + "reviews_clean.csv")

#df_csv.head(20)

