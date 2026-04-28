import re
import string
from typing import List
from bs4 import BeautifulSoup
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK data (einmalig) später in config.py auslagern
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')  # wichtig für mehrsprachige Lemmatization

# Stopwords für mehrere Sprachen kombinieren
STOPWORDS = set(stopwords.words('english')) \
    | set(stopwords.words('german')) 

LEMMATIZER = WordNetLemmatizer()

def clean_text(text: str) -> str:
    """
    Bereinigt einen Text:
    - Lowercase
    - Entfernt Sonderzeichen und Zahlen
    - Entfernt Stopwords
    - Entfernt HTML-Tags
    - Entfernt Emojis und Sonderzeichen
    - Lemmatization
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)  # Satzzeichen entfernen
    text = re.sub(r"\d+", " ", text)  # Zahlen entfernen
    # Entferne HTML-Tags, falls vorhanden
    text = BeautifulSoup(text, "html.parser").get_text()
    # entferne Emojis und Sonderzeichen
    text = ''.join(e for e in text if e.isalnum() or e.isspace())
    words = text.split()
    #Lemmatization nicht richtig gut für deutsch, da WordNet hauptsächlich für englisch optimiert ist.
    # eventuell keine lemmatization ->entscheiden
    words = [LEMMATIZER.lemmatize(word) for word in words if word not in STOPWORDS]
    return " ".join(words)

def add_structured_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fügt zusätzliche strukturierte Features hinzu:
    - review_length
    - has_response
    - verified (bereits boolean)
    """
    df = df.copy()
    df["review_length"] = df["review_text"].astype(str).apply(lambda x: len(x.split()))
    df["has_response"] = df["supplier_response"].notna().astype(int)
    df["verified"] = df["verified"].astype(int)

    return df