import re
import string
from typing import List
from bs4 import BeautifulSoup
import pandas as pd


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from src.utils.text_translation import ReviewTranslator

# Download NLTK data (einmalig) später in config.py auslagern
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')  # wichtig für mehrsprachige Lemmatization

# Stopwords für mehrere Sprachen kombinieren
STOPWORDS = set(stopwords.words('english')) 
LEMMATIZER = WordNetLemmatizer()

#sicherstellen, dass alle Spalten da sind, damit die Funktionen nicht crashen
def ensure_schema(df):

    defaults = {
        "verified": 0,
        "supplier_response": None,
        "review_text_en": df.get("review_text", "")
    }

    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val

    return df

#Sentiment Analyse(sehr starkes neues feature sentiment!!!)
analyzer = SentimentIntensityAnalyzer()
def get_sentiment(text):
    if pd.isna(text):
        return None
    return analyzer.polarity_scores(text)["compound"]


def clean_text(text: str) -> str:
    """
    Bereinigt einen Text:
    - Entfernt HTML-Tags
    - Lowercase
    - Entfernt Emojis und Sonderzeichen
    - Entfernt Zahlen
    - Lemmatization
    - Entfernt Stopwords 
      (aber behält Negationen wie "not", "no", "never" für sentiment analysis!)
    """
    if not isinstance(text, str):
        return ""

    # 1. HTML entfernen (früh!)
    text = BeautifulSoup(text, "html.parser").get_text()

    # 2. lowercase
    text = text.lower()

    # 3. Emojis / Sonderzeichen (weniger aggressiv)
    text = re.sub(r"[^\w\s!?]", " ", text)

    # 4. Zahlen entfernen
    text = re.sub(r"\d+", " ", text)

    # 5. Tokenize
    words = text.split()

    # 6. Lemmatization FIRST
    words = [LEMMATIZER.lemmatize(word) for word in words]

    # 7. Stopwords entfernen (vorsichtig!)
    words = [w for w in words if w not in STOPWORDS or w in ["not", "no", "never"]]

    return " ".join(words)


def add_structured_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fügt zusätzliche strukturierte Features hinzu:
    - review_length
    - sentiment (VADER Sentiment Score)
    - has_negation (ob die Review Negationen enthält, wichtig für Sentiment Analysis)
    - verified (bereits boolean)
    """
    df = df.copy()
    df = ensure_schema(df)
    df["review_length"] = df["review_text"].astype(str).apply(lambda x: len(x.split()))
    df["sentiment"] = df["review_text"].apply(get_sentiment)
    #eventuell nur auf die englischen reviews anwenden, 
    # da vaderSentiment hauptsächlich für englisch optimiert ist.
    df["has_negation"] = df["review_text_en"].str.contains("not|no|never", case=False)
    df["verified"] = df["verified"].astype(int)

    return df